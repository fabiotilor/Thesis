import numpy as np
import torch
from einops import rearrange
import torch.nn.functional as F

from vggt4d.models.vggt4d import VGGTFor4D
from vggt4d.masks.dynamic_mask import (
    adaptive_multiotsu_variance,
    cluster_attention_maps,
    batch_extract_dyn_map,
)
from vggt4d.masks.refine_dyn_mask import RefineDynMask
from vggt.utils.load_fn import load_and_preprocess_images

from vggt4d.models.vggt4d import VGGTFor4D
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def inference(model: VGGTFor4D, images: torch.Tensor, dyn_masks: torch.Tensor = None, query_points: torch.Tensor = None) -> tuple[dict, dict, dict, list]:
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[
        0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            predictions, qk_dict, enc_feat, agg_tokens_list = model(
                images, dyn_masks=dyn_masks, query_points=query_points)

    # Offload attention dictionaries to CPU to free VRAM for downstream stages
    for key in qk_dict:
        if isinstance(qk_dict[key], torch.Tensor):
            qk_dict[key] = qk_dict[key].cpu()

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].to(device="cpu", dtype=torch.float32) \
                .numpy().squeeze(0)  # remove batch dimension

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(
        depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # save memory intermediate aggregated tokens for tracking
    for i in range(len(agg_tokens_list)):
        if i not in [4, 11, 17, 23]:
            agg_tokens_list[i] = None

    torch.cuda.empty_cache()

    n_img = images.shape[0]
    pred_extrinsic = predictions["extrinsic"]
    pad = np.zeros((n_img, 1, 4))
    pad[:, 0, -1] = 1
    pred_extrinsic = np.concatenate([pred_extrinsic, pad], axis=1)
    pred_cam2world = np.linalg.inv(pred_extrinsic)
    predictions["cam2world"] = pred_cam2world
    predictions["depth"] = predictions["depth"].squeeze(-1)
    return predictions, qk_dict, enc_feat.detach().cpu(), agg_tokens_list


def organize_qk_dict(qk_dict, n_img):
    global_q = qk_dict.pop("global_q")
    global_k = qk_dict.pop("global_k")
    frame_q = qk_dict.pop("frame_q")
    frame_k = qk_dict.pop("frame_k")

    n_tok = global_q.shape[-2] // n_img

    patch_start_idx = 5

    global_q = rearrange(
        global_q, "n_layer 1 1 n_head (n_img n_tok) c -> n_img n_layer n_head n_tok c", n_img=n_img, n_tok=n_tok)
    global_k = rearrange(
        global_k, "n_layer 1 1 n_head (n_img n_tok) c -> n_img n_layer n_head n_tok c", n_img=n_img, n_tok=n_tok)

    global_cam_q = global_q[..., 0:1, :]
    global_cam_k = global_k[..., 0:1, :]
    global_reg_q = global_q[..., 1:patch_start_idx, :]
    global_reg_k = global_k[..., 1:patch_start_idx, :]
    global_tok_q = global_q[..., patch_start_idx:, :]
    global_tok_k = global_k[..., patch_start_idx:, :]

    frame_q = rearrange(
        frame_q, "n_layer 1 n_img n_head n_tok c -> n_img n_layer n_head n_tok c", n_img=n_img, n_tok=n_tok)
    frame_k = rearrange(
        frame_k, "n_layer 1 n_img n_head n_tok c -> n_img n_layer n_head n_tok c", n_img=n_img, n_tok=n_tok)

    frame_cam_q = frame_q[..., 0:1, :]
    frame_cam_k = frame_k[..., 0:1, :]
    frame_reg_q = frame_q[..., 1:patch_start_idx, :]
    frame_reg_k = frame_k[..., 1:patch_start_idx, :]
    frame_tok_q = frame_q[..., patch_start_idx:, :]
    frame_tok_k = frame_k[..., patch_start_idx:, :]

    return {
        "global_cam_q": global_cam_q,
        "global_cam_k": global_cam_k,
        "global_reg_q": global_reg_q,
        "global_reg_k": global_reg_k,
        "global_tok_q": global_tok_q,
        "global_tok_k": global_tok_k,
        "frame_cam_q": frame_cam_q,
        "frame_cam_k": frame_cam_k,
        "frame_reg_q": frame_reg_q,
        "frame_reg_k": frame_reg_k,
        "frame_tok_q": frame_tok_q,
        "frame_tok_k": frame_tok_k,

        "global_q": global_tok_q,
        "global_k": global_tok_k,
        "frame_q": frame_tok_q,
        "frame_k": frame_tok_k,
    }


def run_vggt4d_3stage_inference(model, frame_paths, device):
    """
    Run the full 3-stage VGGT4D pipeline on an ordered sequence of paths.

    The caller decides the interleaving; here we just process T images.

    Returns
    -------
    dict with numpy arrays (leading dim = T = len(frame_paths)):
        world_points      (T, H, W, 3)
        world_points_conf (T, H, W)
        cam2world         (T, 4, 4)
        extrinsic         (T, 3, 4)
        intrinsic         (T, 3, 3)
        depth             (T, H, W)
        depth_conf        (T, H, W)
        dynamic_masks     (T, H, W) bool   — True = dynamic pixel
    """
    imgs = load_and_preprocess_images(
        [str(p) for p in frame_paths]
    ).to(device)
    n_img, _, h_img, w_img = imgs.shape

    # Stage 1 — depth + dynamic map extraction
    print(f"      [Stage 1] depth + dynamic map  T={n_img}")
    predictions1, qk_dict, enc_feat, agg_tokens_list = inference(
        model, imgs)
    del agg_tokens_list

    qk_dict = organize_qk_dict(qk_dict, n_img)
    dyn_maps = batch_extract_dyn_map(qk_dict, imgs)

    h_tok, w_tok = h_img // 14, w_img // 14
    feat_map = rearrange(
        enc_feat, "n_img (h w) c -> n_img h w c", h=h_tok, w=w_tok)
    norm_dyn_map, _ = cluster_attention_maps(feat_map, dyn_maps)

    upsampled_map = F.interpolate(
        rearrange(norm_dyn_map, "n_img h w -> n_img 1 h w"),
        size=(h_img, w_img), mode="bilinear", align_corners=False,
    )
    upsampled_map = rearrange(upsampled_map, "n_img 1 h w -> n_img h w")

    thres = adaptive_multiotsu_variance(upsampled_map.cpu().numpy())
    dyn_masks = upsampled_map > thres

    del enc_feat, feat_map, qk_dict, dyn_maps, norm_dyn_map, upsampled_map
    torch.cuda.empty_cache()

    # Stage 2 — refine extrinsics using dynamic masks
    print("      [Stage 2] refine extrinsics")
    predictions2, _, _, _ = inference(
        model, imgs, dyn_masks.to(device))

    # Stage 3 — refine dynamic masks geometrically
    print("      [Stage 3] refine dynamic masks")
    torch.cuda.empty_cache()

    refiner = RefineDynMask(
        imgs,
        torch.tensor(predictions1["depth"]).to(device),
        dyn_masks.to(device),
        torch.tensor(predictions2["cam2world"]).float().to(device),
        torch.tensor(predictions1["intrinsic"]).to(device),
        device,
    )
    refined_mask = refiner.refine_masks()
    del refiner, imgs, dyn_masks
    torch.cuda.empty_cache()

    if isinstance(refined_mask, torch.Tensor):
        refined_mask_np = refined_mask.cpu().numpy().astype(bool)
    else:
        refined_mask_np = np.asarray(refined_mask, dtype=bool)

    return {
        "world_points":      predictions2["world_points_from_depth"],
        "world_points_conf": predictions2["world_points_conf"],
        "cam2world":         predictions2["cam2world"],
        "extrinsic":         predictions2["extrinsic"],
        "intrinsic":         predictions2["intrinsic"],
        "depth":             predictions2["depth"],
        "depth_conf":        predictions2["depth_conf"],
        "dynamic_masks":     refined_mask_np,
    }
