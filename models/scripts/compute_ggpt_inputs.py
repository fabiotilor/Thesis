import os
import sys
import argparse
import glob
import time
import json
import numpy as np
import torch
import shutil
import cv2
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval_config import SUBJECT_NAMES, SUBJECT_BY_CODE, DATASET_BASE_ROOT
from sfm.dataloader.dexycb_dataset import DexYCBDataset
from feedforward import FeedForward_Model
from sfm.sfm_func import run_sfm
from utils.geometry import unproject_depth_map_to_point_map_torch, project_point_map_to_depth_map_torch
from utils.points import umeyama_alignment

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED_MODELS = ['mast3r', 'vggt-point', 'vggt-point-v2']


def move_to_device(batch, device):
    if isinstance(batch, list):
        return [move_to_device(b, device) for b in batch]
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch


def prepare_batch(batch, output_width=518):
    batch_ffres = []
    for batch_ in batch:
        batch_ffres_ = {}
        for key, value in batch_.items():
            if isinstance(value, str):
                batch_ffres_[key] = value
            elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
                batch_ffres_[key] = value[0]
            elif key == 'img_names' and isinstance(value, list):
                if len(value) == 1 and isinstance(value[0], list):
                    batch_ffres_[key] = value[0]
                else:
                    batch_ffres_[key] = value

        # Ensure inputs are tensors and remove batch dimension if present
        for key in ['images', 'depths', 'intrinsics', 'extrinsics', 'point_masks']:
            if key in batch_:
                val = batch_[key]
                if isinstance(val, list):
                    # For lists of numpy arrays or tensors
                    if isinstance(val[0], np.ndarray):
                        val = [torch.from_numpy(v) for v in val]
                    val = torch.stack(val, dim=0)

                if isinstance(val, torch.Tensor) and val.ndim > 0:
                    # If batch size was 1, remove the batch dimension: (1, V, ...) -> (V, ...)
                    if val.shape[0] == 1 and val.ndim >= 4 and key in ['images', 'depths', 'point_masks']:
                        val = val.squeeze(0)
                    elif val.shape[0] == 1 and val.ndim >= 3 and key in ['intrinsics', 'extrinsics']:
                        val = val.squeeze(0)
                batch_ffres_[key] = val

        in_h, in_w = batch_ffres_['images'][0].shape[:2]
        batch_ffres_['images'] = preprocess(batch_ffres_['images'], output_width=output_width).to(
            batch_ffres_['images'][0].device)
        ff_h, ff_w = batch_ffres_['images'].shape[1:3]

        if 'depths' in batch_ffres_:
            if batch_ffres_['depths'].shape[1] != ff_h or batch_ffres_['depths'].shape[2] != ff_w:
                batch_ffres_['depths'] = torch.nn.functional.interpolate(batch_ffres_['depths'].unsqueeze(1),
                                                                         size=(ff_h, ff_w),
                                                                         mode='nearest-exact').squeeze(1)

        if 'point_masks' in batch_ffres_:
            if batch_ffres_['point_masks'].shape[1] != ff_h or batch_ffres_['point_masks'].shape[2] != ff_w:
                batch_ffres_['point_masks'] = torch.nn.functional.interpolate(
                    batch_ffres_['point_masks'].unsqueeze(1).float(), size=(ff_h, ff_w), mode='nearest-exact').squeeze(
                    1).bool()

        if 'intrinsics' in batch_ffres_:
            Ks = batch_ffres_['intrinsics'].clone()
            Ks[:, 0, 0] *= ff_w / in_w
            Ks[:, 1, 1] *= ff_h / in_h
            Ks[:, 0, 2] = (ff_w - 1) / 2.0
            Ks[:, 1, 2] = (ff_h - 1) / 2.0
            batch_ffres_['intrinsics'] = Ks

        # Determine masks if not provided
        if 'point_masks' not in batch_ffres_ and 'depths' in batch_ffres_:
            batch_ffres_['point_masks'] = (batch_ffres_['depths'] > 0)

        if 'depths' in batch_ffres_:
            batch_ffres_['points'] = unproject_depth_map_to_point_map_torch(depth_map=batch_ffres_['depths'],
                                                                            extrinsics_cam=batch_ffres_['extrinsics'],
                                                                            intrinsics_cam=batch_ffres_['intrinsics'])
        elif 'points' in batch_:
            # Handle case where points are already present
            pass

        batch_ffres.append(batch_ffres_)
    return batch_ffres


def preprocess(images, output_width=518):
    # images: (V, H, W, 3) or (H, W, 3)
    if images.ndim == 3:
        images = images.unsqueeze(0)

    V, H, W, C = images.shape
    if C != 3 and H == 3:  # Handle (V, 3, H, W)
        images = images.permute(0, 2, 3, 1)
        V, H, W, C = images.shape

    new_w = output_width
    new_h = int(H * (new_w / W))

    if images.dtype == torch.uint8:
        images = images.float() / 255.0

    images = images.permute(0, 3, 1, 2)  # (V, 3, H, W)
    images = torch.nn.functional.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return images.permute(0, 2, 3, 1)  # (V, H, W, 3)


class DummyConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, DummyConfig(**v) if isinstance(v, dict) else v)

    def get(self, key, default=None):
        return getattr(self, key, default)


def main():
    parser = argparse.ArgumentParser(description="Compute GGPT inputs for DexYCB")
    parser.add_argument("--model", type=str, default="vggt-point", choices=SUPPORTED_MODELS)
    parser.add_argument("--matcher", type=str, default="romav2-base")
    parser.add_argument("--all", action="store_true", help="Process all subjects")
    for i in range(1, 11):
        parser.add_argument(f"--{i:02d}", action="store_true", help=f"Process subject {i:02d}")
    parser.add_argument("--views", nargs='+', type=int, default=[2, 3, 4], help="Number of views to process")
    args = parser.parse_args()

    # Determine subjects
    subjects_to_run = []
    for i in range(1, 11):
        if getattr(args, f"{i:02d}"):
            subjects_to_run.append(SUBJECT_BY_CODE[f"{i:02d}"])
    if not subjects_to_run:
        subjects_to_run = [SUBJECT_BY_CODE["01"]]

    # Load Model
    ff_cfg = DummyConfig(model=args.model)
    ff_model = FeedForward_Model(ff_cfg).to(DEVICE)
    ff_model.eval()

    match_models = None
    sfm_cfg = None
    if args.model != 'mast3r':
        from matching import Match_Models
        match_models = Match_Models(DummyConfig(matcher=args.matcher)).to(DEVICE)
        sfm_cfg = DummyConfig(dlt_config={'max_epipolar_error': 1.0, 'min_tri_angle': 1.0})

    output_width = 512 if args.model == 'mast3r' else 518

    for subject_full in tqdm(subjects_to_run, desc="Subjects"):
        dataset_root = os.path.join(DATASET_BASE_ROOT, subject_full)

        for nviews in args.views:
            print(f"\nProcessing {subject_full} - {nviews} views")

            # Discover available views and select the first nviews
            available_views = sorted([d for d in os.listdir(dataset_root) if d.startswith("view_")])
            view_names = available_views[:nviews]

            dataset = DexYCBDataset(root=dataset_root, name=subject_full, view_names=view_names)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

            out_dir = os.path.join("ggpt_inputs", args.model, subject_full, f"{nviews}views")
            os.makedirs(out_dir, exist_ok=True)

            for batch_idx, batch_cpu in enumerate(dataloader):
                # ── Extract metadata (seq_name etc.) from raw batch ──────────
                raw_batch = batch_cpu  # keep raw reference
                frame_name = None
                if 'seq_name' in raw_batch:
                    val = raw_batch['seq_name']
                    frame_name = val[0] if isinstance(val, list) else val
                if not frame_name:
                    frame_name = f"frame_{batch_idx:05d}"

                out_path = os.path.join(out_dir, f"{frame_name}.npz")
                if os.path.exists(out_path):
                    print(f"Skipping {frame_name}, already computed.")
                    continue

                print(f"  Frame {frame_name}...")

                if args.model == 'mast3r':
                    # ── MASt3R path: pass RAW images, let model handle resizing ──
                    # Extract raw images from batch and squeeze DataLoader batch dim
                    raw_images = raw_batch['images']
                    if isinstance(raw_images, torch.Tensor) and raw_images.ndim == 5:
                        raw_images = raw_images.squeeze(0)  # (1,V,H,W,3) → (V,H,W,3)
                    raw_images = raw_images.to(DEVICE)

                    # Extract GT camera params at native sensor resolution
                    gt_K = raw_batch['intrinsics']
                    if isinstance(gt_K, torch.Tensor) and gt_K.ndim == 4:
                        gt_K = gt_K.squeeze(0)
                    gt_extr = raw_batch['extrinsics']
                    if isinstance(gt_extr, torch.Tensor) and gt_extr.ndim == 4:
                        gt_extr = gt_extr.squeeze(0)

                    gt_depths = raw_batch.get('depths', None)
                    if gt_depths is not None:
                        if isinstance(gt_depths, torch.Tensor) and gt_depths.ndim == 4:
                            gt_depths = gt_depths.squeeze(0)
                    gt_masks = (gt_depths > 0) if gt_depths is not None else None

                    with torch.no_grad():
                        # MASt3R forward: saves images to disk → load_images(size=512) → alignment
                        # preprocessed=False lets the model's internal preprocess() handle it
                        ff_outputs = ff_model(raw_images, preprocessed=False)

                    # MASt3R produces geo_points at its internal resolution (384×512)
                    mast3r_pts = ff_outputs['geo_points']
                    mask = (ff_outputs['points_conf'] > 0.0)

                    geo_pts = mast3r_pts.cpu().numpy()
                    geo_masks = mask.cpu().numpy()

                    # MASt3R's output resolution
                    V_m, H_m, W_m, _ = ff_outputs['points'].shape

                    # Use predicted camera params for the refinement stage to ensure consistency
                    # GGPT expects these to match the unprojection of the pointmaps
                    est_K = ff_outputs['intrinsics'].cpu().numpy()
                    est_poses = ff_outputs['extrinsics'].cpu().numpy()

                    # Resize GT masks to model resolution for saving
                    if gt_masks is not None:
                        gt_masks_resized = torch.nn.functional.interpolate(
                            gt_masks.unsqueeze(1).float().to(DEVICE), size=(H_m, W_m),
                            mode='nearest-exact').squeeze(1).bool()
                    else:
                        gt_masks_resized = torch.ones(V_m, H_m, W_m, dtype=torch.bool, device=DEVICE)

                    frame_idx = int(frame_name.replace("frame_", ""))

                    # Compute GT points from raw depths if available
                    if gt_depths is not None and gt_extr is not None and gt_K is not None:
                        gt_pts_tensor = unproject_depth_map_to_point_map_torch(
                            gt_depths, gt_extr, gt_K
                        )
                        gt_pts_np = gt_pts_tensor.cpu().numpy().reshape(-1, 3)
                    else:
                        gt_pts_np = np.zeros((V_m * H_m * W_m, 3))

                    np.savez(out_path,
                             ff_points=ff_outputs['points'].cpu().numpy(),
                             ff_points_conf=ff_outputs['points_conf'].cpu().numpy(),
                             images_ff=ff_outputs['images_ff'].cpu().numpy(),

                             geo_points=geo_pts,
                             geo_point_masks=geo_masks,

                             frame_idx=frame_idx,
                             Ks=est_K,  # Use est_K here so refinement is consistent
                             R_ts=est_poses,  # Use est_poses here
                             gt_pts=gt_pts_np,  # Save original GT for metric
                             masks_2d=gt_masks_resized.cpu().numpy(),

                             est_poses=est_poses,
                             est_intrinsics=est_K,

                             pointmaps=ff_outputs['points'].cpu().numpy(),
                             pointmaps_confs=ff_outputs['points_conf'].cpu().numpy(),
                             )

                else:
                    # ── Non-MASt3R path (VGGT etc.): use prepare_batch as before ──
                    batch_cpu_list = [batch_cpu]
                    batch_rawres = move_to_device(batch_cpu_list, DEVICE)
                    batch_ffres = prepare_batch(batch_rawres, output_width=output_width)
                    batch_ffres = batch_ffres[0]

                    with torch.no_grad():
                        imgs_in = batch_ffres['images'].permute(0, 3, 1, 2)
                        ff_outputs = ff_model(imgs_in, preprocessed=True, gt_dict=batch_ffres)

                    sfm_outputs = run_sfm(batch_rawres[0]['images'], ff_outputs, match_models, sfm_cfg, gt=batch_ffres,
                                          output_dir=None)

                    if not sfm_outputs['points_success']:
                        print(f"    WARNING: SfM failed for {frame_name}")
                        geo_pts = np.zeros_like(ff_outputs['points'].cpu().numpy())
                        geo_masks = np.zeros_like(ff_outputs['points_conf'].cpu().numpy(), dtype=bool)
                    else:
                        geo_pts = sfm_outputs['points'].cpu().numpy()
                        geo_masks = sfm_outputs['point_masks'].cpu().numpy()

                    frame_idx = int(frame_name.replace("frame_", ""))

                    np.savez(out_path,
                             ff_points=ff_outputs['points'].cpu().numpy(),
                             ff_points_conf=ff_outputs['points_conf'].cpu().numpy(),
                             images_ff=batch_ffres['images'].cpu().numpy(),

                             geo_points=geo_pts,
                             geo_point_masks=geo_masks,

                             frame_idx=frame_idx,
                             Ks=batch_ffres['intrinsics'].cpu().numpy(),
                             R_ts=batch_ffres['extrinsics'].cpu().numpy(),
                             gt_pts=batch_ffres['points'].cpu().numpy().reshape(-1, 3),
                             masks_2d=batch_ffres['point_masks'].cpu().numpy(),

                             est_poses=ff_outputs.get('extrinsics',
                                                      torch.eye(4).unsqueeze(0).repeat(nviews, 1, 1)).cpu().numpy(),
                             est_intrinsics=ff_outputs.get('intrinsics', batch_ffres['intrinsics']).cpu().numpy(),

                             pointmaps=ff_outputs['points'].cpu().numpy(),
                             pointmaps_confs=ff_outputs['points_conf'].cpu().numpy(),
                             )


if __name__ == "__main__":
    main()
