import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import time
import glob
from tqdm import tqdm
import rerun as rr
from omegaconf import OmegaConf
from hydra import initialize, compose
from hydra.utils import instantiate

import sys

# Add current directory to path so local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ggpt.dataloader.eval_dataset import EvalDataset
from ggpt.model.base import BasePredictor
from utils.common import move_to_device
from utils.points import aggregate_chunks

try:
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'scripts')
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    from eval_config import RERUN_ADDR, RERUN_EYE_UP, SUBJECT_BY_CODE, DATASET_BASE_ROOT
    from eval_config import get_subject_by_code, get_dataset_root_for_subject, get_view_config, \
        get_pair_name_for_subject, get_dataset_config
    from utils.rerun_logging import configure_rerun_view_defaults, log_pointcloud, init_recording
    from utils.camera_utils import get_rgb_path
    from utils.optical_flow import compute_static_mask
except ImportError:
    RERUN_ADDR = "rerun+http://127.0.0.1:9876/proxy"
    RERUN_EYE_UP = [0, -1, 0]
    configure_rerun_view_defaults = None
    log_pointcloud = None
    init_recording = None
    get_rgb_path = None
    compute_static_mask = None


def parse_args():
    parser = argparse.ArgumentParser(description="GGPT Refinement Script")
    parser.add_argument("--subject", type=str, default="all", help="Subject code (e.g. 01) or 'all'")
    parser.add_argument("--views", nargs="+", type=int, default=[2, 3, 4], help="Number of views to use (e.g. 2 3 4)")
    parser.add_argument("--ckpt", type=str, default="ckpts/model.step228000.pth", help="Path to GGPT checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", type=str, default="vggt",
                        help="Name of the base model being refined (e.g. vggt, mast3r, pi3)")
    parser.add_argument("--no_rerun", action="store_true", help="Disable Rerun logging")
    parser.add_argument("--dataset", type=str, choices=["dex-ycb", "hi4d", "monofusion"], default="dex-ycb",
                        help="Dataset to use")
    parser.add_argument("--base_input_dir", type=str, default=None, help="Base directory for inputs")
    return parser.parse_args()


def run_refinement(model, cfg, scene_dir, args, subj_code, n_views):
    # 0. Setup Rerun Recording for this specific (Model, Subject, View-Count) trio
    if not args.no_rerun:
        if init_recording:
            init_recording(subj_code, n_views, model_name=args.model)
        if configure_rerun_view_defaults:
            configure_rerun_view_defaults("world", RERUN_EYE_UP)
        else:
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # 1. Setup Dataset for the specific scene
    input_abs_path = os.path.abspath(scene_dir)
    data_dict = {"vggt_input": input_abs_path}

    # We use EvalDataset to load the chunks
    chunk_size = cfg.valdataset_configs.chunk_size if 'chunk_size' in cfg.valdataset_configs else 0.2
    dataset = EvalDataset(data_dict=data_dict, chunk_size=chunk_size)

    if len(dataset) == 0:
        # print(f"[WARN] No valid scene found in {input_abs_path}")
        return

    unnormalize_func = dataset.unnormalize_pts

    for i in range(len(dataset)):
        scene_chunks, scene = dataset[i]
        scene_name = scene['scene_name']
        subj_name = os.path.basename(os.path.dirname(input_abs_path))
        view_count_name = os.path.basename(input_abs_path)

        full_label = f"{subj_name}/{view_count_name}"
        print(f"[INFO] Refining: {full_label}")

        chunks_batch = [[chunk] for chunk in scene_chunks]
        to_collect = {'ff_pts': [], 'ff_pts_conf': []}

        t0 = time.time()
        for chunk_batch in tqdm(chunks_batch, desc=f"Refining {full_label}", leave=False):
            chunk_batch = move_to_device(chunk_batch, args.device)
            with torch.no_grad():
                out = model(chunk_batch)

            # Unnormalize predictions
            pred_pts = unnormalize_func(chunk_batch[0], out['ff_pts_out'])
            to_collect['ff_pts'].append(pred_pts)
            to_collect['ff_pts_conf'].append(out['ff_pts_conf_out'])

        if len(scene_chunks) == 0:
            print(f"[WARN] No valid chunks found for {full_label}")
            pred_pts_agg = scene['ff_pts'].clone().to(args.device)
            pred_confs_agg = scene['ff_conf'].clone().to(args.device)
            pred_mask_agg = torch.ones_like(scene['ff_conf'], dtype=torch.bool).to(args.device)
        else:
            ff_pts_all = torch.cat(to_collect['ff_pts'], dim=0)
            ff_pts_conf_all = torch.cat(to_collect['ff_pts_conf'], dim=0)
            msks_in_scene = torch.stack([chunk['msks_in_scene'] for chunk in scene_chunks], dim=0).to(args.device)
            pred_pts_agg, pred_confs_agg, pred_mask_agg = aggregate_chunks(ff_pts_all, ff_pts_conf_all, msks_in_scene,
                                                                           scene)

        t1 = time.time()
        # print(f"[INFO] Refinement complete in {t1 - t0:.2f}s")

        # 2. Rerun Logging
        if not args.no_rerun:
            log_root = f"world/{subj_name}/{view_count_name}"

            # Initial Predictions (VGGT) - Log per frame and per view
            vggt_pts = scene['ff_pts_original'].cpu().numpy()  # (N, H, W, 3)
            vggt_conf = scene['ff_conf'].cpu().numpy()  # (N, H, W)
            refined_pts = pred_pts_agg.cpu().numpy()  # (N, H, W, 3)
            refined_mask = pred_mask_agg.cpu().numpy()  # (N, H, W)
            gt_pts = scene['gt_pts_metric'].cpu().numpy()  # (N, H, W, 3)
            gt_msks = scene['gt_msks'].cpu().numpy()  # (N, H, W)

            total_N = vggt_pts.shape[0]
            # Since prepare_ggpt_inputs.py stacks as (t0_v0, t0_v1, ..., t1_v0, ...)
            # we use n_views to recover t and v
            V = n_views

            for idx in range(total_N):
                t = idx // V
                v = idx % V

                # VGGT
                v_pts = vggt_pts[idx].reshape(-1, 3)
                v_mask = (vggt_conf[idx] > 0).reshape(-1)
                if v_mask.any():
                    entity = f"{log_root}/vggt/view_{v:02d}"
                    if log_pointcloud:
                        log_pointcloud(t, entity, v_pts[v_mask], color=[255, 100, 0])
                    else:
                        rr.set_time("frame", sequence=t)
                        rr.log(entity, rr.Points3D(v_pts[v_mask], colors=[255, 100, 0], radii=0.002))

                # GGPT Refined
                r_pts = refined_pts[idx].reshape(-1, 3)
                r_mask = refined_mask[idx].reshape(-1)
                if r_mask.any():
                    entity = f"{log_root}/ggpt_refined/view_{v:02d}"
                    if log_pointcloud:
                        log_pointcloud(t, entity, r_pts[r_mask], color=[0, 200, 255])
                    else:
                        rr.set_time("frame", sequence=t)
                        rr.log(entity, rr.Points3D(r_pts[r_mask], colors=[0, 200, 255], radii=0.002))

                # GT
                g_pts = gt_pts[idx].reshape(-1, 3)
                g_mask = gt_msks[idx].reshape(-1)
                if g_mask.any():
                    entity = f"{log_root}/gt/view_{v:02d}"
                    if log_pointcloud:
                        log_pointcloud(t, entity, g_pts[g_mask], color=[0, 255, 100])
                    else:
                        rr.set_time("frame", sequence=t)
                        rr.log(entity, rr.Points3D(g_pts[g_mask], colors=[0, 255, 100], radii=0.002))

            print(f"[SUCCESS] Results logged to Rerun under {log_root}")

        # 3. Save Refined Outputs
        # We save in two formats:
        # 1. A single .bin for quick loading
        # 2. Per-frame .npz files compatible with 4D_Umeyama.py evaluation

        # Format 1: .bin
        bin_save_path = os.path.join(scene_dir, "ggpt_outputs.bin")
        print(f"[INFO] Saving refined .bin to {bin_save_path}...")
        torch.save({
            "points": pred_pts_agg.cpu(),
            "point_masks": pred_mask_agg.cpu(),
            "points_conf": pred_confs_agg.cpu()
        }, bin_save_path)

        # Format 2: .npz sequence for 4D_Umeyama.py
        output_model_name = f"{args.model}-refined"
        dataset_type = args.dataset
        subject_map = get_subject_by_code(dataset_type)
        # vggt4d uses pair-based directory structure for hi4d
        if args.model == "vggt4d" and dataset_type == "hi4d":
            ds_cfg = get_dataset_config(dataset_type)
            subject_map = {}
            for name in ds_cfg["subject_names"]:
                pair, action = name.split("/")
                subject_map[action] = f"subject-{pair}/{action}"
        full_subject_name = subject_map.get(subj_code, f"subject-{subj_code}")
        out_strategy_dir = os.path.join("aligned_outputs", output_model_name, "baseline", full_subject_name,
                                        f"{n_views}views")
        os.makedirs(out_strategy_dir, exist_ok=True)

        print(f"[INFO] Saving refined .npz frames to {out_strategy_dir}...")

        vggt_pts_cpu = pred_pts_agg.cpu().numpy()
        vggt_conf_cpu = pred_confs_agg.cpu().numpy()
        geo_msks_cpu = scene['geo_msks'].cpu().numpy()
        ff_extri_cpu = scene['ff_extrinsics'].cpu().numpy()
        ff_intri_cpu = scene['ff_intrinsics'].cpu().numpy()
        gt_pts_cpu = scene['gt_pts_metric'].cpu().numpy()
        gt_extri_cpu = scene['gt_extrinsics'].cpu().numpy() if scene.get('gt_extrinsics') is not None else ff_extri_cpu

        # Resolve view names and load native GT intrinsics where an external dataset exists.
        pair_name = get_pair_name_for_subject(dataset_type, full_subject_name)
        raw_view_list = get_view_config(dataset_type, n_views, pair_name=pair_name)
        if dataset_type == "hi4d":
            # Hi4D views are just camera IDs like "4", "16"
            view_name_list = [str(v) for v in raw_view_list]
        elif dataset_type == "monofusion":
            view_name_list = [str(v) for v in raw_view_list]
        else:
            from eval_config import VIEW_CONFIGS, DEFAULT_TARGET_VIEWS
            view_name_list = [f"view_{v}" if not str(v).startswith("view_") else str(v) for v in raw_view_list]

        if dataset_type == "monofusion":
            native_gt_Ks = scene['gt_intrinsics'].cpu().numpy()[:n_views] if scene.get(
                'gt_intrinsics') is not None else ff_intri_cpu[:n_views]
        else:
            # Get the dataset root for loading GT intrinsics
            ds_root = get_dataset_root_for_subject(dataset_type, full_subject_name)
            from utils.gt import load_gt_params as _load_gt_params

            native_gt_Ks = []
            for vname in view_name_list:
                view_dir = os.path.join(ds_root, vname)
                try:
                    K_native, _ = _load_gt_params(view_dir, dataset_type=dataset_type)
                    native_gt_Ks.append(K_native)
                except Exception:
                    native_gt_Ks.append(ff_intri_cpu[0])  # fallback
            native_gt_Ks = np.array(native_gt_Ks)

        V = n_views
        num_frames = vggt_pts_cpu.shape[0] // V

        # ── Pre-compute per-frame static masks from optical flow / SAM2 ─────────
        # The input scene_dir contains ff_outputs.bin with images but no static
        # mask.  We recompute them from the dataset RGB frames exactly as Pi3's
        # align_reconstruction_umeyama.py does, so that masks_2d correctly
        # marks True=static, False=dynamic (arm / hand / can).
        ds_root_for_masks = get_dataset_root_for_subject(dataset_type,
                                                         full_subject_name) if dataset_type != "monofusion" else None
        static_masks_all = None  # (num_frames * V, H_mod, W_mod) or None
        if (get_rgb_path is not None and compute_static_mask is not None
                and ds_root_for_masks is not None and os.path.isdir(ds_root_for_masks)
                and len(view_name_list) > 0):
            try:
                H_mod, W_mod = vggt_pts_cpu.shape[1], vggt_pts_cpu.shape[2]
                static_masks_list = []  # one (H_mod, W_mod) per (frame, view)
                for t in range(num_frames):
                    for v_idx in range(len(view_name_list)):
                        from models.scripts.utils.camera_utils import discover_view_name as _discover_view_name
                        vname = _discover_view_name(ds_root_for_masks, native_gt_Ks[v_idx], dataset_type=dataset_type)
                        if vname is None:
                            vname = view_name_list[v_idx]

                        if dataset_type == "hi4d":
                            view_dir = os.path.join(ds_root_for_masks, "images", vname)
                        else:
                            view_dir = os.path.join(ds_root_for_masks, vname)

                        # Use two adjacent frames so flow can be computed
                        rgb_paths = []
                        for dt in [t, t + 1, t - 1]:
                            p = get_rgb_path(view_dir, dt, dataset_type=dataset_type)
                            if p is not None:
                                rgb_paths.append(p)
                            if len(rgb_paths) == 2:
                                break
                        mask = compute_static_mask(rgb_paths)
                        if mask is None:
                            print(
                                f"[WARN] compute_static_mask returned None for view {vname} at t={t}. Falling back to all-static.")
                            mask = np.ones((H_mod, W_mod), dtype=bool)
                        elif mask.shape != (H_mod, W_mod):
                            import cv2 as _cv2
                            mask = _cv2.resize(
                                mask.astype(np.uint8), (W_mod, H_mod),
                                interpolation=_cv2.INTER_NEAREST,
                            ).astype(bool)
                        static_masks_list.append(mask)
                static_masks_all = np.stack(static_masks_list)  # (num_frames * V, H_mod, W_mod)
                print(f"[INFO] Computed static masks for {num_frames} frames × {len(view_name_list)} views.")
            except Exception as e:
                print(f"[WARN] Could not compute static masks, falling back to geo_msks: {e}")
                static_masks_all = None

        for t in range(num_frames):
            start = t * V
            end = (t + 1) * V

            f_pts = vggt_pts_cpu[start:end]
            f_conf = vggt_conf_cpu[start:end]
            f_w2c = ff_extri_cpu[start:end]
            f_intri = ff_intri_cpu[start:end]
            f_gt = gt_pts_cpu[start:end]
            f_gt_extri = gt_extri_cpu[start:end]

            # Use SAM2/flow static masks if available, else fall back to geo_msks
            if static_masks_all is not None:
                f_masks = static_masks_all[start:end]  # True=static, False=dynamic
            else:
                f_masks = geo_msks_cpu[start:end]

            f_c2w = np.stack([np.linalg.inv(w2c) for w2c in f_w2c])

            save_dict = {
                'pointmaps': f_pts,
                'pointmaps_confs': f_conf,
                'est_poses': f_c2w,
                'est_intrinsics': f_intri,
                'Ks': native_gt_Ks,  # Native GT intrinsics for discover_view_name
                'R_ts': f_gt_extri[:, :3, :4],  # GT extrinsics for split_points_by_mask
                'masks_2d': f_masks,
                'gt_pts': f_gt,
                'frame_idx': t,
                'view_names': np.array(view_name_list),  # Explicit view names
            }

            out_path = os.path.join(out_strategy_dir, f"frame_{t:05d}.npz")
            np.savez(out_path, **save_dict)


def main():
    args = parse_args()

    # Resolve base_input_dir if not provided
    if args.base_input_dir is None:
        if args.model in ["pi3", "pi3x"]:
            if args.dataset == "hi4d":
                args.base_input_dir = os.path.expanduser(f"~/Pi3/ggpt_inputs/hi4d/{args.model}")
            else:
                args.base_input_dir = os.path.expanduser(f"~/Pi3/ggpt_inputs/{args.model}")
        elif args.model == "vggt" and args.dataset == "hi4d":
            args.base_input_dir = os.path.expanduser("~/vggt/ggpt_inputs/hi4d")
        elif args.model == "vggt4d":
            if os.path.exists("/local/home/frrajic/xode/fabio/vggt4d_repo/ggpt_inputs"):
                args.base_input_dir = "/local/home/frrajic/xode/fabio/vggt4d_repo/ggpt_inputs"
            else:
                args.base_input_dir = os.path.expanduser(f"~/vggt4d/ggpt_inputs")
        elif args.dataset == "monofusion":
            if os.path.exists("/local/home/frrajic/xode/fabio/monofusion/ggpt_inputs"):
                args.base_input_dir = "/local/home/frrajic/xode/fabio/monofusion/ggpt_inputs"
            else:
                args.base_input_dir = os.path.expanduser("~/monofusion/ggpt_inputs")
        elif args.dataset == "hi4d":
            args.base_input_dir = os.path.expanduser(f"~/{args.model}/ggpt_inputs/hi4d")
        else:
            args.base_input_dir = os.path.expanduser(f"~/{args.model}/ggpt_inputs")
    else:
        args.base_input_dir = os.path.abspath(os.path.expanduser(args.base_input_dir))

    print(f"[INFO] Dataset: {args.dataset}")
    print(f"[INFO] Base input dir: {args.base_input_dir}")

    # 1. Setup Rerun (Removed global init, moved to run_refinement for per-instance tabs)
    # if not args.no_rerun:
    #     print(f"[INFO] Connecting to Rerun at {RERUN_ADDR}...")
    #     rr.init("GGPT_Refinement", spawn=False)
    #     rr.connect_grpc(RERUN_ADDR)
    #
    #     if configure_rerun_view_defaults:
    #         configure_rerun_view_defaults("world", RERUN_EYE_UP)
    #     else:
    #         rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # 2. Initialize Hydra manually to bypass CLI conflicts
    # This loads the config from configs/benchmark_ggpt.yaml
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="benchmark_ggpt")

    # 3. Load Model
    print(f"[INFO] Loading GGPT model from {args.ckpt}...")
    model = instantiate(cfg.ggptmodel_config).eval()
    ckpt = torch.load(args.ckpt, map_location='cpu')
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    model = model.to(args.device)

    # For vggt4d+hi4d, precompute action -> pair mapping
    vggt4d_hi4d_pair_map = {}
    if args.model == "vggt4d" and args.dataset == "hi4d":
        ds_cfg = get_dataset_config(args.dataset)
        for name in ds_cfg["subject_names"]:
            pair, action = name.split("/")
            vggt4d_hi4d_pair_map[action] = pair

    # 4. Resolve Subjects
    if args.subject == "all":
        if args.model == "vggt4d" and args.dataset == "hi4d":
            # vggt4d uses pair-based structure: subject-pair00/dance00/Nviews/
            subjects = []
            for pair_dir in sorted(glob.glob(os.path.join(args.base_input_dir, "subject-pair*"))):
                for action_dir in sorted(os.listdir(pair_dir)):
                    if os.path.isdir(os.path.join(pair_dir, action_dir)):
                        subjects.append(action_dir)
            print(f"[INFO] Found {len(subjects)} subjects: {subjects}")
        else:
            subj_folders = sorted(glob.glob(os.path.join(args.base_input_dir, "subject-*")))
            subjects = [os.path.basename(f).replace("subject-", "") for f in subj_folders]
            print(f"[INFO] Found {len(subjects)} subjects: {subjects}")
    else:
        subj_code = args.subject.split("/")[-1]
        if subj_code.startswith("subject-"):
            subj_code = subj_code.replace("subject-", "", 1)
        subjects = [subj_code]

    # 5. Process loop
    for subj in subjects:
        for v in args.views:
            if args.model == "vggt4d" and args.dataset == "hi4d" and subj in vggt4d_hi4d_pair_map:
                pair = vggt4d_hi4d_pair_map[subj]
                scene_dir = os.path.join(args.base_input_dir, f"subject-{pair}", subj, f"{v}views")
            else:
                scene_dir = os.path.join(args.base_input_dir, f"subject-{subj}", f"{v}views")
            if not os.path.isdir(scene_dir):
                # print(f"[WARN] Scene directory not found: {scene_dir}")
                continue
            run_refinement(model, cfg, scene_dir, args, subj, v)

    print("[SUCCESS] Refinement and logging complete.")


if __name__ == "__main__":
    main()
