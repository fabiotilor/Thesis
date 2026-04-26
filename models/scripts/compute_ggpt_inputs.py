import os
import sys
import argparse
import numpy as np
import torch
import cv2

# Ensure we can import from the parent thesis directories and ggpt base
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from feedforward import FeedForward_Model, preprocess
from matching import init_match_models
from sfm.sfm_func import run_sfm
from utils.geometry import unproject_depth_map_to_point_map_torch, project_point_map_to_depth_map_torch
from sfm.dataloader.dexycb_dataset import DexYCBDataset

from eval_config import (
    DATASET_BASE_ROOT, SUBJECT_NAMES, SUBJECT_BY_CODE,
    VIEW_CONFIGS, SUPPORTED_MODELS, GGPT_INPUTS_ROOT, DEVICE
)


def move_to_device(batch, device):
    if type(batch) is dict:
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif type(batch) is list:
        return [move_to_device(v, device) for v in batch]
    elif type(batch) is torch.Tensor:
        return batch.to(device)
    else:
        return batch


def prepare_batch(batch, output_width=518):
    batch_ffres = []
    for batch_ in batch:
        batch_ffres_ = {key: value for key, value in batch_.items() if type(value) is str}
        in_h, in_w = batch_['images'][0].shape[:2]
        batch_ffres_['images'] = preprocess(batch_['images'], output_width=output_width).to(batch_['images'][0].device)
        ff_h, ff_w = batch_ffres_['images'].shape[1:3]
        if 'depths' in batch_:
            if batch_['depths'].shape[1] == ff_h and batch_['depths'].shape[2] == ff_w:
                batch_ffres_['depths'] = batch_['depths']
            else:
                batch_ffres_['depths'] = torch.nn.functional.interpolate(batch_['depths'].unsqueeze(1),
                                                                         size=(ff_h, ff_w),
                                                                         mode='nearest-exact').squeeze(1)

        if 'point_masks' in batch_:
            if batch_['point_masks'].shape[1] == ff_h and batch_['point_masks'].shape[2] == ff_w:
                batch_ffres_['point_masks'] = batch_['point_masks']
            else:
                batch_ffres_['point_masks'] = torch.nn.functional.interpolate(
                    batch_['point_masks'].unsqueeze(1).float(), size=(ff_h, ff_w), mode='nearest-exact').squeeze(
                    1).bool()
        if 'intrinsics' in batch_:
            Ks = batch_['intrinsics'].clone()
            Ks[:, 0, 0] *= ff_w / batch_['images'][0].shape[1]
            Ks[:, 1, 1] *= ff_h / batch_['images'][0].shape[0]
            Ks[:, 0, 2] = (ff_w - 1) / 2.0
            Ks[:, 1, 2] = (ff_h - 1) / 2.0
            batch_ffres_['intrinsics'] = Ks
        if 'extrinsics' in batch_:
            batch_ffres_['extrinsics'] = batch_['extrinsics']
        if 'depths' in batch_ffres_:
            batch_ffres_['points'] = unproject_depth_map_to_point_map_torch(depth_map=batch_ffres_['depths'],
                                                                            extrinsics_cam=batch_['extrinsics'],
                                                                            intrinsics_cam=batch_ffres_['intrinsics'])
        elif 'points' in batch_:
            if in_h == ff_h and in_w == ff_w:
                batch_ffres_['points'] = batch_['points']
            else:
                batch_ffres_['points'] = torch.nn.functional.interpolate(batch_['points'].permute(0, 3, 1, 2).float(),
                                                                         size=(ff_h, ff_w),
                                                                         mode='nearest-exact').permute(0, 2, 3,
                                                                                                       1).bool()
            batch_ffres_['depths'] = project_point_map_to_depth_map_torch(point_map=batch_ffres_['points'],
                                                                          extrinsics_cam=batch_['extrinsics'],
                                                                          intrinsics_cam=batch_ffres_['intrinsics'])
            batch_ffres_['depths'][batch_ffres_['point_masks'] == False] = 0
        batch_ffres.append(batch_ffres_)
    return batch_ffres


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
        subjects_to_run = SUBJECT_NAMES  # default all

    print(f"Loading FeedForward model: {args.model}")
    ff_cfg = DummyConfig(model=args.model, dav3={'input_pose': False},
                         pi3x={'input_intrinsics': False, 'input_extrinsics': False})
    ff_model = FeedForward_Model(ff_cfg).to(DEVICE)
    ff_model.eval()

    print(f"Loading Matcher: {args.matcher}")
    match_models = init_match_models([args.matcher], device=DEVICE)

    # Config for SfM
    sfm_cfg = DummyConfig(
        ba_config={'score_thresh': 0.6, 'cycle_err_thresh': 2, 'ff_err_thresh': None, 'mintrack_per_view': 2048,
                   'shared_camera': True, 'refine_focal_length': True, 'calibrated': False,
                   'loss_function_type': 'cauchy',
                   'loss_function_scale': 0.1, 'camera_type': 'PINHOLE'},
        dlt_config={'score_thresh': 0.1, 'cycle_err_thresh': 4, 'max_epipolar_error': 4, 'min_tri_angle': 3,
                    'max_reproj_error': 4, 'batch_size': 50000},
        match_config={'models': [args.matcher], 'save_vis': False},
        common_config={'save_outputs': False, 'save_vis': False, 'reduce_memory': False}
    )

    output_width = 504 if args.model == 'dav3' else 518

    for subject in subjects_to_run:
        subject_code = subject.split("subject-")[1][:2]
        for nviews in args.views:
            if nviews not in VIEW_CONFIGS or VIEW_CONFIGS[nviews] is None:
                continue

            view_names = [f"view_{v}" for v in VIEW_CONFIGS[nviews]]
            dataset_root = os.path.join(DATASET_BASE_ROOT, subject)

            print(f"\\nProcessing {subject_code} - {nviews} views")

            dataset = DexYCBDataset(
                name=f"{subject_code}_{nviews}views",
                root=dataset_root,
                view_names=view_names,
                img_size=output_width,
                load_depths=True
            )

            out_dir = os.path.join(GGPT_INPUTS_ROOT, args.model, subject, f"{nviews}views")
            os.makedirs(out_dir, exist_ok=True)

            for i in range(len(dataset)):
                batch_cpu = dataset[i]
                # Format of batch_cpu: images (V, H, W, 3), extrinsics (V, 4, 4), intrinsics (V, 3, 3), depths (V, H, W), point_masks (V, H, W)
                # Need to wrap in list for batch size 1
                batch_cpu = [batch_cpu]

                batch_rawres = move_to_device(batch_cpu, DEVICE)
                batch_ffres = prepare_batch(batch_rawres, output_width=output_width)
                batch_ffres = batch_ffres[0]

                frame_name = batch_ffres['seq_name']
                out_path = os.path.join(out_dir, f"{frame_name}.npz")

                if os.path.exists(out_path):
                    print(f"Skipping {frame_name}, already computed.")
                    continue

                print(f"  Frame {frame_name}...")

                with torch.no_grad():
                    ff_outputs = ff_model(batch_ffres['images'], preprocessed=True, gt_dict=batch_ffres)

                # SfM Step
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

                         est_poses=ff_outputs.get('extrinsics',
                                                  torch.eye(4).unsqueeze(0).repeat(nviews, 1, 1)).cpu().numpy(),
                         est_intrinsics=ff_outputs.get('intrinsics', batch_ffres['intrinsics']).cpu().numpy(),
                         gt_pts=batch_ffres['points'].cpu().numpy().reshape(-1, 3),
                         # Full GT dense pointcloud for evaluation

                         # Aliases for compatibility
                         pointmaps=ff_outputs['points'].cpu().numpy(),
                         pointmaps_confs=ff_outputs['points_conf'].cpu().numpy()
                         )


if __name__ == "__main__":
    main()
