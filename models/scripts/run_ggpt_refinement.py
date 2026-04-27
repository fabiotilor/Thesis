import os
import sys
import argparse
import numpy as np
import torch
import glob
from tqdm import tqdm

# Ensure we can import from the parent thesis directories and ggpt base
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval_config import (
    SUBJECT_NAMES, SUBJECT_BY_CODE,
    VIEW_CONFIGS, SUPPORTED_MODELS, GGPT_INPUTS_ROOT, DEVICE, GGPT_CKPT
)

from hydra.utils import instantiate
from ggpt.dataloader.demo_dataset import DemoDataset
from utils.points import aggregate_chunks


class DummyGGPTConfig:
    def __init__(self):
        self._target_ = "ggpt.ggpt_func.run_ggpt"
        self.chunk_size = 4000
        self.depth_scale = 10.0
        self.grid_size = 0.05


def move_to_device(batch, device):
    if type(batch) is dict:
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif type(batch) is list:
        return [move_to_device(v, device) for v in batch]
    elif type(batch) is torch.Tensor:
        return batch.to(device)
    else:
        return batch


def main():
    parser = argparse.ArgumentParser(description="Run GGPT refinement on precomputed inputs")
    parser.add_argument("--model", type=str, default="vggt-point", choices=SUPPORTED_MODELS)
    parser.add_argument("--all", action="store_true", help="Process all subjects")
    for i in range(1, 11):
        parser.add_argument(f"--{i:02d}", action="store_true", help=f"Process subject {i:02d}")
    parser.add_argument("--views", nargs='+', type=int, default=[2, 3, 4], help="Number of views to process")
    args = parser.parse_args()

    subjects_to_run = []
    for i in range(1, 11):
        if getattr(args, f"{i:02d}"):
            subjects_to_run.append(SUBJECT_BY_CODE[f"{i:02d}"])
    if not subjects_to_run:
        subjects_to_run = SUBJECT_NAMES

    # Load GGPT model
    # The model target is defined in cfg.ggptmodel_config._target_
    # which is ggpt.model.base.BasePredictor
    # Actually, run_demo.py instantiates the model via hydra. Let's use hydra compose or just build it manually.
    # It's better to use OmegaConf to load configs/demo.yaml
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs",
                     "demo.yaml"))

    ggpt_model = instantiate(cfg.ggptmodel_config).eval()
    ckpt = torch.load(
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), GGPT_CKPT),
        map_location='cpu')
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    ggpt_model.load_state_dict(ckpt, strict=True)
    ggpt_model = ggpt_model.to(DEVICE)
    print(f"Loaded GGPT model from {GGPT_CKPT}")

    out_model_name = f"{args.model}_ggpt"

    for subject in subjects_to_run:
        subject_code = subject.split("subject-")[1][:2]
        for nviews in args.views:
            if nviews not in VIEW_CONFIGS or VIEW_CONFIGS[nviews] is None:
                continue

            in_dir = os.path.join(GGPT_INPUTS_ROOT, args.model, subject, f"{nviews}views")
            out_dir = os.path.join(GGPT_INPUTS_ROOT, out_model_name, subject, f"{nviews}views")

            if 'mast3r' in args.model:
                os.environ['SKIP_GGPT_PREALIGN'] = '1'
            else:
                os.environ['SKIP_GGPT_PREALIGN'] = '0'

            if not os.path.exists(in_dir):
                print(f"WARNING: No input directory {in_dir}")
                continue

            os.makedirs(out_dir, exist_ok=True)

            npz_files = sorted(glob.glob(os.path.join(in_dir, "*.npz")))
            print(f"\\nProcessing {subject_code} - {nviews} views - {len(npz_files)} frames")

            for npz_path in tqdm(npz_files, desc="Frames"):
                frame_name = os.path.basename(npz_path).replace(".npz", "")
                out_path = os.path.join(out_dir, f"{frame_name}.npz")

                if os.path.exists(out_path):
                    continue

                data = dict(np.load(npz_path))

                # Construct ff_data and geo_data expected by DemoDataset
                ff_data = {
                    'points': torch.from_numpy(data['ff_points']).to(DEVICE),
                    'points_conf': torch.from_numpy(data['ff_points_conf']).to(DEVICE),
                    'images_ff': torch.from_numpy(data['images_ff']).to(DEVICE),
                    'extrinsics': torch.from_numpy(data['est_poses']).to(DEVICE),
                    'intrinsics': torch.from_numpy(data['est_intrinsics']).to(DEVICE),
                }

                geo_data = {
                    'points': torch.from_numpy(data['geo_points']).to(DEVICE),
                    'point_masks': torch.from_numpy(data['geo_point_masks']).to(DEVICE),
                }

                demo_dataset = DemoDataset(name='demo', ff_data=ff_data, geo_data=geo_data)
                scene_chunks, scene = demo_dataset[0]
                # Process all chunks in a single batch to speed up refinement
                chunks_batch = move_to_device(scene_chunks, DEVICE)
                with torch.no_grad():
                    out = ggpt_model(chunks_batch)

                # Unnormalize and collect
                to_collect = {'ff_pts': [], 'ff_pts_conf': []}
                for i in range(len(scene_chunks)):
                    # Extract the individual chunk's output from the batch
                    chunk_out_pts = out['ff_pts_out'][i]
                    chunk_out_conf = out['ff_pts_conf_out'][i]

                    to_collect['ff_pts'].append(demo_dataset.unnormalize_pts(scene_chunks[i], chunk_out_pts))
                    to_collect['ff_pts_conf'].append(chunk_out_conf)

                ff_pts_all = torch.cat(to_collect['ff_pts'], dim=0)
                ff_pts_conf_all = torch.cat(to_collect['ff_pts_conf'], dim=0)
                msks_in_scene = torch.stack([chunk['msks_in_scene'] for chunk in scene_chunks], dim=0).to(DEVICE)

                pred_pts, pred_confs, pred_mask = aggregate_chunks(ff_pts_all, ff_pts_conf_all, msks_in_scene, scene)

                # Update data with refined points
                data['pointmaps'] = pred_pts.cpu().numpy()
                data['pointmaps_confs'] = pred_confs.cpu().numpy()
                # Also update ff_points to be safe
                data['ff_points'] = data['pointmaps']
                data['ff_points_conf'] = data['pointmaps_confs']

                np.savez(out_path, **data)


if __name__ == "__main__":
    main()
