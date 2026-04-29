import torch
from utils.geometry import unproject_depth_map_to_point_map_torch, closed_form_inverse_se3
from PIL import Image
import numpy as np
import sys
import os


def preprocess(images, output_width=518):
    # Used to reproduce the submitted results
    if isinstance(images, torch.Tensor) == False:
        # first convert to torch tensor
        images = torch.from_numpy(np.stack(images, axis=0)).float() / 255.0
    _, original_height, original_width, _ = images.shape
    if original_width == output_width and original_height % 14 == 0:
        return images
    if original_width < original_height:
        images = images.permute(0, 2, 1, 3)  # (N,H,W,C)
        original_width, original_height = original_height, original_width
    output_height = round(output_width * original_height / original_width / 14) * 14
    images_ff = torch.nn.functional.interpolate(images.permute(0, 3, 1, 2), size=(output_height, output_width),
                                                mode='bilinear', align_corners=False)
    images_ff = images_ff.permute(0, 2, 3, 1)  # (N,H,W,C)
    return images_ff


class FeedForward_Model(torch.nn.Module):
    def __init__(self, configs):
        super(FeedForward_Model, self).__init__()
        self.configs = configs
        if 'vggt' in self.configs.model:
            if 'point' in self.configs.model:
                self.vggt_branch = 'point'
            else:
                self.vggt_branch = 'depth'
            vggt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'vggt')
            if vggt_path not in sys.path:
                sys.path.insert(0, vggt_path)
            from vggt.models.vggt import VGGT
            '''
            self.model = VGGT(enable_point=(self.vggt_branch=='point'), enable_track=False, enable_depth=(self.vggt_branch=='depth'), enable_camera=True)
            downloaded_ckpt = 'feedforward/checkpoints/vggt-b/model.pt'
            self.model.load_state_dict(torch.load(downloaded_ckpt), strict=False)
            '''
            self.model = VGGT.from_pretrained("facebook/VGGT-1B")
        elif self.configs.model == 'dav3':
            from depth_anything_3.api import DepthAnything3
            self.model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
        elif self.configs.model == 'pi3':
            from Pi3.pi3.models.pi3 import Pi3
            self.model = Pi3.from_pretrained("yyfz233/Pi3")
        elif self.configs.model == 'pi3x':
            from pi3.models.pi3x import Pi3X
            self.model = Pi3X()
            from safetensors.torch import load_file
            weight = load_file('/iopsstor/scratch/cscs/cyutong/code/Pi3X/ckpt/model.safetensors')
            self.model.load_state_dict(weight, strict=False)
        elif self.configs.model == 'ma':
            from mapanything.models import MapAnything
            self.model = MapAnything.from_pretrained("facebook/map-anything")
        elif self.configs.model == 'mast3r':
            mast3r_path = '/home/fabio/mast3r'
            # Robust path discovery for remote execution
            potential_paths = [mast3r_path]
            # Try to find dust3r and croco submodules
            for root in [mast3r_path]:
                d_path = os.path.join(root, 'dust3r')
                if os.path.isdir(d_path):
                    potential_paths.append(d_path)
                    potential_paths.append(os.path.join(d_path, 'dust3r'))  # some structures have nested pkg
                    potential_paths.append(os.path.join(d_path, 'croco'))

            for p in potential_paths:
                if os.path.isdir(p) and p not in sys.path:
                    sys.path.insert(0, p)

            # Shadowing fix: Force 'models' to point to croco/models
            croco_path = os.path.join(mast3r_path, 'dust3r', 'croco')
            import types
            m = types.ModuleType('models')
            m.__path__ = [os.path.join(croco_path, 'models')]
            old_models = sys.modules.get('models')
            sys.modules['models'] = m

            try:
                from mast3r.model import AsymmetricMASt3R
                self.model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
            finally:
                if old_models:
                    sys.modules['models'] = old_models
        else:
            raise NotImplementedError(f"Model {self.configs.model} not implemented in FeedForward_Model.")
        self.model.eval()

    def forward(self, images, preprocessed=False, gt_dict=None):
        output_dict = {}
        device = images.device if isinstance(images, torch.Tensor) else 'cuda'
        output_width = 504 if self.configs.model == 'dav3' else 518
        images_ff = preprocess(images, output_width).to(device) if not preprocessed else images
        output_dict['images_ff'] = images_ff
        if 'vggt' in self.configs.model:
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.autocast(device_type='cuda', dtype=dtype):
                raw_outputs = self.model(images_ff.permute(0, 3, 1, 2))  # (N,3,H,W)
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            output_dict['extrinsics'], output_dict['intrinsics'] = pose_encoding_to_extri_intri(raw_outputs['pose_enc'],
                                                                                                images_ff.shape[1:3])
            output_dict['extrinsics'], output_dict['intrinsics'] = output_dict['extrinsics'][0], \
            output_dict['intrinsics'][0]  # squeeze the batch=1 dimension
            if 'depth' in self.configs.model:
                B, N, H, W, D = raw_outputs['depth'].shape
                output_dict['points'] = unproject_depth_map_to_point_map_torch(
                    depth_map=raw_outputs['depth'].view(B * N, H, W),
                    extrinsics_cam=output_dict['extrinsics'].view(B * N, 3, 4),
                    intrinsics_cam=output_dict['intrinsics'].view(B * N, 3, 3)).view(B, N, H, W, 3)[0]
                output_dict['points_conf'] = raw_outputs['depth_conf'][0]  # squeeze the batch=1 dimension
            else:
                output_dict['points'] = raw_outputs['world_points'][0]  # (N,H,W,3)
                output_dict['points_conf'] = raw_outputs['world_points_conf'][0]  # (N,H,W)
        elif self.configs.model == 'dav3':
            if self.configs.dav3.get('input_pose', False):
                input_intrinsics = gt_dict['intrinsics'].clone()
                input_intrinsics[..., 1, 2] = input_intrinsics[..., 1, 2] + 0.5
                input_intrinsics[..., 0, 2] = input_intrinsics[..., 0, 2] + 0.5
                input_extrinsics = torch.eye(4, device=device)[None, :, :].repeat(images_ff.shape[0], 1, 1).to(device)
                input_extrinsics[:, :3, :4] = gt_dict['extrinsics'].clone()
            else:
                input_intrinsics = None
                input_extrinsics = None
            dav3_results = self.model.inference(
                image=images_ff.permute(0, 3, 1, 2).unsqueeze(0),
                intrinsics=input_intrinsics,
                extrinsics=input_extrinsics)  # 1, N, C, H, W

            H, W = images_ff.shape[1:3]

            output_dict['extrinsics'] = dav3_results['extrinsics'][0].float()
            output_dict['intrinsics'] = dav3_results['intrinsics'][0].float().clone()
            # Convert to opencv convention (cx = W-1/2, cy=H/2)
            output_dict['intrinsics'][..., 1, 2] = output_dict['intrinsics'][..., 1, 2] - 0.5
            output_dict['intrinsics'][..., 0, 2] = output_dict['intrinsics'][..., 0, 2] - 0.5
            output_dict['points'] = unproject_depth_map_to_point_map_torch(
                depth_map=dav3_results.depth.view(-1, H, W),
                extrinsics_cam=output_dict['extrinsics'][..., :3, :].view(-1, 3, 4),
                intrinsics_cam=output_dict['intrinsics'].view(-1, 3, 3))
            output_dict['points_conf'] = dav3_results['depth_conf'].view(-1, H, W)
        elif self.configs.model in ['pi3', 'pi3x']:
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            if self.configs.model == 'pi3x':
                if self.configs.pi3x.get('input_intrinsics', False):
                    input_intrinsics = gt_dict['intrinsics'].clone()
                    input_intrinsics[..., 1, 2] = input_intrinsics[..., 1, 2] + 0.5
                    input_intrinsics[..., 0, 2] = input_intrinsics[
                                                      ..., 0, 2] + 0.5  # Not sure here. But I guess it follows colmap-convention?
                    input_intrinsics = input_intrinsics.unsqueeze(0)
                else:
                    input_intrinsics = None

                if self.configs.pi3x.get('input_extrinsics', False):
                    input_extrinsics = torch.eye(4, device=device)[None, :, :].repeat(images_ff.shape[0], 1, 1).to(
                        device)
                    input_extrinsics[:, :3, :4] = closed_form_inverse_se3(gt_dict['extrinsics'].clone())[
                        :, :3, :4]  # opencv c2w
                    input_extrinsics = input_extrinsics.unsqueeze(0)
                else:
                    input_extrinsics = None
                # TODO accept partial depth
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=dtype):
                        res = self.model(
                            imgs=images_ff.permute(0, 3, 1, 2).unsqueeze(0),  # B, N, C, H, W [0,1]
                            intrinsics=input_intrinsics,
                            poses=input_extrinsics,
                        )

            else:
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=dtype):
                        res = self.model(images_ff.permute(0, 3, 1, 2).unsqueeze(0))
            from MoGe.moge.utils.geometry_torch import recover_focal_shift
            hh, ww = res['local_points'].shape[-3:-1]
            aspect_ratio = ww / hh
            focal, shift = recover_focal_shift(points=res['local_points'],
                                               mask=torch.sigmoid(res['conf'][..., 0]) > 0.1)
            f = (focal / 2 * (1 + aspect_ratio ** 2) ** 0.5) * hh
            intrinsics = torch.eye(3, device=device)[None, :, :].repeat(res['camera_poses'].shape[1], 1, 1)
            intrinsics[:, 0, 0] = f[0]
            intrinsics[:, 1, 1] = f[0]  # squeeze the batch=1 dimension
            intrinsics[:, 0, 2] = (ww - 1) / 2
            intrinsics[:, 1, 2] = (hh - 1) / 2
            bb, nn = res['camera_poses'].shape[:2]
            extrinsics = closed_form_inverse_se3(res['camera_poses'].view(-1, 4, 4)).view(*res['camera_poses'].shape)
            output_dict['extrinsics'] = extrinsics[0]
            output_dict['intrinsics'] = intrinsics
            output_dict['points'] = res['points'][0]
            output_dict['points_conf'] = res['conf'][0, ..., 0].exp() + 1
        elif self.configs.model == 'ma':
            from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
            import torchvision.transforms as tvf
            img_norm = IMAGE_NORMALIZATION_DICT['dinov2']
            N, H, W, C = images_ff.shape
            MA_RESOLUTION_LIST = [  # (w,h)
                (518, 518),  # 1:1
                (518, 392),  # 4:3
                (518, 336),  # 3:2
                (518, 294),  # 16:9
                (518, 252),  # 2:1
                (518, 168),  # 3.2:1
                (392, 518),  # 3:4
                (336, 518),  # 2:3
                (294, 518),  # 9:16
                (252, 518),  # 1:2

            ]
            assert (W, H) in MA_RESOLUTION_LIST, f'mapanything only supports {MA_RESOLUTION_LIST} now, but got {(W, H)}'
            images_ff = tvf.functional.normalize(images_ff.permute(0, 3, 1, 2), mean=img_norm.mean,
                                                 std=img_norm.std).reshape(1, N, C, H, W)
            input_list = [dict(img=images[:, ii], true_shape=np.int32([[H, W]]), idx=ii, instance=str(ii),
                               data_norm_type=['dinov2']) for ii in range(N)]
            # add multi-modality
            ma_results = self.model.infer(
                train_mode=False,
                views=input_list,
                memory_efficient_inference=False,  # Trades off speed for more views (up to 2000 views on 140 GB)
                use_amp=True,  # Use mixed precision inference (recommended)
                amp_dtype="bf16",  # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
                apply_mask=False,  # Apply masking to dense geometry outputs (We set to False!)
                mask_edges=False,  # Remove edge artifacts by using normals and depth (We set to False!)
                apply_confidence_mask=True,  # Filter low-confidence regions
                confidence_percentile=10,  # Remove bottom 10 percentile confidence pixels (Useless?)
            )
            output_dict['intrinsics'] = []
            output_dict['extrinsics'] = []
            output_dict['points'] = []
            output_dict['points_conf'] = []
            for i, pred in enumerate(ma_results):
                cam2world = pred["camera_poses"]  # 1,4,4
                output_dict['extrinsics'].append(closed_form_inverse_se3(cam2world))
                output_dict['intrinsics'].append(pred["intrinsics"])
                output_dict['points'].append(pred["pts3d"])  # 1,H,W,3
                output_dict['points_conf'].append(pred["conf"])
            output_dict = {k: torch.stack(v, axis=1).squeeze(0) for k, v in output_dict.items()}
        elif self.configs.model == 'mast3r':
            mast3r_path = '/home/fabio/mast3r'
            potential_paths = [mast3r_path]
            for root in [mast3r_path]:
                d_path = os.path.join(root, 'dust3r')
                if os.path.isdir(d_path):
                    potential_paths.append(d_path)
                    potential_paths.append(os.path.join(d_path, 'dust3r'))
                    potential_paths.append(os.path.join(d_path, 'croco'))

            for p in potential_paths:
                if os.path.isdir(p) and p not in sys.path:
                    sys.path.insert(0, p)

            # Shadowing fix: Force 'models' to point to croco/models
            croco_path = os.path.join(mast3r_path, 'dust3r', 'croco')
            import types
            m = types.ModuleType('models')
            m.__path__ = [os.path.join(croco_path, 'models')]
            old_models = sys.modules.get('models')
            sys.modules['models'] = m

            try:
                from mast3r.image_pairs import make_pairs
                from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
                from dust3r.utils.image import load_images
                from dust3r.utils.device import to_numpy
            finally:
                if old_models:
                    sys.modules['models'] = old_models

            import tempfile
            import shutil

            # Save raw images to temp files since dust3r/mast3r expects file paths
            temp_dir = tempfile.mkdtemp(prefix="mast3r_ggpt_")
            image_paths = []

            # Handle list of images or tensor
            if isinstance(images, list):
                N = len(images)
                for i in range(N):
                    img_path = os.path.join(temp_dir, f"frame_{i}.png")
                    img_np = images[i] if isinstance(images[i], np.ndarray) else np.array(images[i])
                    # Assuming RGB 0-255
                    Image.fromarray(img_np.astype(np.uint8)).save(img_path)
                    image_paths.append(img_path)
            else:
                N = images.shape[0]
                for i in range(N):
                    img_path = os.path.join(temp_dir, f"frame_{i}.png")
                    img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
                    if img_np.shape[0] == 3:  # C, H, W -> H, W, C
                        img_np = np.transpose(img_np, (1, 2, 0))
                    Image.fromarray(img_np).save(img_path)
                    image_paths.append(img_path)

            try:
                # Load via dust3r util (resizes to 512 internally)
                imgs_loaded = load_images(image_paths, size=512)
                pairs = make_pairs(imgs_loaded, scene_graph="complete", symmetrize=True)

                # Suppress output if desired, or let it print
                scene = sparse_global_alignment(
                    image_paths,
                    pairs,
                    temp_dir,
                    self.model,
                    device=device,
                    matching_conf_thr=0.0
                )

                pts3d_world_list, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=True))

                try:
                    im_poses = scene.get_im_poses()
                except AttributeError:
                    im_poses = scene.get_poses()

                try:
                    est_intrinsics_all = scene.get_intrinsics()
                except AttributeError:
                    est_intrinsics_all = scene.intrinsics

                H, W = confs[0].shape

                # Store outputs
                output_dict['points'] = torch.from_numpy(np.stack(pts3d_world_list)).to(device).reshape(N, H, W, 3)
                output_dict['geo_points'] = torch.from_numpy(np.stack(pts3d_world_list)).to(device).reshape(N, H, W, 3)
                output_dict['points_conf'] = torch.from_numpy(np.stack(confs)).to(device)
                output_dict['extrinsics'] = closed_form_inverse_se3(
                    im_poses.to(device))  # Convert cam2world to world2cam
                output_dict['intrinsics'] = est_intrinsics_all.to(device)

                # Update images_ff to match MASt3R resolution (usually 512 on long side)
                mast3r_imgs = []
                for img_data in imgs_loaded:
                    # img_data['img'] is (1, 3, H, W), normalized to approx [-1, 1]
                    mast3r_imgs.append(img_data['img'][0].permute(1, 2, 0) * 0.5 + 0.5)
                output_dict['images_ff'] = torch.stack(mast3r_imgs).to(device)

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        else:
            raise NotImplementedError(f"Model {self.configs.model} not implemented in FeedForward_Model.")

        return output_dict
