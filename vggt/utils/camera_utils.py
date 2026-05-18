import os
import glob
import numpy as np

_CAMERA_CACHE = {}


def build_camera_intrinsics_cache(dataset_root, dataset_type="dex-ycb"):
    """
    Builds a mapping from (rounded) K.flatten() to the folder name (e.g. 'view_01' or '16').
    """
    subject_name = os.path.basename(dataset_root)
    cache_key = f"{dataset_type}_{subject_name}"
    if cache_key in _CAMERA_CACHE:
        return _CAMERA_CACHE[cache_key]

    cache = {}
    if dataset_type == "dex-ycb":
        view_dirs = sorted([os.path.join(dataset_root, d) for d in os.listdir(dataset_root)
                            if os.path.isdir(os.path.join(dataset_root, d))])
        for subdir in view_dirs:
            if not os.path.exists(os.path.join(subdir, "intrinsics_extrinsics.npz")):
                continue
            vname = os.path.basename(subdir)
            from .gt import load_gt_params
            try:
                K, _ = load_gt_params(subdir, dataset_type=dataset_type)
                key = tuple(np.round(K.flatten(), decimals=3))
                cache[key] = vname
            except:
                pass
    elif dataset_type == "hi4d":
        cam_path = os.path.join(dataset_root, "cameras", "rgb_cameras.npz")
        if os.path.exists(cam_path):
            data = np.load(cam_path)
            ids = data['ids']
            intrinsics = data['intrinsics']
            for i, cid in enumerate(ids):
                K = intrinsics[i]
                key = tuple(np.round(K.flatten(), decimals=3))
                cache[key] = str(cid)

    _CAMERA_CACHE[cache_key] = cache
    return cache


def discover_view_name(dataset_root, K, dataset_type="dex-ycb"):
    """
    Given an intrinsic matrix K, return the folder name (e.g. 'view_01').
    """
    cache = build_camera_intrinsics_cache(dataset_root, dataset_type=dataset_type)
    key = tuple(np.round(K.flatten(), decimals=3))
    return cache.get(key)


def get_rgb_path(view_dir: str, frame_t: int, dataset_type="dex-ycb") -> str | None:
    """
    Robust RGB path discovery given a view directory and a frame index.
    """
    if dataset_type == "dex-ycb":
        rgb_dir = os.path.join(view_dir, "rgb") if os.path.isdir(os.path.join(view_dir, "rgb")) else view_dir
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(rgb_dir, f"{frame_t:05d}{ext}")
            if os.path.exists(p):
                return p
    elif dataset_type == "hi4d":
        # view_dir is .../pairXX/actionXX/ID
        # images are in .../pairXX/actionXX/images/ID/000XXX.jpg
        action_dir = os.path.dirname(view_dir)
        cam_id = os.path.basename(view_dir)
        rgb_dir = os.path.join(action_dir, "images", cam_id)
        p = os.path.join(rgb_dir, f"{frame_t:06d}.jpg")
        if os.path.exists(p):
            return p

    return None
