import os
import glob
import numpy as np
from collections import defaultdict

_CAMERA_CACHE = {}


def build_camera_intrinsics_cache(dataset_root):
    """
    Builds a mapping from (rounded) K.flatten() to the folder name (e.g. 'view_01').
    """
    subject_name = os.path.basename(dataset_root)
    if subject_name in _CAMERA_CACHE:
        return _CAMERA_CACHE[subject_name]

    cache = {}
    view_dirs = sorted([os.path.join(dataset_root, d) for d in os.listdir(dataset_root)
                        if os.path.isdir(os.path.join(dataset_root, d))])
    for subdir in view_dirs:
        # Check if the folder contains the necessary GT metadata
        if not os.path.exists(os.path.join(subdir, "intrinsics_extrinsics.npz")):
            continue
        # Try to find any frame NPZ to get the K
        vname = os.path.basename(subdir)
        # In DexYCB, we can usually find one NPZ in the aligned_outputs if they were saved
        # But better to check the metadata if available, OR just check the view_X folder.
        # Here we look for the first depth or any frame to get K?
        # Actually, in DexYCB we often have a fixed K per view.
        # We'll stick to the logic used in 4D_Umeyama.py which expected NPZs in a certain place,
        # but optimized for the general case.
        from .gt import load_gt_params
        try:
            K, _ = load_gt_params(subdir)
            key = tuple(np.round(K.flatten(), decimals=3))
            cache[key] = vname
        except:
            pass

    _CAMERA_CACHE[subject_name] = cache
    return cache


def discover_view_name(dataset_root, K):
    """
    Given an intrinsic matrix K, return the folder name (e.g. 'view_01').
    """
    cache = build_camera_intrinsics_cache(dataset_root)
    key = tuple(np.round(K.flatten(), decimals=3))
    return cache.get(key)


def build_views(dataset_root, target_views=None):
    """
    Build a {view_name: [frame_path, ...]} mapping.

    Parameters
    ----------
    dataset_root : str  — path to the subject directory
    target_views : list[str] or None
        If provided, only include views matching these suffixes
        (e.g. ["01", "06"]).

    Returns
    -------
    dict[str, list[str]]
    """
    img_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    views = defaultdict(list)
    dirs = (
        [os.path.join(dataset_root, f"view_{v}") for v in target_views]
        if target_views
        else sorted(glob.glob(os.path.join(dataset_root, "view_*")))
    )
    for vd in dirs:
        if not os.path.isdir(vd):
            continue
        vname = os.path.basename(vd)
        rgb_dir = os.path.join(vd, "rgb")
        search = rgb_dir if os.path.isdir(rgb_dir) else vd
        frames = sorted(
            f
            for f in glob.glob(os.path.join(search, "*"))
            if os.path.splitext(f.lower())[1] in img_exts
        )
        if frames:
            views[vname] = frames
    return dict(views)
