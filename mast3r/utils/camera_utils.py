import os
import glob
import numpy as np

_CAMERA_CACHE = {}


def build_camera_intrinsics_cache(dataset_root):
    """
    Builds a mapping from (rounded) K.flatten() to the folder name (e.g. 'view_01').
    """
    subject_name = os.path.basename(dataset_root)
    if subject_name in _CAMERA_CACHE:
        return _CAMERA_CACHE[subject_name]

    cache = {}
    view_dirs = sorted(glob.glob(os.path.join(dataset_root, "view_*")))
    for subdir in view_dirs:
        if not os.path.isdir(subdir): continue
        # Try to find any frame NPZ to get the K
        vname = os.path.basename(subdir)
        # In DexYCB, we can usually find one NPZ in the aligned_outputs if they were saved
        # But better to check the metadata if available, OR just check the view_X folder.
        # Here we look for the first depth or any frame to get K?
        # Actually, in DexYCB we often have a fixed K per view.
        # We'll stick to the logic used in 4D_Umeyama.py which expected NPZs in a certain place,
        # but optimized for the general case.
        from mast3r.utils.gt import load_gt_params
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
