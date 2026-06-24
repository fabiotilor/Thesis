import os
import glob
import numpy as np

_CAMERA_CACHE = {}


def build_camera_intrinsics_cache(dataset_root, dataset_type="dex-ycb"):
    """
    Builds a mapping from (rounded) K.flatten() to the folder name (e.g. 'view_01').
    """
    subject_name = os.path.basename(dataset_root)
    if subject_name in _CAMERA_CACHE:
        return _CAMERA_CACHE[subject_name]

    cache = {}
    # Try Hi4D-style multi-camera NPZ first
    hi4d_cameras = os.path.join(dataset_root, "cameras", "rgb_cameras.npz")
    if os.path.exists(hi4d_cameras):
        data = np.load(hi4d_cameras)
        print(f"  [DEBUG] Hi4D camera file keys: {list(data.keys())}")

        # Hi4D: try different possible key names for intrinsics
        if 'intrinsic' in data:
            intrinsics = data['intrinsic']  # (C, 3, 3)
        elif 'intrinsics' in data:
            intrinsics = data['intrinsics']  # (C, 3, 3)
        elif 'K' in data:
            intrinsics = data['K']  # (C, 3, 3)
        else:
            print(f"  [ERROR] No intrinsics found in Hi4D camera file: {hi4d_cameras}")
            return {}

        # Try different possible key names for camera IDs
        if 'ids' in data:
            cam_ids = [str(cid) for cid in data['ids']]
        elif 'cam_ids' in data:
            cam_ids = [str(cid) for cid in data['cam_ids']]
        else:
            print(f"  [ERROR] No camera IDs found in Hi4D camera file: {hi4d_cameras}")
            return {}

        print(f"  [DEBUG] Found {len(cam_ids)} Hi4D cameras: {cam_ids}")
        for i, cid in enumerate(cam_ids):
            key = tuple(np.round(intrinsics[i].flatten(), decimals=3))
            cache[key] = cid
        _CAMERA_CACHE[subject_name] = cache
        return cache

    view_dirs = sorted([os.path.join(dataset_root, d) for d in os.listdir(dataset_root)
                        if os.path.isdir(os.path.join(dataset_root, d))])
    for subdir in view_dirs:
        # Check if the folder contains the necessary GT metadata (DexYCB style)
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
        from mast3r.utils.gt import load_gt_params
        try:
            K, _ = load_gt_params(subdir, dataset_type=dataset_type)
            key = tuple(np.round(K.flatten(), decimals=3))
            cache[key] = vname
        except:
            pass

    _CAMERA_CACHE[subject_name] = cache
    return cache


def discover_view_name(dataset_root, K, dataset_type="dex-ycb"):
    """
    Given an intrinsic matrix K, return the folder name (e.g. 'view_01').
    """
    cache = build_camera_intrinsics_cache(dataset_root, dataset_type=dataset_type)
    key = tuple(np.round(K.flatten(), decimals=3))
    return cache.get(key)


def get_rgb_path(view_dir: str, frame_t: int) -> str | None:
    """
    Robust RGB path discovery given a view directory (e.g., dataset/view_00/)
    and a frame index. Checks both dataset/view_00/{t}.png and dataset/view_00/rgb/{t}.png.
    """
    rgb_dir = os.path.join(view_dir, "rgb") if os.path.isdir(os.path.join(view_dir, "rgb")) else view_dir
    for ext in (".png", ".jpg", ".jpeg"):
        for pad in (5, 6):
            p = os.path.join(rgb_dir, f"{frame_t:0{pad}d}{ext}")
            if os.path.exists(p):
                return p


def remove_outliers(pts, nb_neighbors=20, std_ratio=1.0, return_mask=False):
    """
    Removes sparse noise/floaters using statistical outlier removal.
    """
    if pts is None or len(pts) < nb_neighbors:
        if return_mask:
            return pts, np.ones(len(pts), dtype=bool) if pts is not None else None
        return pts
    try:
        from scipy.spatial import cKDTree
        import numpy as np
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=nb_neighbors)
        mean_dists = dists[:, 1:].mean(axis=1)
        avg = np.mean(mean_dists)
        std = np.std(mean_dists)
        mask = mean_dists < (avg + std_ratio * std)
        if return_mask:
            return pts[mask], mask
        return pts[mask]
    except ImportError:
        if return_mask:
            return pts, np.ones(len(pts), dtype=bool)
        return pts
