import os
import glob
import numpy as np
from collections import defaultdict

_CAMERA_CACHE = {}


def build_camera_intrinsics_cache(dataset_root, dataset_type="dex-ycb"):
    """
    Builds a mapping from (rounded) K.flatten() to the folder name (e.g. 'view_01').
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
            # Check if the folder contains the necessary GT metadata
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
        # Hi4D has cameras in .../pairXX/actionXX/cameras/rgb_cameras.npz
        cam_path = os.path.join(dataset_root, "cameras", "rgb_cameras.npz")
        if not os.path.exists(cam_path):
            # Try one level up if we are in the subject root but the cameras are in action dir
            cam_path = os.path.join(os.path.dirname(dataset_root), "cameras", "rgb_cameras.npz")

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


def build_views(dataset_root, target_views=None, dataset_type="dex-ycb"):
    """
    Build a {view_name: [frame_path, ...]} mapping.

    Parameters
    ----------
    dataset_root : str  — path to the subject directory
    target_views : list[str] or None
        If provided, only include views matching these suffixes
        (e.g. ["01", "16"]).
    dataset_type : str

    Returns
    -------
    dict[str, list[str]]
    """
    img_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    views = defaultdict(list)

    if dataset_type == "dex-ycb":
        dirs = (
            [os.path.join(dataset_root, f"view_{v}") for v in target_views]
            if target_views
            else sorted(glob.glob(os.path.join(dataset_root, "view_*")))
        )
    elif dataset_type == "hi4d":
        # Hi4D views are just camera IDs (e.g. "16", "40")
        img_root = os.path.join(dataset_root, "images")
        if target_views:
            dirs = [os.path.join(img_root, str(v)) for v in target_views]
        else:
            dirs = sorted(glob.glob(os.path.join(img_root, "*")))
    else:
        dirs = []

    for vd in dirs:
        if not os.path.isdir(vd):
            continue
        vname = os.path.basename(vd)

        if dataset_type == "dex-ycb":
            rgb_dir = os.path.join(vd, "rgb")
            search = rgb_dir if os.path.isdir(rgb_dir) else vd
        else:
            search = vd

        frames = sorted(
            f
            for f in glob.glob(os.path.join(search, "*"))
            if os.path.splitext(f.lower())[1] in img_exts
        )

        if dataset_type == "hi4d" and frames:
            from eval_config import HI4D_START_FRAME, HI4D_STEP_SIZE, HI4D_TOTAL_FRAMES
            # Slice according to Hi4D indexing rules
            # Start frame is 1-indexed in some contexts but here we assume it corresponds to the filename/sort order
            # If the user says start at 22, and we have 000000.jpg, 000001.jpg... then index 22 is 000022.jpg
            frames = frames[HI4D_START_FRAME: HI4D_START_FRAME + HI4D_TOTAL_FRAMES * HI4D_STEP_SIZE: HI4D_STEP_SIZE]

        if frames:
            views[vname] = frames
    return dict(views)
