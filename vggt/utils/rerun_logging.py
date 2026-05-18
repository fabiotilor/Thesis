import os
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .gt import load_gt_params, build_gt_validity_masks, _load_hi4d_seg_mask
from .camera_utils import discover_view_name, get_rgb_path
from .umeyama_alignment import apply_similarity_transform
from eval_config import CONF_PERCENTILE


def init_recording(subject_code: str, n_views: int) -> None:
    """
    Initialise a fresh Rerun recording for one (subject, view-count) pair.
    """
    application_id = f"vggt_{subject_code}_{n_views}views"
    rr.init(application_id, spawn=False)
    try:
        from eval_config import RERUN_ADDR
        rr.connect_grpc(RERUN_ADDR)
    except Exception as e:
        print(f"[WARN] Rerun connection failed: {e}")


def configure_rerun_view_defaults(log_root, eye_up):
    """
    Sets the best-effort default 3D view orientation for the given log root.
    """
    blueprint_variants = []

    # Variant 1: EyeControls3D as direct symbol.
    try:
        eye_controls = rrb.EyeControls3D(eye_up=eye_up)
        blueprint_variants.append(
            rrb.Blueprint(rrb.Spatial3DView(origin=log_root, name=f"{log_root}_3d", eye_controls=eye_controls))
        )
    except Exception:
        pass

    # Variant 2: EyeControls3D under archetypes namespace.
    try:
        eye_controls = rrb.archetypes.EyeControls3D(eye_up=eye_up)
        blueprint_variants.append(
            rrb.Blueprint(rrb.Spatial3DView(origin=log_root, name=f"{log_root}_3d", eye_controls=eye_controls))
        )
    except Exception:
        pass

    # Fallback variants for compatibility.
    for kwargs in (
            {"origin": log_root, "name": f"{log_root}_3d", "eye_up": eye_up},
            {"origin": log_root, "name": f"{log_root}_3d", "up": eye_up},
            {"origin": log_root, "name": f"{log_root}_3d"},
    ):
        try:
            blueprint_variants.append(rrb.Blueprint(rrb.Spatial3DView(**kwargs)))
        except Exception:
            continue

    for blueprint in blueprint_variants:
        try:
            rr.send_blueprint(blueprint)
            return
        except Exception:
            continue


def log_cameras_rerun(t, view_names, dataset_root, log_root, dataset_type="dex-ycb"):
    """
    Logs pinhole cameras with RGB image content.
    """
    rr.set_time("frame", sequence=t)
    for vname in view_names:
        view_dir = os.path.join(dataset_root, vname)
        K, c2w = load_gt_params(view_dir, dataset_type=dataset_type)

        rgb_path = get_rgb_path(view_dir, t, dataset_type=dataset_type)

        if rgb_path:
            img_bgr = cv2.imread(rgb_path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                H, W = img_rgb.shape[:2]

                entity = f"{log_root}/cameras/{vname}"
                rr.log(entity, rr.Pinhole(image_from_camera=K, width=W, height=H, image_plane_distance=0.2))
                rr.log(entity, rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]))
                rr.log(f"{entity}/rgb", rr.Image(img_rgb))
            else:
                print(f"  [WARN] Failed to read image: {rgb_path}")
        else:
            print(f"  [WARN] Image not found for {vname} at t={t}")


def log_pointcloud(t, entity, positions, color=None, radii=0.002, max_points=50000):
    """Basic reusable pointcloud logger."""
    rr.set_time("frame", sequence=t)
    if len(positions) > max_points:
        idx = np.random.choice(len(positions), max_points, replace=False)
        positions = positions[idx]
        if color is not None and isinstance(color, np.ndarray) and len(color) == len(positions):
            pass  # Handle per-point colors if we ever use them
    kwargs = {"positions": positions, "radii": radii}
    if color is not None:
        kwargs["colors"] = color
    rr.log(entity, rr.Points3D(**kwargs))


def log_alignment_results(t, gt_pts, aligned_pts, refined_pts=None, gt_static_pts=None, log_root="world"):
    """Used by 3D alignment scripts to visualise registration quality."""
    if gt_pts is not None:
        log_pointcloud(t, f"{log_root}/gt", gt_pts, color=[0, 255, 0])
    if aligned_pts is not None:
        log_pointcloud(t, f"{log_root}/baseline/pointcloud", aligned_pts, color=[0, 0, 255])
    if refined_pts is not None:
        log_pointcloud(t, f"{log_root}/estimated/stabilised", refined_pts, color=[255, 0, 255])
    if gt_static_pts is not None:
        log_pointcloud(t, f"{log_root}/gt_static", gt_static_pts, color=[255, 165, 0])


def log_gt_sequence(paths, dataset_root=None, log_root="4d_eval", dataset_type="dex-ycb"):
    """Logs the GT sequence from a list of NPZ paths."""
    from .gt import load_gt_params, DEPTH_SCALE, DEPTH_MAX_M
    from .camera_utils import discover_view_name
    entity = f"{log_root}/gt"
    entity_static = f"{log_root}/gt_static"
    for p in paths:
        data = np.load(p)
        t = int(data['frame_idx'])
        gt_pts = data['gt_pts']
        if np.any(np.linalg.norm(gt_pts, axis=-1) > 10.0):
            gt_pts = gt_pts / 1000.0
        log_pointcloud(t, entity, gt_pts, color=[0, 255, 0])

        if dataset_type == "hi4d":
            # Hi4D has no depth-based static GT; skip static GT logging
            continue

        if 'masks_2d' in data and dataset_root is not None:
            view_names = [discover_view_name(dataset_root, k, dataset_type=dataset_type) for k in data['Ks']]
            static_mask = data['masks_2d']
            all_pts_static = []
            for i, vname in enumerate(view_names):
                view_dir = os.path.join(dataset_root, vname)
                depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
                if not os.path.exists(depth_path): continue
                depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_raw is None: continue
                depth_m = depth_raw.astype(np.float32) * DEPTH_SCALE

                mask_2d = static_mask[i]
                if mask_2d.shape != depth_m.shape:
                    mask_2d = cv2.resize(mask_2d.astype(np.uint8), (depth_m.shape[1], depth_m.shape[0]),
                                         interpolation=cv2.INTER_NEAREST).astype(bool)

                keep = (depth_m > 0) & mask_2d
                ys, xs = np.where(keep)
                z = depth_m[ys, xs]
                K, cam2world = load_gt_params(view_dir, dataset_type=dataset_type)
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                pts_cam = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=-1)
                pts_world = (cam2world[:3, :3] @ pts_cam.T).T + cam2world[:3, 3]
                all_pts_static.append(pts_world)

            if all_pts_static:
                gt_static = np.concatenate(all_pts_static, axis=0)
                if np.any(np.linalg.norm(gt_static, axis=-1) > 10.0):
                    gt_static = gt_static / 1000.0
                log_pointcloud(t, entity_static, gt_static, color=[255, 165, 0])


def log_aligned_sequence(paths, frame_transforms, s_glob, R_glob, tr_glob, label, color, dataset_root,
                         log_root="4d_eval", dataset_type="dex-ycb"):
    """
    Robust 4D pointcloud logger. Handles inter-frame and global alignment composition.
    """
    from .alignment_4d import normalize_spatial_dims, normalize_array

    entity_root = f"{log_root}/{label}"
    for i, p in enumerate(paths):
        data = np.load(p)
        V, H, W = normalize_spatial_dims(data)
        if H == 0:
            continue

        pm = normalize_array(data['pointmaps'], V, H, W).astype(np.float32)
        conf = normalize_array(data['pointmaps_confs'], V, H, W) if 'pointmaps_confs' in data else None

        t, ks = int(data['frame_idx']), data['Ks']
        view_names = [discover_view_name(dataset_root, k, dataset_type=dataset_type) for k in ks]
        vmasks = build_gt_validity_masks(t, view_names, dataset_root, target_hw=(H, W), dataset_type=dataset_type)

        # Log cameras and images only on the first strategy to avoid redundant writes.
        if i == 0:
            log_cameras_rerun(t, view_names, dataset_root, log_root, dataset_type=dataset_type)

        s_i, R_i, tr_i = frame_transforms[i]
        s_tot = s_glob * s_i
        R_tot = R_glob @ R_i
        tr_tot = s_glob * (R_glob @ tr_i) + tr_glob

        all_pts_final = []
        for v in range(V):
            mask = np.ones((H, W), dtype=bool)
            if vmasks[v] is not None:
                mask &= vmasks[v]
            if conf is not None:
                thr = np.percentile(conf[v], 100 * (1 - CONF_PERCENTILE))
                mask &= (conf[v] > thr)
            p_v = pm[v][mask]
            if len(p_v) > 0:
                all_pts_final.append(apply_similarity_transform(p_v, s_tot, R_tot, tr_tot))

        if all_pts_final:
            merged_pts = np.concatenate(all_pts_final, axis=0)
            log_pointcloud(t, f"{entity_root}/pointcloud", merged_pts, color=color)