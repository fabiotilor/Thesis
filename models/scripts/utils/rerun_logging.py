import os
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import time

from .gt import load_gt_params, build_gt_validity_masks
from .camera_utils import discover_view_name, get_rgb_path
from .umeyama_alignment import apply_similarity_transform
from eval_config import CONF_PERCENTILE


def init_recording(subject_code: str, n_views: int, model_name: str = "vggt") -> None:
    """
    Initialise a fresh Rerun recording for one (subject, view-count) pair.
    """
    application_id = f"{model_name}_{subject_code}_{n_views}views"

    # Aggressively try to clear previous state
    try:
        rr.disconnect()
        time.sleep(0.5)  # Give gRPC time to settle
    except:
        pass

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
            # Ensure the root exists
            rr.log(log_root, rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
            return
        except Exception:
            continue


def log_cameras_rerun(t, view_names, dataset_root, log_root, dataset_type="dex-ycb"):
    """
    Logs pinhole cameras with RGB image content.
    Expects dataset_root/{vname}/rgb/{t:05d}.png
    """
    rr.set_time("frame", sequence=t)
    for vname in view_names:
        if vname is None: continue
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


def log_pointcloud(t, entity, positions, color=None, radii=0.002, max_points=25000):
    """Basic reusable pointcloud logger. Lowered max_points to 25k to reduce channel load."""
    rr.set_time("frame", sequence=t)
    if len(positions) > max_points:
        idx = np.random.choice(len(positions), max_points, replace=False)
        positions = positions[idx]
    kwargs = {"positions": positions, "radii": radii}
    if color is not None:
        kwargs["colors"] = color
    rr.log(entity, rr.Points3D(**kwargs))


def log_alignment_results(t, gt_pts, aligned_pts, refined_pts=None, log_root="world"):
    """Used by 3D alignment scripts to visualise registration quality."""
    if gt_pts is not None:
        log_pointcloud(t, f"{log_root}/gt", gt_pts, color=[0, 255, 0])
    if aligned_pts is not None:
        log_pointcloud(t, f"{log_root}/baseline/pointcloud", aligned_pts, color=[0, 0, 255])
    if refined_pts is not None:
        log_pointcloud(t, f"{log_root}/estimated/stabilised", refined_pts, color=[255, 0, 255])

    # Moderate sleep to prevent backpressure
    time.sleep(0.01)


def log_gt_sequence(paths, dataset_root, dataset_type="dex-ycb", log_root="4d_eval"):
    """Logs the GT sequence from a list of NPZ paths.

    For Hi4D we use the pre-stored gt_pts from the .npz (written by
    run_ggpt_refinement.py) rather than re-loading from disk.  The stored
    gt_pts are guaranteed to use the same frame mapping as the model outputs;
    re-loading with `frame_idx + _get_hi4d_offset()` can be wrong when the
    GGPT inputs were prepared with a stride or a different start frame.
    """
    from .gt import load_gt_params, DEPTH_SCALE, DEPTH_MAX_M
    from .camera_utils import discover_view_name
    entity = f"{log_root}/gt"
    entity_static = f"{log_root}/gt_static"
    for p in paths:
        data = np.load(p)
        t = int(data['frame_idx'])

        if dataset_type == "hi4d":
            # Use the pre-stored gt_pts — correct frame mapping guaranteed.
            gt_pts = np.array(data['gt_pts'])
            if gt_pts.ndim > 2:
                # Stored as (V, H, W, 3) pointmap — flatten valid (non-zero) points.
                gt_pts = gt_pts.reshape(-1, 3)
                gt_pts = gt_pts[np.linalg.norm(gt_pts, axis=-1) > 1e-6]
            if len(gt_pts) == 0:
                gt_pts = np.zeros((1, 3))
        else:
            gt_pts = data['gt_pts']
            if np.any(np.linalg.norm(gt_pts, axis=-1) > 10.0):
                gt_pts = gt_pts / 1000.0

        log_pointcloud(t, entity, gt_pts, color=[0, 255, 0])

        # Static GT (dex-ycb only — depth-based)
        if dataset_type != "hi4d" and 'masks_2d' in data:
            view_names = [discover_view_name(dataset_root, k, dataset_type=dataset_type) for k in data['Ks']]
            static_mask = data['masks_2d']
            all_pts_static = []
            for i, vname in enumerate(view_names):
                if vname is None: continue
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

        # Moderate sleep
        time.sleep(0.01)


def log_aligned_sequence(paths, frame_transforms, s_glob, R_glob, tr_glob, label, color, dataset_root,
                         dataset_type="dex-ycb", log_root="4d_eval"):
    """
    Robust 4D pointcloud logger. Handles inter-frame and global alignment composition.

    NOTE: The pointmaps in `paths` are the raw (unfiltered) outputs from the model.
    We apply only the GT validity mask here (segmentation / depth mask) — we do NOT
    re-apply confidence thresholding because that was already done in
    run_baseline_alignment / run_strategy_alignment when selecting correspondence
    points. Re-applying it here would cause inconsistent filtering frame-to-frame.
    """
    from .alignment_4D import normalize_spatial_dims, normalize_array

    entity_root = f"{log_root}/{label}"
    for i, p in enumerate(paths):
        data = np.load(p)
        V, H, W = normalize_spatial_dims(data)
        if H == 0:
            continue

        pm = normalize_array(data['pointmaps'], V, H, W).astype(np.float32)

        t = int(data['frame_idx'])

        # Prefer explicit view_names stored in the .npz.
        if 'view_names' in data:
            view_names = [v.decode() if isinstance(v, bytes) else str(v)
                          for v in data['view_names']]
        else:
            ks = data['Ks']
            view_names = [discover_view_name(dataset_root, k, dataset_type=dataset_type) for k in ks]

        # For Hi4D: use masks_2d from the .npz directly (= geo_msks written by
        # run_ggpt_refinement.py, correctly frame-indexed).  Re-loading seg masks
        # via build_gt_validity_masks uses t+offset which can be wrong if the GGPT
        # inputs were prepared with a stride or different start frame.
        if dataset_type == "hi4d" and 'masks_2d' in data:
            from .alignment_4D import normalize_array
            raw_m = normalize_array(data['masks_2d'], V, H, W, is_mask=True)
            vmasks = [raw_m[v] if raw_m[v].any() else None for v in range(V)]
        else:
            vmasks = build_gt_validity_masks(t, view_names, dataset_root, target_hw=(H, W),
                                             dataset_type=dataset_type)

        # Log cameras and images only on the first frame to avoid redundant writes.
        if i == 0:
            log_cameras_rerun(t, view_names, dataset_root, log_root, dataset_type=dataset_type)

        s_i, R_i, tr_i = frame_transforms[i]
        s_tot = s_glob * s_i
        R_tot = R_glob @ R_i
        tr_tot = s_glob * (R_glob @ tr_i) + tr_glob

        all_pts_final = []
        for v in range(V):
            # Apply only the stored validity mask — no confidence re-filtering.
            mask = np.ones((H, W), dtype=bool)
            if vmasks[v] is not None:
                mask &= vmasks[v]
            p_v = pm[v][mask]
            if len(p_v) > 0:
                all_pts_final.append(apply_similarity_transform(p_v, s_tot, R_tot, tr_tot))

        if all_pts_final:
            merged_pts = np.concatenate(all_pts_final, axis=0)
            log_pointcloud(t, f"{entity_root}/pointcloud", merged_pts, color=color)

        # Moderate sleep to prevent backpressure
        time.sleep(0.02)
