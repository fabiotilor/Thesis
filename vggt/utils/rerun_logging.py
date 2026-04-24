import os
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .gt import load_gt_params, build_gt_validity_masks
from .camera_utils import discover_view_name
from .umeyama_alignment import apply_similarity_transform
from eval_config import CONF_PERCENTILE

_RAW_COLOURS = [
    [255, 128, 0],  # orange
    [128, 0, 255],  # purple
    [0, 200, 200],  # teal
    [200, 200, 0],  # yellow
]


def init_recording(subject_code: str, n_views: int) -> None:
    """
    Initialise a fresh Rerun recording for one (subject, view-count) pair.

    Call this once before processing each (subject, n_views) combination.
    Each call creates an independent entry in the Rerun Sources panel, e.g.:

        Local
          mast3r_01_2views   10:55:18 — 35.3 MiB
          mast3r_01_3views   10:55:44 — 38.1 MiB
          mast3r_01_4views   10:56:12 — 41.0 MiB

    The entity paths and timeline used by every other function in this module
    are unaffected — they always operate on whichever recording is current.
    """
    application_id = f"vggt_{subject_code}_{n_views}views"
    try:
        from eval_config import RERUN_ADDR
        rr.init(application_id, spawn=False)
        rr.connect_grpc(RERUN_ADDR)
    except Exception as e:
        print(f"  [WARN] Failed to connect to Rerun: {e}")


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


def log_cameras_rerun(t, view_names, dataset_root, log_root):
    """
    Logs pinhole cameras with RGB image content.
    Expects dataset_root/{vname}/rgb/{t:05d}.png
    """
    rr.set_time("frame", sequence=t)
    for vname in view_names:
        view_dir = os.path.join(dataset_root, vname)
        K, c2w = load_gt_params(view_dir)

        rgb_dir = os.path.join(view_dir, "rgb")
        if not os.path.isdir(rgb_dir):
            rgb_dir = view_dir

        rgb_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(rgb_dir, f"{t:05d}{ext}")
            if os.path.exists(p):
                rgb_path = p
                break

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

    if positions is not None and len(positions) > max_points:
        idx = np.random.choice(len(positions), max_points, replace=False)
        if color is not None and isinstance(color, np.ndarray) and len(color) == len(positions):
            color = color[idx]
        positions = positions[idx]

    kwargs = {"positions": positions, "radii": radii}
    if color is not None:
        kwargs["colors"] = color
    try:
        rr.log(entity, rr.Points3D(**kwargs))
    except Exception as e:
        print(f"  [WARN] Rerun logging failed for {entity}: {e}")


def log_alignment_results(t, gt_pts, aligned_pts, refined_pts=None, log_root="world"):
    """Used by 3D alignment scripts to visualise registration quality."""
    if gt_pts is not None:
        log_pointcloud(t, f"{log_root}/gt", gt_pts, color=[0, 255, 0])
    if aligned_pts is not None:
        log_pointcloud(t, f"{log_root}/baseline/pointcloud", aligned_pts, color=[0, 0, 255])
    if refined_pts is not None:
        log_pointcloud(t, f"{log_root}/estimated/stabilised", refined_pts, color=[255, 0, 255])


def log_gt_sequence(paths, log_root="4d_eval"):
    """Logs the GT sequence from a list of NPZ paths."""
    entity = f"{log_root}/gt"
    for p in paths:
        data = np.load(p)
        t = int(data['frame_idx'])
        gt_pts = data['gt_pts']
        if np.any(np.linalg.norm(gt_pts, axis=-1) > 10.0):
            gt_pts = gt_pts / 1000.0
        log_pointcloud(t, entity, gt_pts, color=[0, 255, 0])


def log_aligned_sequence(paths, frame_transforms, s_glob, R_glob, tr_glob, label, color, dataset_root,
                         log_root="4d_eval"):
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
        view_names = [discover_view_name(dataset_root, k) for k in ks]
        vmasks = build_gt_validity_masks(t, view_names, dataset_root, target_hw=(H, W))

        # Log cameras and images only on the first strategy to avoid redundant writes.
        if i == 0:
            log_cameras_rerun(t, view_names, dataset_root, log_root)

        s_i, R_i, tr_i = frame_transforms[i]
        s_tot = s_glob * s_i
        R_tot = R_glob @ R_i
        tr_tot = s_glob * (R_glob @ tr_i) + tr_glob

        # ── Global threshold for the whole frame ──
        frame_thr = np.quantile(conf, 1.0 - CONF_PERCENTILE) if conf is not None else 0.0

        all_pts_final = []
        for v in range(V):
            mask = np.ones((H, W), dtype=bool)
            if vmasks[v] is not None:
                mask &= vmasks[v]
            if conf is not None:
                mask &= (conf[v] > frame_thr)
            p_v = pm[v][mask]
            if len(p_v) > 0:
                all_pts_final.append(apply_similarity_transform(p_v, s_tot, R_tot, tr_tot))

        if all_pts_final:
            merged_pts = np.concatenate(all_pts_final, axis=0)
            log_pointcloud(t, f"{entity_root}/pointcloud", merged_pts, color=color)


def log_raw_window(window_idx, frame_indices, view_names,
                   raw_per_view, conf_per_view, log_root):
    """
    Log the unaligned output of one sliding window to Rerun.

    Entity path  <log_root>/raw_windows/win_<N>/<vname>/pointcloud

    Shows whether the model is internally consistent across views within
    a single window — before any GT registration.
    """
    for fi, frame_t in enumerate(frame_indices):
        rr.set_time("frame", sequence=frame_t)
        for vi, vname in enumerate(view_names):
            pts = raw_per_view[vname][fi].reshape(-1, 3)
            conf = conf_per_view[vname][fi].ravel()
            # ── Global threshold for the whole frame ──
            frame_thr = np.quantile(conf_per_view[vi], 1.0 - CONF_PERCENTILE)
            valid = conf > frame_thr
            if not valid.any():
                continue
            entity = f"{log_root}/raw_windows/win_{window_idx:02d}/{vname}/pointcloud"
            try:
                rr.log(entity, rr.Points3D(
                    positions=pts[valid],
                    radii=0.002,
                    colors=np.array(_RAW_COLOURS[vi % len(_RAW_COLOURS)],
                                    dtype=np.uint8),
                ))
            except Exception:
                pass


def log_fused_frame(t, merged_pts, log_root):
    """
    Log the final fused, aligned pointcloud for one frame.

    Entity path  <log_root>/fused_aligned/pointcloud
    """
    if merged_pts is None or len(merged_pts) == 0:
        return
    rr.set_time("frame", sequence=t)
    try:
        rr.log(
            f"{log_root}/fused_aligned/pointcloud",
            rr.Points3D(
                positions=merged_pts,
                radii=0.002,
                colors=np.array([0, 128, 255], dtype=np.uint8),  # blue-cyan
            ),
        )
    except Exception:
        pass