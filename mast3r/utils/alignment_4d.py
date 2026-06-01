import os
import numpy as np
import cv2
from .umeyama_alignment import estimate_similarity_transform, apply_similarity_transform
from .gt import build_gt_validity_masks, DEPTH_MAX_M, load_gt_params
from .camera_utils import discover_view_name
from .temporal_metrics import compute_static_jitter
from eval_config import CONF_PERCENTILE

# Ensure DUSt3R and Mast3r paths are initialized
try:
    from . import path_to_dust3r  # noqa
except ImportError:
    pass


def normalize_spatial_dims(data):
    """Detects canonical (V, H, W) from NPZ data."""
    if 'pointmaps_confs' in data:
        conf = data['pointmaps_confs']
        if conf.ndim == 3: return conf.shape
        if conf.ndim == 2: return 1, conf.shape[0], conf.shape[1]
    pm = data['pointmaps']
    if pm.ndim == 4: return pm.shape[:3]
    if pm.ndim == 3:
        V, HW = pm.shape[0], pm.shape[1]
        if HW == 196608: return V, 384, 512
        if HW == 307200: return V, 480, 640
    return 0, 0, 0


def normalize_array(arr, V, H, W, is_mask=False):
    """Standardizes array shape to (V, H, W, ...)."""
    arr = np.array(arr)
    if arr.ndim >= 3 and arr.shape[:3] == (V, H, W): return arr
    if arr.ndim >= 2 and arr.shape[1] == H * W:
        return arr.reshape((V, H, W) + arr.shape[2:])
    if arr.ndim == 2 and arr.shape == (H, W):
        return np.stack([arr] * V)
    if arr.ndim >= 3 and (arr.shape[1] != H or arr.shape[2] != W):
        res = [cv2.resize(arr[i].astype(np.float32), (W, H),
                          interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR)
               for i in range(arr.shape[0])]
        res = np.stack(res)
        return res.astype(bool) if is_mask else res
    return arr


def extract_clean_gt_correspondences(data, dataset_root, n_samples=2048, use_static_mask=True, vmasks=None, **kwargs):
    """
    Extracts cleaner correspondences for GT registration.
    Branches between DexYCB (depth-based) and Hi4D (mesh-projection based).
    """
    dataset_type = kwargs.get('dataset_type', 'dex-ycb')
    V, H_mod, W_mod = normalize_spatial_dims(data)
    if H_mod == 0: return None

    pm_est = normalize_array(data['pointmaps'], V, H_mod, W_mod).astype(np.float32)
    conf_est = normalize_array(data['pointmaps_confs'], V, H_mod, W_mod) if 'pointmaps_confs' in data else None
    m_static = normalize_array(data['masks_2d'], V, H_mod, W_mod, is_mask=True)

    t, ks_gt = int(data['frame_idx']), data['Ks']

    # Use view names directly from NPZ data if available, otherwise discover them
    if 'view_names' in data:
        view_names = data['view_names'].tolist() if hasattr(data['view_names'], 'tolist') else list(data['view_names'])
    else:
        # Fallback to discovering view names from intrinsics
        view_names = [discover_view_name(dataset_root, k) for k in ks_gt]

    if vmasks is None:
        vmasks = build_gt_validity_masks(t, view_names, dataset_root, target_hw=(H_mod, W_mod),
                                         dataset_type=dataset_type)

    all_src, all_dst = [], []
    rng = np.random.default_rng(42)

    for v in range(V):
        if view_names[v] is None or vmasks[v] is None: continue

        if dataset_type == "hi4d":
            # Hi4D: Mesh-to-2D projection logic
            from .gt import _load_hi4d_mesh_gt
            from .camera_utils import get_rgb_path
            vertices = _load_hi4d_mesh_gt(dataset_root, t)
            if vertices is None: continue

            view_dir = os.path.join(dataset_root, "images", view_names[v])
            K_orig, cam2world = load_gt_params(view_dir, dataset_type=dataset_type)
            world2cam = np.linalg.inv(cam2world)
            v_cam = (world2cam[:3, :3] @ vertices.T).T + world2cam[:3, 3]
            valid_z = v_cam[:, 2] > 0
            if not np.any(valid_z): continue
            v_cam = v_cam[valid_z];
            v_world = vertices[valid_z]

            # Scale K
            rgb_path = get_rgb_path(view_dir, t)
            sc_x, sc_y = W_mod / (1280 if not rgb_path else cv2.imread(rgb_path).shape[1]), H_mod / (
                940 if not rgb_path else cv2.imread(rgb_path).shape[0])
            fx_s, fy_s = K_orig[0, 0] * sc_x, K_orig[1, 1] * sc_y
            cx_s, cy_s = K_orig[0, 2] * sc_x, K_orig[1, 2] * sc_y

            u = np.round((v_cam[:, 0] * fx_s / v_cam[:, 2]) + cx_s).astype(int)
            vp = np.round((v_cam[:, 1] * fy_s / v_cam[:, 2]) + cy_s).astype(int)

            ib = np.full((H_mod, W_mod), -1, dtype=int)
            valid_uv = (u >= 0) & (u < W_mod) & (vp >= 0) & (vp < H_mod)
            u_v, v_v, z_v = u[valid_uv], vp[valid_uv], v_cam[:, 2][valid_uv]
            indices = np.arange(len(u))[valid_uv]
            sort_idx = np.argsort(z_v)[::-1]
            ib[v_v[sort_idx], u_v[sort_idx]] = indices[sort_idx]

            mask_final = (ib >= 0) & vmasks[v]
            if conf_est is not None:
                thr = np.percentile(conf_est[v], 100 * (1 - CONF_PERCENTILE))
                mask_final &= (conf_est[v] > thr)

            ys, xs = np.where(mask_final)
            all_src.append(pm_est[v][ys, xs])
            all_dst.append(v_world[ib[ys, xs]])
            continue

        view_dir = os.path.join(dataset_root, view_names[v])

        # Load GT depth at sensor resolution
        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
        if not os.path.exists(depth_path): continue
        d_img_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        H_gt, W_gt = d_img_gt.shape

        # Exact scaling logic from gt.py
        K, c2w = load_gt_params(view_dir, dataset_type=dataset_type)
        scale_x, scale_y = W_mod / W_gt, H_mod / H_gt
        fx_s, fy_s, cx_s, cy_s = K[0, 0] * scale_x, K[1, 1] * scale_y, K[0, 2] * scale_x, K[1, 2] * scale_y

        # Downsample GT depth to model resolution
        d_mod_gt = cv2.resize(d_img_gt, (W_mod, H_mod), interpolation=cv2.INTER_NEAREST)

        # Build total mask for this view
        valid = (d_mod_gt > 0) & vmasks[v]
        if use_static_mask:
            valid &= m_static[v]
        if conf_est is not None:
            thr = np.percentile(conf_est[v], 100 * (1 - CONF_PERCENTILE))
            valid &= (conf_est[v] > thr)

        ys, xs = np.where(valid)
        if len(ys) < 6: continue

        # Sample
        idx = rng.choice(len(ys), size=min(len(ys), n_samples), replace=False)
        ys, xs = ys[idx], xs[idx]

        # GT Point World
        z_gt = d_mod_gt[ys, xs]
        pts_cam_gt = np.stack([(xs - cx_s) * z_gt / fx_s, (ys - cy_s) * z_gt / fy_s, z_gt], axis=-1)
        pts_world_gt = (c2w[:3, :3] @ pts_cam_gt.T).T + c2w[:3, 3]

        # Est Point (already in its own system)
        pts_est = pm_est[v][ys, xs]

        all_src.append(pts_est)
        all_dst.append(pts_world_gt)

    if not all_src:
        return None
    # Return source (estimate static) and destination (GT static) correspondences
    return np.concatenate(all_src), np.concatenate(all_dst)


def get_pointmap_correspondences(path_a, path_b, dataset_root, vmasks_a=None, vmasks_b=None, **kwargs):
    """Inter-frame alignment using only static/valid pixels."""
    dataset_type = kwargs.get('dataset_type', 'dex-ycb')
    data_a, data_b = np.load(path_a), np.load(path_b)
    V, H, W = normalize_spatial_dims(data_a)
    if H == 0: return None

    pm_a = normalize_array(data_a['pointmaps'], V, H, W).astype(np.float32)
    pm_b = normalize_array(data_b['pointmaps'], V, H, W).astype(np.float32)

    # Simple strategy: align based on static pixels existing in both frames
    m_a = normalize_array(data_a['masks_2d'], V, H, W, is_mask=True)
    m_b = normalize_array(data_b['masks_2d'], V, H, W, is_mask=True)

    # We use vmasks to ensure we only align on high-quality regions (hand/table)
    if vmasks_a is None:
        # Use view names directly from NPZ data if available, otherwise discover them
        if 'view_names' in data_a:
            view_names_a = data_a['view_names'].tolist() if hasattr(data_a['view_names'], 'tolist') else list(
                data_a['view_names'])
        else:
            view_names_a = [discover_view_name(dataset_root, k) for k in data_a['Ks']]
        vmasks_a = build_gt_validity_masks(int(data_a['frame_idx']), view_names_a, dataset_root,
                                           target_hw=(H, W), dataset_type=dataset_type)
    if vmasks_b is None:
        # Use view names directly from NPZ data if available, otherwise discover them
        if 'view_names' in data_b:
            view_names_b = data_b['view_names'].tolist() if hasattr(data_b['view_names'], 'tolist') else list(
                data_b['view_names'])
        else:
            view_names_b = [discover_view_name(dataset_root, k) for k in data_b['Ks']]
        vmasks_b = build_gt_validity_masks(int(data_b['frame_idx']), view_names_b, dataset_root,
                                           target_hw=(H, W), dataset_type=dataset_type)

    conf_a = normalize_array(data_a['pointmaps_confs'], V, H, W) if 'pointmaps_confs' in data_a else None
    conf_b = normalize_array(data_b['pointmaps_confs'], V, H, W) if 'pointmaps_confs' in data_b else None

    src_list, dst_list = [], []
    for v in range(V):
        if vmasks_a[v] is None or vmasks_b[v] is None: continue
        # Intersection of static and valid
        if dataset_type == "hi4d":
            # For Hi4D, we MUST align on the STATIC BACKGROUND (the floor).
            # If we align on the moving person, the sequence becomes person-centric,
            # which fails to match the global motion of the GT mesh.
            mask = m_a[v] & m_b[v]
        else:
            # DexYCB: Align on static background + valid region (within 1.5m)
            mask = m_a[v] & m_b[v] & vmasks_a[v] & vmasks_b[v]

        if conf_a is not None:
            thr_a = np.percentile(conf_a[v], 100 * (1 - CONF_PERCENTILE))
            mask &= (conf_a[v] > thr_a)
        if conf_b is not None:
            thr_b = np.percentile(conf_b[v], 100 * (1 - CONF_PERCENTILE))
            mask &= (conf_b[v] > thr_b)
        ys, xs = np.where(mask)
        if len(ys) > 6:
            src_list.append(pm_b[v][ys, xs])
            dst_list.append(pm_a[v][ys, xs])

    if not src_list: return None
    return np.concatenate(src_list), np.concatenate(dst_list)


def estimate_interframe_transform_pointmap(path_a, path_b, dataset_root, return_error=False, vmasks_a=None,
                                           vmasks_b=None, **kwargs):
    res = get_pointmap_correspondences(path_a, path_b, dataset_root, vmasks_a=vmasks_a, vmasks_b=vmasks_b, **kwargs)
    if res is None:
        return (None, None) if return_error else None

    s, R, tr = estimate_similarity_transform(res[0], res[1])
    if not return_error:
        return s, R, tr

    pred = apply_similarity_transform(res[0], s, R, tr)
    err = np.linalg.norm(pred - res[1], axis=-1).mean()
    return (s, R, tr), err


def strategy1_reference(frame_npz_paths, dataset_root, dataset_type="dex-ycb"):
    n_frames = len(frame_npz_paths)
    transforms = [(1.0, np.eye(3), np.zeros(3))]
    for i in range(1, n_frames):
        res = estimate_interframe_transform_pointmap(
            frame_npz_paths[0], frame_npz_paths[i], dataset_root,
            vmasks_a=None,
            vmasks_b=None,
            dataset_type=dataset_type
        )
        transforms.append(res if res else (1.0, np.eye(3), np.zeros(3)))
    return transforms


def strategy2_hierarchical(frame_npz_paths, dataset_root, dataset_type="dex-ycb"):
    n_frames = len(frame_npz_paths)
    groups = [[(i, (1.0, np.eye(3), np.zeros(3)))] for i in range(n_frames)]
    while len(groups) > 1:
        new_groups = []
        for i in range(0, len(groups) - 1, 2):
            g_a, g_b = groups[i], groups[i + 1]
            path_a, path_b = frame_npz_paths[g_a[0][0]], frame_npz_paths[g_b[0][0]]
            res = estimate_interframe_transform_pointmap(
                path_a, path_b, dataset_root,
                vmasks_a=None,
                vmasks_b=None,
                dataset_type=dataset_type
            )
            s_ba, R_ba, tr_ba = res if res else (1.0, np.eye(3), np.zeros(3))
            merged = list(g_a)
            for idx_b, (s_ib, R_ib, tr_ib) in g_b:
                s_new, R_new, tr_new = s_ba * s_ib, R_ba @ R_ib, s_ba * (R_ba @ tr_ib) + tr_ba
                merged.append((idx_b, (s_new, R_new, tr_new)))
            new_groups.append(merged)
        if len(groups) % 2 != 0: new_groups.append(groups[-1])
        groups = new_groups
    return [t for _, t in sorted(groups[0], key=lambda x: x[0])]


def rotation_average(R_list, weights, max_iters=50, tol=1e-6):
    """Weiszfeld algorithm for geodesic L1 mean on SO(3)"""
    R_mean = R_list[0].copy()
    for _ in range(max_iters):
        v_sum = np.zeros(3)
        w_sum = 0.0
        for R_k, w in zip(R_list, weights):
            R_rel = R_mean.T @ R_k
            # Log map
            trace = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
            theta = np.arccos(trace)
            if theta < 1e-8:
                continue

            v = theta * np.array([R_rel[2, 1] - R_rel[1, 2], R_rel[0, 2] - R_rel[2, 0], R_rel[1, 0] - R_rel[0, 1]]) / (
                    2 * np.sin(theta))

            # Weiszfeld re-weighting
            norm_v = np.linalg.norm(v)
            w_eff = w / max(norm_v, 1e-8)
            v_sum += w_eff * v
            w_sum += w_eff

        if w_sum == 0:
            break

        delta = v_sum / w_sum

        if np.linalg.norm(delta) < tol:
            break

        # Exp map
        theta_d = np.linalg.norm(delta)
        if theta_d > 1e-8:
            n = delta / theta_d
            K = np.array([[0, -n[2], n[1]],
                          [n[2], 0, -n[0]],
                          [-n[1], n[0], 0]])
            R_delta = np.eye(3) + np.sin(theta_d) * K + (1 - np.cos(theta_d)) * (K @ K)
            R_mean = R_mean @ R_delta

    # Cleanup to ensure exact SO(3)
    U, _, Vt = np.linalg.svd(R_mean)
    S = np.eye(3)
    S[2, 2] = np.linalg.det(U) * np.linalg.det(Vt)
    R_mean = U @ S @ Vt

    return R_mean


def invert_similarity_transform(s, R, t):
    """Inverse of y = s * R * x + t."""
    s_inv = 1.0 / max(float(s), 1e-12)
    R_inv = R.T
    t_inv = -s_inv * (R_inv @ t)
    return s_inv, R_inv, t_inv


def compose_similarity_transform(a, b):
    """
    Compose two similarity transforms:
      A: x_a = s_a * R_a * x_b + t_a
      B: x_b = s_b * R_b * x_c + t_b
    Returns C so x_a = C(x_c) = A(B(x_c)).
    """
    s_a, R_a, t_a = a
    s_b, R_b, t_b = b
    s_c = s_a * s_b
    R_c = R_a @ R_b
    t_c = s_a * (R_a @ t_b) + t_a
    return s_c, R_c, t_c


def strategy3_pgo(frame_npz_paths, dataset_root, num_iters=50, dataset_type="dex-ycb"):
    n_frames = len(frame_npz_paths)
    print("    [PGO] Computing T(T-1)/2 pairwise edges...")
    edges = {}

    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            res = estimate_interframe_transform_pointmap(
                frame_npz_paths[i], frame_npz_paths[j], dataset_root,
                return_error=True,
                vmasks_a=None,
                vmasks_b=None,
                dataset_type=dataset_type
            )
            if res[0] is not None:
                (s, R, t), err = res
                weight = 1.0 / (err + 1e-6)
                edges[(i, j)] = ((s, R, t), weight)

    print(f"    [PGO] Found {len(edges)} valid edges. Initializing loops...")

    # Initialize with strategy 1
    T_global = strategy1_reference(frame_npz_paths, dataset_root, dataset_type=dataset_type)
    T_global = [list(val) for val in T_global]

    print("    [PGO] Optimizing...")
    for it in range(num_iters):
        T_new = []
        for i in range(n_frames):
            if i == 0:
                T_new.append(T_global[0])  # anchor at identity
                continue

            votes_s = []
            votes_R = []
            votes_t = []
            weights = []

            for j in range(n_frames):
                if i == j: continue

                s_j, R_j, t_j = T_global[j]

                if (i, j) in edges:
                    # estimate_interframe_transform_pointmap(path_i, path_j) returns edge j -> i
                    (s_ji, R_ji, t_ji), w = edges[(i, j)]
                    # T_i = T_j o inv(E_{j->i})
                    pred = compose_similarity_transform(
                        (s_j, R_j, t_j),
                        invert_similarity_transform(s_ji, R_ji, t_ji)
                    )
                    votes_s.append(pred[0])
                    votes_R.append(pred[1])
                    votes_t.append(pred[2])
                    weights.append(w)

                elif (j, i) in edges:
                    # estimate_interframe_transform_pointmap(path_j, path_i) returns edge i -> j
                    (s_ij, R_ij, t_ij), w = edges[(j, i)]
                    # T_i = T_j o E_{i->j}
                    pred = compose_similarity_transform((s_j, R_j, t_j), (s_ij, R_ij, t_ij))
                    votes_s.append(pred[0])
                    votes_R.append(pred[1])
                    votes_t.append(pred[2])
                    weights.append(w)

            if len(weights) > 0:
                weights = np.array(weights)
                weights /= weights.sum()

                # Scale is multiplicative; average in log-space for better stability.
                safe_scales = np.clip(np.array(votes_s), 1e-8, None)
                new_s = float(np.exp(np.sum(np.log(safe_scales) * weights)))
                new_t = np.sum(np.array(votes_t) * weights[:, None], axis=0)
                new_R = rotation_average(votes_R, weights)
                T_new.append([new_s, new_R, new_t])
            else:
                T_new.append(T_global[i])

        T_global = T_new

    return [(val[0], val[1], val[2]) for val in T_global]


def solve_final_gt_registration(frame_npz_paths, frame_transforms, dataset_root, use_static_mask=True,
                                dataset_type="dex-ycb"):
    print("    [4D-GT] Registration: Precomputing shared validity masks...")
    vmask_cache = {}
    for path in frame_npz_paths:
        data = np.load(path)
        t, V, H, W = int(data['frame_idx']), *normalize_spatial_dims(data)

        # Use view names directly from NPZ data if available, otherwise discover them
        if 'view_names' in data:
            view_names = data['view_names'].tolist() if hasattr(data['view_names'], 'tolist') else list(
                data['view_names'])
        else:
            # Fallback to discovering view names from intrinsics
            view_names = [discover_view_name(dataset_root, k) for k in data['Ks']]

        vmask_cache[path] = build_gt_validity_masks(t, view_names,
                                                    dataset_root, target_hw=(H, W), dataset_type=dataset_type)

    all_src, all_dst = [], []
    for i, path in enumerate(frame_npz_paths):
        data = np.load(path)
        res = extract_clean_gt_correspondences(data, dataset_root, vmasks=vmask_cache[path],
                                               use_static_mask=use_static_mask, dataset_type=dataset_type)
        if res is None: continue
        src, dst = res
        # Apply inter-frame transform to bring to unified space
        s_i, R_i, tr_i = frame_transforms[i]
        src_aligned = apply_similarity_transform(src, s_i, R_i, tr_i)
        all_src.append(src_aligned)
        all_dst.append(dst)

    if not all_src:
        return 1.0, np.eye(3), np.zeros(3)

    # Perform global alignment using the concatenated static points from the entire sequence
    src_concat = np.concatenate(all_src)
    dst_concat = np.concatenate(all_dst)
    s_glob, R_glob, tr_glob = estimate_similarity_transform(src_concat, dst_concat)

    # Diagnostic
    pred = s_glob * (src_concat @ R_glob.T) + tr_glob
    err = np.linalg.norm(pred - dst_concat, axis=-1).mean()
    print(f"  [4D-GT] Registration: Aligned concatenated STATIC points to STATIC GT.")
    print(f"  [4D-GT] Scale: {s_glob:.4f}  Residual Err: {err:.4f}  Corrs: {len(pred):,}")
    return s_glob, R_glob, tr_glob


def compute_4d_jitter_complete(frame_npz_paths, frame_transforms, s_glob, R_glob, tr_glob, dataset_root,
                               dataset_type="dex-ycb"):
    all_pm_mv, all_masks_mv = [], []
    for i, path in enumerate(frame_npz_paths):
        data = np.load(path)
        V, H, W = normalize_spatial_dims(data)
        if H == 0: continue
        pm = normalize_array(data['pointmaps'], V, H, W).astype(np.float32)
        s_i, R_i, tr_i = frame_transforms[i]
        s_tot, R_tot, tr_tot = s_glob * s_i, R_glob @ R_i, s_glob * (R_glob @ tr_i) + tr_glob

        aligned_pm = np.stack(
            [apply_similarity_transform(pm[v].reshape(-1, 3), s_tot, R_tot, tr_tot).reshape(H, W, 3) for v in range(V)])
        all_pm_mv.append(aligned_pm)

        # Stricter static+valid mask for jitter
        m = normalize_array(data['masks_2d'], V, H, W, is_mask=True)
        # Use simple build_gt_validity_masks here as jitter isn't O(N^2)
        vms = build_gt_validity_masks(int(data['frame_idx']), [discover_view_name(dataset_root, k) for k in data['Ks']],
                                      dataset_root, target_hw=(H, W), dataset_type=dataset_type)
        for v in range(V):
            if vms[v] is not None: m[v] &= vms[v]
        all_masks_mv.append(m)

    return compute_static_jitter(all_pm_mv, all_masks_mv)
