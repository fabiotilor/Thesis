import os
import numpy as np
import cv2
from .umeyama_alignment import estimate_similarity_transform, apply_similarity_transform
from .gt import build_gt_validity_masks, DEPTH_MAX_M, load_gt_params, _load_hi4d_seg_mask
from .camera_utils import discover_view_name
from .temporal_metrics import compute_static_jitter
from eval_config import CONF_PERCENTILE


def get_view_names_and_masks(data, dataset_root, dataset_type="dex-ycb"):
    """Robustly resolve view names and build validity masks."""
    V, H, W = normalize_spatial_dims(data)
    t = int(data['frame_idx'])
    if 'view_names' in data:
        view_names = [v.decode() if isinstance(v, bytes) else str(v) for v in data['view_names']]
    else:
        view_names = [discover_view_name(dataset_root, k, dataset_type=dataset_type) for k in data['Ks']]

    valid_idxs = [i for i, v in enumerate(view_names) if v is not None]
    if not valid_idxs:
        return view_names, [None] * V

    vnames_clean = [view_names[i] for i in valid_idxs]
    vmasks_raw = build_gt_validity_masks(t, vnames_clean, dataset_root, target_hw=(H, W),
                                         dataset_type=dataset_type)

    vmasks = [None] * V
    for i_c, i_o in enumerate(valid_idxs):
        vmasks[i_o] = vmasks_raw[i_c]

    return view_names, vmasks


def precompute_vmasks(frame_npz_paths, dataset_root, dataset_type="dex-ycb"):
    """Precompute GT validity masks to avoid redundant disk I/O."""
    vmask_cache = {}
    print("    [I/O] Precomputing validity masks...")
    for path in frame_npz_paths:
        data = np.load(path)
        if normalize_spatial_dims(data)[1] == 0: continue
        _, vmasks = get_view_names_and_masks(data, dataset_root, dataset_type=dataset_type)
        vmask_cache[path] = vmasks
    return vmask_cache


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


def extract_clean_gt_correspondences(data, dataset_root, n_samples=2000,
                                     precomputed_vmasks=None, use_static_mask=True,
                                     dataset_type="dex-ycb"):
    """
    Implements the robust GT projection logic.
    For dex-ycb: matches pointmap pixels to GT back-projected world points using depth.
    For hi4d: uses pre-computed gt_pts pointmap from the data directly.
    """
    V, H_mod, W_mod = normalize_spatial_dims(data)
    if H_mod == 0: return None

    pm_est = normalize_array(data['pointmaps'], V, H_mod, W_mod).astype(np.float32)
    conf_est = normalize_array(data['pointmaps_confs'], V, H_mod, W_mod) if 'pointmaps_confs' in data else None
    m_static = normalize_array(data['masks_2d'], V, H_mod, W_mod, is_mask=True)

    view_names, vmasks = get_view_names_and_masks(data, dataset_root, dataset_type=dataset_type)
    t = int(data['frame_idx'])

    # ── HI4D path: use gt_pts from the data directly ──
    if dataset_type == "hi4d":
        return _extract_correspondences_hi4d(
            data, pm_est, conf_est, m_static, view_names, vmasks,
            V, H_mod, W_mod, t, dataset_root, n_samples
        )

    # ── DexYCB path: load depth from disk ──
    if all(m is None for m in vmasks): return None

    all_src, all_dst = [], []
    rng = np.random.default_rng(42)

    for v in range(V):
        if view_names[v] is None or vmasks[v] is None: continue
        view_dir = os.path.join(dataset_root, view_names[v])

        # Load GT depth at sensor resolution
        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
        if not os.path.exists(depth_path): continue
        d_img_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        H_gt, W_gt = d_img_gt.shape

        # Exact scaling logic from gt.py
        K, c2w = load_gt_params(view_dir, dataset_type=dataset_type)
        scale_x, scale_y = W_mod / W_gt, H_mod / H_gt
        fx_s, fy_s = K[0, 0] * scale_x, K[1, 1] * scale_y
        cx_s, cy_s = K[0, 2] * scale_x, K[1, 2] * scale_y

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

    if not all_src: return None
    return np.concatenate(all_src), np.concatenate(all_dst)


def _extract_correspondences_hi4d(data, pm_est, conf_est, m_static, view_names, vmasks,
                                  V, H_mod, W_mod, t, dataset_root, n_samples):
    """HI4D correspondence extraction using gt_pts pointmap from data."""
    gt_pts_raw = data.get('gt_pts')
    if gt_pts_raw is None:
        return None

    gt_pts_arr = np.array(gt_pts_raw)

    # If gt_pts is (V, H, W, 3) pointmap — pair directly at pixel level
    if gt_pts_arr.ndim == 4 and gt_pts_arr.shape[0] == V:
        gt_pm = normalize_array(gt_pts_arr, V, H_mod, W_mod).astype(np.float32)
        all_src, all_dst = [], []
        rng = np.random.default_rng(42)

        for v in range(V):
            # GT valid where non-zero
            gt_valid = np.linalg.norm(gt_pm[v], axis=-1) > 1e-6
            valid = gt_valid & m_static[v]

            if conf_est is not None:
                min_conf = 0.01
                thr = np.percentile(conf_est[v], 100 * (1 - CONF_PERCENTILE))
                valid &= (conf_est[v] > max(thr, min_conf))

            if vmasks[v] is not None:
                valid &= vmasks[v]

            ys, xs = np.where(valid)
            if len(ys) < 6:
                continue

            idx = rng.choice(len(ys), size=min(len(ys), n_samples), replace=False)
            ys, xs = ys[idx], xs[idx]

            all_src.append(pm_est[v][ys, xs])
            all_dst.append(gt_pm[v][ys, xs])

            if all_src:
                return np.concatenate(all_src), np.concatenate(all_dst)
            # If all_src is empty (e.g. pointmap is all zeros), fall through to mesh fallback
            print("    [HI4D] Pointmap empty, falling back to mesh projection.")

    # If gt_pts is flat mesh vertices or pointmap was empty — use mesh projection (fallback)
    from .gt import _get_correspondences_hi4d

    # Only keep predictions for views that have a valid name
    valid_pts3d = []
    valid_confs = []
    valid_vnames = []

    for v in range(V):
        if view_names[v] is not None:
            valid_vnames.append(view_names[v])
            valid_pts3d.append(pm_est[v])
            valid_confs.append(conf_est[v] if conf_est is not None else np.ones((H_mod, W_mod)))

    if not valid_vnames:
        return None

    res = _get_correspondences_hi4d(t, valid_vnames, valid_pts3d, valid_confs, dataset_root)
    if res[0] is None:
        return None
    return res


def get_pointmap_correspondences(path_a, path_b, dataset_root, vmask_cache=None,
                                 dataset_type="dex-ycb"):
    """Inter-frame alignment using only static/valid pixels."""
    data_a, data_b = np.load(path_a), np.load(path_b)
    V, H, W = normalize_spatial_dims(data_a)
    if H == 0: return None

    pm_a = normalize_array(data_a['pointmaps'], V, H, W).astype(np.float32)
    pm_b = normalize_array(data_b['pointmaps'], V, H, W).astype(np.float32)

    m_a = normalize_array(data_a['masks_2d'], V, H, W, is_mask=True)
    m_b = normalize_array(data_b['masks_2d'], V, H, W, is_mask=True)

    conf_a = normalize_array(data_a['pointmaps_confs'], V, H, W) if 'pointmaps_confs' in data_a else None
    conf_b = normalize_array(data_b['pointmaps_confs'], V, H, W) if 'pointmaps_confs' in data_b else None

    if vmask_cache is not None and path_a in vmask_cache:
        vmasks_a = vmask_cache[path_a]
    else:
        _, vmasks_a = get_view_names_and_masks(data_a, dataset_root, dataset_type=dataset_type)

    if vmask_cache is not None and path_b in vmask_cache:
        vmasks_b = vmask_cache[path_b]
    else:
        _, vmasks_b = get_view_names_and_masks(data_b, dataset_root, dataset_type=dataset_type)

    src_list, dst_list = [], []
    for v in range(V):
        mask = m_a[v] & m_b[v]

        # For hi4d, vmasks might be None (no seg mask available) — skip masking
        if vmasks_a[v] is not None:
            mask &= vmasks_a[v]
        if vmasks_b[v] is not None:
            mask &= vmasks_b[v]

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


def estimate_interframe_transform_pointmap(path_a, path_b, dataset_root,
                                           return_error=False, vmask_cache=None,
                                           dataset_type="dex-ycb"):
    res = get_pointmap_correspondences(path_a, path_b, dataset_root,
                                       vmask_cache=vmask_cache, dataset_type=dataset_type)
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
            frame_npz_paths[0], frame_npz_paths[i], dataset_root, dataset_type=dataset_type)
        transforms.append(res if res else (1.0, np.eye(3), np.zeros(3)))
    return transforms


def strategy2_hierarchical(frame_npz_paths, dataset_root, dataset_type="dex-ycb"):
    n_frames = len(frame_npz_paths)
    groups = [[(i, (1.0, np.eye(3), np.zeros(3)))] for i in range(n_frames)]
    while len(groups) > 1:
        new_groups = []
        for i in range(0, len(groups) - 1, 2):
            g_a, g_b = groups[i], groups[i + 1]
            res = estimate_interframe_transform_pointmap(
                frame_npz_paths[g_a[0][0]], frame_npz_paths[g_b[0][0]],
                dataset_root, dataset_type=dataset_type)
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
            trace = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
            theta = np.arccos(trace)
            if theta < 1e-8:
                continue
            v = theta * np.array([R_rel[2, 1] - R_rel[1, 2], R_rel[0, 2] - R_rel[2, 0], R_rel[1, 0] - R_rel[0, 1]]) / (
                    2 * np.sin(theta))
            norm_v = np.linalg.norm(v)
            w_eff = w / max(norm_v, 1e-8)
            v_sum += w_eff * v
            w_sum += w_eff

        if w_sum == 0:
            break

        delta = v_sum / w_sum
        if np.linalg.norm(delta) < tol:
            break

        theta_d = np.linalg.norm(delta)
        if theta_d > 1e-8:
            n = delta / theta_d
            K = np.array([[0, -n[2], n[1]],
                          [n[2], 0, -n[0]],
                          [-n[1], n[0], 0]])
            R_delta = np.eye(3) + np.sin(theta_d) * K + (1 - np.cos(theta_d)) * (K @ K)
            R_mean = R_mean @ R_delta

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
    s_a, R_a, t_a = a
    s_b, R_b, t_b = b
    s_c = s_a * s_b
    R_c = R_a @ R_b
    t_c = s_a * (R_a @ t_b) + t_a
    return s_c, R_c, t_c


def strategy3_pgo(frame_npz_paths, dataset_root, num_iters=50, dataset_type="dex-ycb"):
    n_frames = len(frame_npz_paths)
    vmask_cache = precompute_vmasks(frame_npz_paths, dataset_root, dataset_type=dataset_type)

    print("    [PGO] Computing T(T-1)/2 pairwise edges...")
    edges = {}
    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            res = estimate_interframe_transform_pointmap(
                frame_npz_paths[i], frame_npz_paths[j], dataset_root,
                return_error=True, vmask_cache=vmask_cache, dataset_type=dataset_type)
            if res[0] is not None:
                (s, R, t), err = res
                weight = 1.0 / (err + 1e-6)
                edges[(i, j)] = ((s, R, t), weight)

    print(f"    [PGO] Found {len(edges)} valid edges. Initializing loops...")
    T_global = strategy1_reference(frame_npz_paths, dataset_root, dataset_type=dataset_type)
    T_global = [list(val) for val in T_global]

    print("    [PGO] Optimizing...")
    for it in range(num_iters):
        T_new = []
        for i in range(n_frames):
            if i == 0:
                T_new.append(T_global[0])
                continue

            votes_s, votes_R, votes_t, weights = [], [], [], []
            for j in range(n_frames):
                if i == j: continue
                s_j, R_j, t_j = T_global[j]

                if (i, j) in edges:
                    (s_ji, R_ji, t_ji), w = edges[(i, j)]
                    pred = compose_similarity_transform(
                        (s_j, R_j, t_j),
                        invert_similarity_transform(s_ji, R_ji, t_ji)
                    )
                    votes_s.append(pred[0]);
                    votes_R.append(pred[1])
                    votes_t.append(pred[2]);
                    weights.append(w)
                elif (j, i) in edges:
                    (s_ij, R_ij, t_ij), w = edges[(j, i)]
                    pred = compose_similarity_transform((s_j, R_j, t_j), (s_ij, R_ij, t_ij))
                    votes_s.append(pred[0]);
                    votes_R.append(pred[1])
                    votes_t.append(pred[2]);
                    weights.append(w)

            if len(weights) > 0:
                weights = np.array(weights)
                weights /= weights.sum()
                safe_scales = np.clip(np.array(votes_s), 1e-8, None)
                new_s = float(np.exp(np.sum(np.log(safe_scales) * weights)))
                new_t = np.sum(np.array(votes_t) * weights[:, None], axis=0)
                new_R = rotation_average(votes_R, weights)
                T_new.append([new_s, new_R, new_t])
            else:
                T_new.append(T_global[i])
        T_global = T_new

    return [(val[0], val[1], val[2]) for val in T_global]


def solve_final_gt_registration(frame_npz_paths, frame_transforms, dataset_root,
                                use_static_mask=True, dataset_type="dex-ycb"):
    vmask_cache = precompute_vmasks(frame_npz_paths, dataset_root, dataset_type=dataset_type)
    all_src, all_dst = [], []
    for i, path in enumerate(frame_npz_paths):
        res = extract_clean_gt_correspondences(
            np.load(path), dataset_root,
            precomputed_vmasks=vmask_cache.get(path),
            use_static_mask=use_static_mask,
            dataset_type=dataset_type
        )
        if res is None: continue
        src, dst = res
        s_i, R_i, tr_i = frame_transforms[i]
        src_aligned = apply_similarity_transform(src, s_i, R_i, tr_i)
        all_src.append(src_aligned)
        all_dst.append(dst)

    if not all_src: return 1.0, np.eye(3), np.zeros(3)
    s_glob, R_glob, tr_glob = estimate_similarity_transform(np.concatenate(all_src), np.concatenate(all_dst))

    pred = s_glob * (np.concatenate(all_src) @ R_glob.T) + tr_glob
    err = np.linalg.norm(pred - np.concatenate(all_dst), axis=-1).mean()
    print(f"  [4D-GT] Scale: {s_glob:.4f}  Residual Err: {err:.4f}  Corrs: {len(pred):,}")
    return s_glob, R_glob, tr_glob


def compute_4d_jitter_complete(frame_npz_paths, frame_transforms, s_glob, R_glob, tr_glob,
                               dataset_root, dataset_type="dex-ycb"):
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

        m = normalize_array(data['masks_2d'], V, H, W, is_mask=True)
        view_names, _ = get_view_names_and_masks(data, dataset_root, dataset_type=dataset_type)
        vms = build_gt_validity_masks(
            int(data['frame_idx']),
            [vn for vn in view_names if vn is not None],
            dataset_root, target_hw=(H, W), dataset_type=dataset_type
        )
        vi = 0
        for v in range(V):
            if view_names[v] is not None and vi < len(vms):
                if vms[vi] is not None:
                    m[v] &= vms[vi]
                vi += 1
        all_masks_mv.append(m)

    return compute_static_jitter(all_pm_mv, all_masks_mv)