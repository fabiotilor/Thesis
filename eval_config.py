import os
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_BASE_ROOT = "/home/fabio/datasets/dex-ycb-multiview" #/local/home/frrajic/xode/fabio/datasets/dex-ycb-multiview

SUBJECT_NAMES = [
    "20200709-subject-01__20200709_141754",
    "20200813-subject-02__20200813_145653",
    "20200820-subject-03__20200820_135841",
    "20200903-subject-04__20200903_104428",
    "20200908-subject-05__20200908_144409",
    "20200918-subject-06__20200918_114117",
    "20200928-subject-07__20200928_144906",
    "20201002-subject-08__20201002_110227",
    "20201015-subject-09__20201015_144721",
    "20201022-subject-10__20201022_112651",
]

SUBJECT_BY_CODE = {name.split("subject-")[1][:2]: name for name in SUBJECT_NAMES}

# ── Filtering ─────────────────────────────────────────────────────────────────
CONF_PERCENTILE = 0.5  # Retain top 50% of points based on confidence
DEPTH_MAX_M = 1.5

# ── Model ─────────────────────────────────────────────────────────────────────
VGGT4D_CHECKPOINT = "./ckpts/model_tracker_fixed_e20.pt"
IMAGE_SIZE = 518  # VGGT native target size (divisible by patch_size=14)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Evaluation / Strategy ─────────────────────────────────────────────────────
VIEW_CONFIGS = {
    2: ["01", "06"],
    3: ["04", "06", "07"],
    4: None,
}

DEFAULT_TARGET_VIEWS = ["02", "03", "06", "07"]

# ── Visualization ─────────────────────────────────────────────────────────────
RERUN_ADDR = "rerun+http://127.0.0.1:9876/proxy"
RERUN_EYE_UP = [-0.04418, -0.6565, -0.7531]
