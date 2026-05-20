import os
import torch

# ── Global Configuration ──────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
VGGT4D_CHECKPOINT = "ckpts/model_tracker_fixed_e20.pt"
CONF_PERCENTILE = 1.0 # Retain top 50% of points based on confidence for dex-ycb. 100% for hi4d
DEPTH_MAX_M = 1.5

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_BASE_ROOT = "/local/home/frrajic/xode/fabio/datasets/dex-ycb-multiview"
HI4D_BASE_ROOT = "/local/home/frrajic/xode/fabio/datasets/hi4d/Bachelorarbeit/hi4d/"

# DexYCB Subjects
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

# Hi4D Pairs and Sequences
HI4D_PAIRS = [f"pair{i:02d}" for i in range(20)]  # pair00 to pair19
HI4D_SEQUENCES = ["talk", "hug", "dance", "fight", "sports", "wizard", "ghost", "shakehands", "song"]

# Hi4D frame indexing: start at frame 22, step size 1, total 24 frames
HI4D_START_FRAME = 22
HI4D_STEP_SIZE = 1
HI4D_TOTAL_FRAMES = 24

# ── Visualization ─────────────────────────────────────────────────────────────
RERUN_ADDR = "rerun+http://127.0.0.1:9876/proxy"
RERUN_EYE_UP = [-0.04418, -0.6565, -0.7531]
SCENE_GRAPH = "complete"

# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS = {
    "dex-ycb": {
        "root": DATASET_BASE_ROOT,
        "depth_max_m": 1.5,
        "subject_names": SUBJECT_NAMES,
        "view_configs": {
            2: ["01", "06"],
            3: ["04", "06", "07"],
            4: None,
        },
        "default_target_views": ["02", "03", "06", "07"],
        "eye_up": [-0.04418, -0.6565, -0.7531],
    },
    "hi4d": {
        "root": HI4D_BASE_ROOT,
        "depth_max_m": None,  # Hi4D has no depth maps; filtering disabled
        "subject_names": [
            "pair00/dance00",
            "pair00/fight00",
            "pair00/highfive00",
            "pair00/taichi00",
            "pair00/hug00",
            "pair00/yoga00",
            "pair01/basketball01",
            "pair01/talk01",
            "pair01/fight01",
            "pair01/highfive01",
            "pair01/hug01",
            "pair09/talk09",
            "pair09/highfive09",
            "pair09/bend09",
            "pair09/hug09",
        ],
        "view_configs": {
            "default": {
                2: ["4", "16"],
                4: ["4", "16", "28", "40"],
                8: ["4", "16", "28", "40", "52", "64", "76", "88"],
            },
            "pair00": {
                2: ["16", "4"],
                3: ["16", "4", "88"],
                4: ["16", "4", "88", "28"],
            },
            "pair01": {
                2: ["16", "4"],
                3: ["16", "4", "88"],
                4: ["16", "4", "88", "28"],
            },
            "pair09": {
                2: ["28", "40"],
                3: ["28", "40", "16"],
                4: ["28", "40", "16", "52"],
            }
        },
        "default_target_views": ["4", "16", "28", "40", "52", "64", "76", "88"],
        "eye_up": [0, 1, 0], # Placeholder, adjust as needed for Hi4D
    }
}

# ── Helper functions ────────────────────────────────────────────────────────
def get_dataset_config(dataset_name):
    """Get configuration for a specific dataset."""
    return DATASETS.get(dataset_name, DATASETS["dex-ycb"])

def get_subject_by_code(dataset_name):
    """Get subject code mapping for a specific dataset."""
    config = get_dataset_config(dataset_name)
    names = config["subject_names"]
    if dataset_name == "hi4d":
        # For Hi4D: pair00/dance00 -> pair00/dance00 (use as-is)
        return {name: name for name in names}
    else:
        # For DexYCB: extract subject code like "01"
        return {name.split("subject-")[1][:2]: name for name in names}

def get_view_config(dataset_name, nviews, pair_name=None):
    """Get view configuration for dataset and view count."""
    config = get_dataset_config(dataset_name)
    view_configs = config.get("view_configs", {})

    if dataset_name == "hi4d":
        # For Hi4D, check pair-specific config first
        if pair_name and pair_name in view_configs:
            pair_config = view_configs[pair_name]
            return pair_config.get(nviews, pair_config.get(4))  # default to 4 views
        return view_configs.get("default", {}).get(nviews, ["4", "16", "28", "40"])
    else:
        # For DexYCB
        return view_configs.get(nviews, config.get("default_target_views"))

# ── Backward compatibility ───────────────────────────────────────────────────
# Default to dex-ycb for now if not specified.
_DEFAULT_DATASET = "dex-ycb"
DATASET_BASE_ROOT = DATASETS[_DEFAULT_DATASET]["root"]
SUBJECT_NAMES = DATASETS[_DEFAULT_DATASET]["subject_names"]
SUBJECT_BY_CODE = {name.split("subject-")[1][:2]: name for name in SUBJECT_NAMES}
VIEW_CONFIGS = DATASETS[_DEFAULT_DATASET]["view_configs"]
DEFAULT_TARGET_VIEWS = DATASETS[_DEFAULT_DATASET]["default_target_views"]
MODEL_NAME = "facebook/VGGT-1B"
IMAGE_SIZE = 518  # VGGT native target size (divisible by patch_size=14)