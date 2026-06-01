import os
import torch


# ── Registry ──────────────────────────────────────────────────────────────────
DATASETS = {
    "dex-ycb": {
         "root": "/local/home/frrajic/xode/fabio/datasets/dex-ycb-multiview",
        "depth_max_m": 1.5,
        "subject_names": [
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
        ],
        "view_configs": {
            2: ["01", "06"],
            3: ["04", "06", "07"],
            4: None,
        },
        "default_target_views": ["02", "03", "06", "07"],
        "eye_up": [-0.04418, -0.6565, -0.7531],
    },
    "hi4d": {
        "root": "/home/fabio/datasets/hi4d",
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

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_BASE_ROOT = "/local/home/frrajic/xode/fabio/datasets/dex-ycb-multiview"

# ── Dataset Settings (Defaulting to Dex-YCB) ──────────────────────────────────
DATASET_TYPE = "dex-ycb"

def get_dataset_config(dataset_type=None):
    if dataset_type is None:
        dataset_type = DATASET_TYPE
    return DATASETS[dataset_type]

def get_subject_names(dataset_type=None):
    return get_dataset_config(dataset_type).get("subject_names", [])

def get_subject_by_code(dataset_type=None):
    names = get_subject_names(dataset_type)
    if not names:
        return {}
    # For Dex-YCB, codes are 01, 02...
    # For Hi4D, they are the full path e.g. pair00/dance00
    if dataset_type == "dex-ycb":
        return {name.split("subject-")[1][:2]: name for name in names}
    else:
        # For Hi4D, allow both the full path (pair00/dance00) and the basename (dance00) as codes
        mapping = {}
        for name in names:
            mapping[name] = name # Full path
            mapping[os.path.basename(name)] = name # Basename
            mapping[name.replace("/", "_")] = name # Underscore format (pair00_dance00)
        return mapping

# Backwards compatibility for old scripts (will default to Dex-YCB)
SUBJECT_NAMES = get_subject_names("dex-ycb")
SUBJECT_BY_CODE = get_subject_by_code("dex-ycb")

# ── Filtering ─────────────────────────────────────────────────────────────────
CONF_PERCENTILE = 0.5  # Filter to retain the top 70% of points based on confidence
DEPTH_MAX_M = 1.5
CLEAN_DEPTH = True

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
IMAGE_SIZE = 512
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

SCENE_GRAPH = "complete"
