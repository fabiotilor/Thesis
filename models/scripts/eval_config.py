import os
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
if os.path.exists("/local/home/frrajic/xode/fabio/datasets"):
    DATASET_BASE_ROOT = "/local/home/frrajic/xode/fabio/datasets/dex-ycb-multiview"
    if os.path.exists("/local/home/frrajic/xode/fabio/datasets/hi4d/Bachelorarbeit/hi4d"):
        HI4D_BASE_ROOT = "/local/home/frrajic/xode/fabio/datasets/hi4d/Bachelorarbeit/hi4d"
    else:
        HI4D_BASE_ROOT = "/local/home/frrajic/xode/fabio/datasets/hi4d"
else:
    DATASET_BASE_ROOT = "/home/fabio/datasets/dex-ycb-multiview"
    HI4D_BASE_ROOT = "/home/fabio/datasets/hi4d"

if os.path.exists("/local/home/frrajic/xode/fabio/monofusion"):
    MONOFUSION_BASE_ROOT = "/local/home/frrajic/xode/fabio/monofusion"
else:
    MONOFUSION_BASE_ROOT = "/home/fabio/monofusion"

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
CONF_PERCENTILE = 1.0  # Filter to retain the top 50% of points based on confidence
DEPTH_MAX_M = 1.5

# ── Multi-Model / GGPT Support ────────────────────────────────────────────────
SUPPORTED_MODELS = ["vggt-point", "pi3", "pi3x", "mast3r", "vggt4d", "monofusion"]
GGPT_INPUTS_ROOT = "ggpt_inputs"  # where precomputed GGPT inputs will be saved
GGPT_CKPT = "ckpts/model.step228000.pth"

# ── Model ─────────────────────────────────────────────────────────────────────
# Note: VGGT native target size is 518. DAv3 uses 504.
IMAGE_SIZE = 518
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Evaluation / Strategy ─────────────────────────────────────────────────────
DEFAULT_TARGET_VIEWS = ["02", "03", "06", "07"]

VIEW_CONFIGS = {
    2: ["01", "06"],
    3: ["04", "06", "07"],
    4: DEFAULT_TARGET_VIEWS, # Updated to enable 4-view generation
}

# ── Visualization ─────────────────────────────────────────────────────────────
RERUN_ADDR = "rerun+http://127.0.0.1:9876/proxy"
RERUN_EYE_UP = [-0.04418, -0.6565, -0.7531]

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
    },
    "monofusion": {
        "root": MONOFUSION_BASE_ROOT,
        "depth_max_m": None,
        "subject_names": [f"subject-{i:02d}" for i in range(1, 11)],
        "view_configs": {
            2: ["view_00", "view_01"],
            3: ["view_00", "view_01", "view_02"],
            4: ["view_00", "view_01", "view_02", "view_03"],
        },
        "default_target_views": ["view_00", "view_01", "view_02", "view_03"],
        "eye_up": [0, 1, 0],
    }
}

# ── Helper functions ────────────────────────────────────────────────────────

def get_dataset_config(dataset_name):
    """Get configuration for a specific dataset."""
    return DATASETS.get(dataset_name, DATASETS["dex-ycb"])


def get_subject_by_code(dataset_name):
    """Get subject code mapping for a specific dataset.
    Returns dict mapping code -> folder name used in aligned_outputs.
    """
    config = get_dataset_config(dataset_name)
    names = config["subject_names"]
    if dataset_name == "hi4d":
        # code "dance00" -> folder "subject-dance00"
        mapping = {}
        for name in names:
            action = name.split("/")[-1]  # e.g., "dance00"
            mapping[action] = f"subject-{action}"
        return mapping
    elif dataset_name == "monofusion":
        return {name.replace("subject-", ""): name for name in names}
    else:
        return {name.split("subject-")[1][:2]: name for name in names}


def get_dataset_root_for_subject(dataset_name, subject_full):
    """Get the actual dataset root path for a subject.
    For dex-ycb: DATASET_BASE_ROOT/subject_full
    For hi4d:    HI4D_BASE_ROOT/pair00/dance00 (resolved from subject_full)
    """
    config = get_dataset_config(dataset_name)
    if dataset_name == "hi4d":
        # subject_full = "subject-dance00" or "subject-pair00_dance00" -> need "pair00/dance00"
        action = subject_full.replace("subject-", "")
        if "_" in action and action.startswith("pair"):
            action_norm = "/".join(action.split("_", 1))
        else:
            action_norm = action
        for name in config["subject_names"]:
            if name.endswith(action_norm) or name.endswith(action):
                return os.path.join(config["root"], name)
        # Fallback
        return os.path.join(config["root"], action_norm)
    elif dataset_name == "monofusion":
        return os.path.join(config["root"], "ggpt_inputs", subject_full)
    else:
        return os.path.join(config["root"], subject_full)


def get_view_config(dataset_name, nviews, pair_name=None):
    """Get view configuration for dataset and view count."""
    config = get_dataset_config(dataset_name)
    view_configs = config.get("view_configs", {})

    if dataset_name == "hi4d":
        if pair_name and pair_name in view_configs:
            pair_config = view_configs[pair_name]
            return pair_config.get(nviews, pair_config.get(4))
        return view_configs.get("default", {}).get(nviews, ["4", "16", "28", "40"])
    elif dataset_name == "monofusion":
        return view_configs.get(nviews, [f"view_{i:02d}" for i in range(nviews)])
    else:
        result = view_configs.get(nviews, config.get("default_target_views"))
        if result is None:
            result = config.get("default_target_views")
        return result


def get_pair_name_for_subject(dataset_name, subject_full):
    """Extract pair name (e.g. 'pair00') for Hi4D subjects."""
    if dataset_name != "hi4d":
        return None
    config = get_dataset_config(dataset_name)
    action = subject_full.replace("subject-", "")
    if "_" in action and action.startswith("pair"):
        action_norm = "/".join(action.split("_", 1))
    else:
        action_norm = action
    for name in config["subject_names"]:
        if name.endswith(action_norm) or name.endswith(action):
            return name.split("/")[0]  # e.g., "pair00"
    return None
