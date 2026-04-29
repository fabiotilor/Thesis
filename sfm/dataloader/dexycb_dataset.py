import os
import cv2
import numpy as np
from .base_dataset import BaseDataset
from utils.geometry import closed_form_inverse_se3


class DexYCBDataset(BaseDataset):
    """
    Adapter for DexYCB datasets for the GGPT SfM pipeline.
    Directory structure expected:
      root/
        view_01/
          rgb/000000.jpg
          depth/000000.npy
          intrinsics_extrinsics.npz
        view_02/
          ...
    """

    def __init__(self, **kwargs):
        # We don't call super().__init__ because it triggers a discovery loop
        # that assumes a different directory structure.
        # We initialize the base attributes ourselves.
        self.root = kwargs.get('root')
        self.name = kwargs.get('name')
        self.img_size = kwargs.get('img_size')
        self.aspect_ratio = kwargs.get('aspect_ratio')
        self.load_depths = kwargs.get('load_depths', True)
        self.use_hash = kwargs.get('use_hash', True)
        self.sample_extracted = True
        self.sampled_list = []
        self.scene_pose_cache = {}

        # Override the sampled_list construction
        self.sampled_list = []
        # In DexYCB, "root" is the subject folder (e.g., .../20200709-subject-01__20200709_141754)
        # We need to find all valid frames across the requested views.
        # But this dataset class might be instantiated per subject.
        # Actually, for compute_ggpt_inputs.py, we can just pass the specific frame/views
        # Let's support passing `frame_idx` and `view_names` directly if possible, or discovering them.

        if 'view_names' in kwargs:
            self.view_names = kwargs['view_names']
        else:
            self.view_names = sorted([d for d in os.listdir(self.root) if d.startswith("view_")])

        if not self.view_names:
            return

        # Discover all frames from the first view
        first_view = self.view_names[0]
        rgb_dir = os.path.join(self.root, first_view, "rgb")
        if not os.path.exists(rgb_dir):
            return

        frame_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png'))])

        # Each item in sampled_list is (scene_name, img_names)
        # We'll treat each frame as a "scene" to process it independently if desired,
        # OR we can treat the whole sequence as one scene.
        # For our per-frame GGPT pipeline, treating each frame as a "scene" where
        # the "images" are the different views of that frame is correct.
        for frame_file in frame_files:
            # frame_file like '00000.jpg'
            # We encode the img_names to include the view name so we know which one to load
            img_names = [f"{v}/{frame_file}" for v in self.view_names]
            frame_idx = frame_file.split('.')[0]
            scene_name = f"frame_{frame_idx}"
            self.sampled_list.append([scene_name, img_names])

    def read_scene_pose(self, scene_name):
        if scene_name in self.scene_pose_cache:
            return self.scene_pose_cache[scene_name]
        # We need to return a dictionary: img_name -> pose dict
        # scene_name is 'frame_000000', img_names are 'view_01/00000.jpg'
        imgname2pose = {}
        for view_name in self.view_names:
            # DexYCB format
            ie_path = os.path.join(self.root, view_name, "intrinsics_extrinsics.npz")
            ie_data = np.load(ie_path)

            # Use keys from gt.py
            K = ie_data['intrinsics'].astype(np.float64)[:3, :3]
            # gt.py says cam2world = inv(data['extrinsics'])
            # which implies data['extrinsics'] is w2c.
            w2c = ie_data['extrinsics'].astype(np.float64)
            c2w = np.linalg.inv(w2c)

            # Find an image file to extract frame_file from
            rgb_dir = os.path.join(self.root, view_name, "rgb")

            # Determine image dimensions from the first available image
            img_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png'))])
            if not img_files:
                continue

            first_img_path = os.path.join(rgb_dir, img_files[0])
            temp_img = cv2.imread(first_img_path)
            height, width = temp_img.shape[:2]

            # In BaseDataset, img_names are iterated over. Let's just construct the key flexibly.
            # We'll just put it in the dict for any file in that view.
            for frame_file in img_files:
                img_name = f"{view_name}/{frame_file}"

                imgname2pose[img_name] = {
                    'c2w': c2w,
                    'w2c': w2c,
                    'K': K,
                    'width': width,
                    'height': height,
                    'intr_convention': 'opencv'
                }
        self.scene_pose_cache[scene_name] = imgname2pose
        return imgname2pose

    def read_img_pose(self, scene_name, img_name):
        return self.read_scene_pose(scene_name)[img_name]

    def read_img_rgb(self, scene_name, img_name):
        # img_name is like 'view_01/00000.jpg'
        img_path = os.path.join(self.root, img_name.replace("rgb/", "").replace("depth/", ""))
        if "rgb" not in img_name:
            # Need to insert 'rgb' into path: view_01/rgb/00000.jpg
            parts = img_name.split("/")
            img_path = os.path.join(self.root, parts[0], "rgb", parts[1])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def read_img_depth(self, scene_name, img_name):
        # img_name is like 'view_01/00000.jpg'
        parts = img_name.split("/")
        depth_name = parts[1].replace('.jpg', '.png').replace('.png', '.png')
        depth_path = os.path.join(self.root, parts[0], "depth", depth_name)

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(f"Depth not found at {depth_path}")
        depth = depth_raw.astype(np.float32) / 1000.0  # Convert mm to meters
        return depth
