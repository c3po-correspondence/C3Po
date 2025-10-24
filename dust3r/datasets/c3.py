import csv
import os
from os.path import join

import numpy as np
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import load_images
from dust3r.datasets.utils.transforms import *


class C3(BaseStereoViewDataset):
    def __init__(self, *args, data_dir, image_dir, augmentation_factor, **kwargs):
        self.data_dir = data_dir
        self.image_dir = image_dir
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data()
        self.augmentation_factor = augmentation_factor
        
    def _load_data(self):
        print("Loading image pairs...")
        with open(join(self.data_dir, f"{self.split}", "image_pairs.csv"), "r") as f:
            self.image_pairs = []            
            reader = csv.reader(f)
            next(reader)  # skip header
            
            for i, landmark, plan_name, photo_name in reader:
                self.image_pairs.append((i, landmark, plan_name, photo_name))
        print(f"{len(self.image_pairs)} image pairs loaded")

    def is_valid(self, plan_corrs, photo_corrs):
        return plan_corrs.shape[0] > 0 and photo_corrs.shape[0] > 0

    def __len__(self):
        return len(self.image_pairs) * self.augmentation_factor

    def __getitem__(self, idx):
        idx = idx[0] if self.split == "train" else idx
        i, landmark, plan_name, photo_name = self.image_pairs[idx % len(self.image_pairs)]
        plan_path = join(self.image_dir, landmark, plan_name)
        photo_path = join(self.image_dir,  landmark, photo_name)
        corrs_path = join(self.data_dir, f"{self.split}", "correspondences", f"{int(i) // 1000}", f"{int(i):06}.npy")
        corrs = np.load(corrs_path)
        size = self._resolutions[0][0]

        if idx < len(self.image_pairs):
            view1, view2 = load_images(
                [plan_path, photo_path], 
                size=size, 
                plan_corrs=corrs[0],
                photo_corrs=corrs[1], 
                augment=False,
                verbose=False
            )
            return view1, view2
        else:
            while True:  # keep trying until a valid pair is found
                view1, view2 = load_images(
                    [plan_path, photo_path], 
                    size=size, 
                    plan_corrs=corrs[0],
                    photo_corrs=corrs[1], 
                    augment=True,
                    verbose=False
                )
                if self.is_valid(view1["corrs"], view2["corrs"]):
                    return view1, view2
