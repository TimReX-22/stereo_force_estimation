import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
import joblib

from typing import Optional, List

import pandas as pd


class StereoForceDataset(Dataset):
    def __init__(self, force_files: List[str], transform: Optional[transforms.Compose] = None):
        self.left_images = []
        self.right_images = []
        self.forces = []
        self.transform = transform

        for force_file in force_files:
            force_df = pd.read_excel(force_file)
            left_images = force_df['ZED Camera Left'].tolist()
            right_images = force_df['ZED Camera Right'].tolist()
            force_values = force_df[['Force_x_smooth',
                                     'Force_y_smooth', 'Force_z_smooth']].values

            for left_image, right_image, force in zip(left_images, right_images, force_values):
                left_image_path = os.path.join(
                    "data", "/".join(left_image.split("/")[9:]))
                right_image_path = os.path.join(
                    "data", "/".join(right_image.split("/")[9:]))
                if os.path.exists(left_image_path) and os.path.exists(right_image_path):
                    self.left_images.append(left_image_path)
                    self.right_images.append(right_image_path)
                    self.forces.append(force)

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.forces = self.scaler.fit_transform(self.forces)
        joblib.dump(self.scaler, 'transformations/target_scaler.pkl')

        assert len(self.left_images) == len(self.right_images) == len(
            self.forces), f"{len(self.left_images)=}, {len(self.forces)=}, {len(self.right_images)=}"

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx: int):
        left_image = Image.open(self.left_images[idx]).convert('RGB')
        right_image = Image.open(self.right_images[idx]).convert('RGB')
        force = torch.tensor(self.forces[idx], dtype=torch.float32)

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        return left_image, right_image, force


class StereoDataset(Dataset):
    def __init__(self, left_dirs: List[str], right_dirs: List[str], transform: Optional[transforms.Compose] = None):
        self.left_dir = left_dirs
        self.right_dir = right_dirs
        self.transform = transform
        self.left_images = []
        self.right_images = []

        for left_dir, right_dir in zip(left_dirs, right_dirs):
            left_images = sorted([os.path.join(left_dir, img)
                                 for img in os.listdir(left_dir)])
            right_images = sorted([os.path.join(right_dir, img)
                                  for img in os.listdir(right_dir)])
            self.left_images.extend(left_images)
            self.right_images.extend(right_images)

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx: int):
        left_image = Image.open(self.left_images[idx]).convert('RGB')
        right_image = Image.open(self.right_images[idx]).convert('RGB')

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        return left_image, right_image
