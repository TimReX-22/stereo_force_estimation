import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from typing import Optional, List


class StereoDataset(Dataset):
    def __init__(self, left_dirs: List[str], right_dirs: List[str], transform: Optional[transforms.Compose] = None):
        self.left_dir = left_dirs
        self.right_dir = right_dirs
        self.transform = transform
        self.left_images = []
        self.right_images = []
        
        for left_dir, right_dir in zip(left_dirs, right_dirs):
            left_images = sorted([os.path.join(left_dir, img) for img in os.listdir(left_dir)])
            right_images = sorted([os.path.join(right_dir, img) for img in os.listdir(right_dir)])
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
