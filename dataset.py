import torch
from PIL import Image
import os
import config
from torch.utils.data import Dataset
import numpy as np

class AgeDataset(Dataset):
    def __init__(self, root_young, root_old, transform = None):
        super().__init__()
        self.root_yong = root_young
        self.root_old = root_old
        self.transform = transform

        self.young_images = os.listdir(root_young)
        self.old_images = os.listdir(root_old)
        self.length_dataset = max(len(self.young_images), len(self.old_images)) # 1000, 1500
        
        self.young_len = len(self.young_images)
        self.old_len = len(self.old_images)
    
    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        young_img = self.young_images[index % self.young_len]
        old_img = self.old_images[index % self.old_len]

        young_path = os.path.join(self.root_yong, young_img)
        old_path = os.path.join(self.root_old, old_img)

        young_img = np.array(Image.open(young_path).convert("RGB"))
        old_img = np.array(Image.open(old_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=young_img, image0=old_img)
            young_img = augmentations['image']
            old_img = augmentations['image0']
        
        return young_img, old_img