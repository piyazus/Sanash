
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import h5py
import cv2

class CrowdDataset(Dataset):
    def __init__(self, root, transform=None, train=True, method='train'):
        self.root = root
        self.transform = transform
        self.train = train
        
        # Expect root to be like 'data/shanghaitech/part_A_final/train_data'
        self.img_dir = os.path.join(root, 'images')
        self.gt_dir = os.path.join(root, 'ground_truth_h5') # Pre-generated h5
        
        self.img_list = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        self.nSamples = len(self.img_list)
        
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        fname = self.img_list[index]
        img_path = os.path.join(self.img_dir, fname)
        gt_path = os.path.join(self.gt_dir, fname.replace('.jpg','.h5'))
        
        img = Image.open(img_path).convert('RGB')
        
        if self.train:
            # Data augmentation?
            pass

        key = 'density'
        with h5py.File(gt_path, 'r') as hf:
            target = np.asarray(hf[key])
            
        target = cv2.resize(target,(target.shape[1]//8,target.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64
        
        if self.transform:
            img = self.transform(img)
            
        target = torch.from_numpy(target).float().unsqueeze(0)
        
        return img, target
