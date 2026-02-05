
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import scipy.io as io
import cv2

class P2PDataset(Dataset):
    def __init__(self, root, transform=None, train=True):
        self.root = root
        self.transform = transform
        self.train = train
        self.img_dir = os.path.join(root, 'images')
        self.gt_dir = os.path.join(root, 'ground_truth')
        self.img_list = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        fname = self.img_list[index]
        img_path = os.path.join(self.img_dir, fname)
        gt_path = os.path.join(self.gt_dir, 'GT_' + fname.replace('.jpg','.mat').replace('IMG_','IMG_')) 
        # Note: ShanghaiTech naming convention varies slightly, helper usually handles it.
        # Here assuming standard names.
        
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        points = []
        if os.path.exists(gt_path):
            mat = io.loadmat(gt_path)
            if 'image_info' in mat:
                points = mat['image_info'][0,0]['location'][0,0]
            elif 'location' in mat:
                points = mat['location']
                
        # Points are x,y
        target = {'points': torch.tensor(points, dtype=torch.float32), 'image_id': index}

        if self.transform:
            img = self.transform(img)
            
        return img, target
