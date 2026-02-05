
import os
import glob
import scipy.io
import cv2
import numpy as np
from tqdm import tqdm

def convert_to_yolo(part_path, output_path, box_size=50):
    os.makedirs(output_path, exist_ok=True)
    images_dst = os.path.join(output_path, 'images')
    labels_dst = os.path.join(output_path, 'labels')
    os.makedirs(images_dst, exist_ok=True)
    os.makedirs(labels_dst, exist_ok=True)
    
    # Process Train and Test
    for split in ['train_data', 'test_data']:
        img_src_dir = os.path.join(part_path, split, 'images')
        gt_src_dir = os.path.join(part_path, split, 'ground_truth')
        
        # In standardized structure:
        # data/shanghaitech/part_A_final/train_data/images
        
        img_files = glob.glob(os.path.join(img_src_dir, '*.jpg'))
        
        split_name = 'train' if 'train' in split else 'val'
        # Simplified: put all in one bucket OR keep splits? 
        # YOLO structure: dataset/train/images, dataset/train/labels
        
        split_img_dir = os.path.join(images_dst, split_name)
        split_lbl_dir = os.path.join(labels_dst, split_name)
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)
        
        for img_path in tqdm(img_files, desc=f"Converting {split}"):
            # 1. Copy/Process Image
            fname = os.path.basename(img_path)
            basename = os.path.splitext(fname)[0]
            
            # Read image to get dims
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            
            # Copy image to dest (symlink might be better but copy is safer for formatting)
            out_img_path = os.path.join(split_img_dir, fname)
            cv2.imwrite(out_img_path, img)
            
            # 2. Process Label
            gt_name = f"GT_{basename}.mat"
            gt_path = os.path.join(gt_src_dir, gt_name)
            
            txt_content = []
            if os.path.exists(gt_path):
                try:
                    mat = scipy.io.loadmat(gt_path)
                    if 'image_info' in mat:
                        points = mat['image_info'][0,0]['location'][0,0]
                    elif 'location' in mat:
                        points = mat['location']
                    else:
                        points = []
                        
                    for p in points:
                        x, y = p[0], p[1]
                        
                        # Create pseudo bbox
                        # x_center, y_center, w, h (normalized)
                        # Box size is fixed pixels, e.g., 50x50
                        
                        # Normalize
                        xn = x / w
                        yn = y / h
                        wn = box_size / w
                        hn = box_size / h
                        
                        # Clip
                        if xn < 0 or xn > 1 or yn < 0 or yn > 1:
                            continue
                            
                        # YOLO class 0 is 'person'
                        txt_content.append(f"0 {xn:.6f} {yn:.6f} {wn:.6f} {hn:.6f}")
                except Exception as e:
                    print(f"Error {gt_path}: {e}")
            
            with open(os.path.join(split_lbl_dir, f"{basename}.txt"), 'w') as f:
                f.write('\n'.join(txt_content))

# Convert Part B (better for YOLO as less dense)
# We will use Part B for YOLO demonstration to get better results
convert_to_yolo('data/shanghaitech/part_B_final', 'data/yolo_dataset', box_size=40)
