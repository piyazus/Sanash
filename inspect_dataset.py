
import os
import glob
import scipy.io
import numpy as np

def analyze_part(part_path, part_name):
    train_img = glob.glob(os.path.join(part_path, 'train_data', 'images', '*.jpg'))
    test_img = glob.glob(os.path.join(part_path, 'test_data', 'images', '*.jpg'))
    
    print(f"--- {part_name} ---")
    print(f"Train images: {len(train_img)}")
    print(f"Test images: {len(test_img)}")
    
    train_gt = glob.glob(os.path.join(part_path, 'train_data', 'ground_truth', '*.mat'))
    
    total_count = 0
    img_count = 0
    
    for gt_path in train_gt[:100]:
        try:
            mat = scipy.io.loadmat(gt_path)
            if 'image_info' in mat:
                info = mat['image_info']
                points = info[0,0]['location'][0,0]
                count = len(points)
            elif 'location' in mat:
                 count = len(mat['location'])
            else:
                 count = 0 
            
            total_count += count
            img_count += 1
        except Exception as e:
            print(f"Error: {e}")
            
    avg = total_count/img_count if img_count else 0
    print(f"Sampled {img_count} images. Average Density: {avg:.2f}")
    return avg

analyze_part('data/shanghaitech/part_A_final', 'Part A')
analyze_part('data/shanghaitech/part_B_final', 'Part B')
