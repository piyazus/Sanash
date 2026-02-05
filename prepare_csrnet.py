
import os
import glob
from tqdm import tqdm
import sys
import argparse

# Add current directory to path to import utils
sys.path.append(os.path.join(os.getcwd(), 'models', 'csrnet'))
from utils import save_density_map

def generate_maps(root):
    # Process train_data and test_data
    for split in ['train_data', 'test_data']:
        img_dir = os.path.join(root, split, 'images')
        gt_dir = os.path.join(root, split, 'ground_truth')
        h5_dir = os.path.join(root, split, 'ground_truth_h5')
        
        os.makedirs(h5_dir, exist_ok=True)
        
        img_files = glob.glob(os.path.join(img_dir, '*.jpg'))
        
        print(f"Generating density maps for {split} in {root}...")
        for img_path in tqdm(img_files):
            fname = os.path.basename(img_path)
            basename = os.path.splitext(fname)[0]
            gt_path = os.path.join(gt_dir, f"GT_{basename}.mat")
            out_path = os.path.join(h5_dir, f"{fname.replace('.jpg','.h5')}")
            
            try:
                save_density_map(img_path, gt_path, out_path)
            except Exception as e:
                print(f"Failed {fname}: {e}")

if __name__ == "__main__":
    # We will just run for Part B first as it is faster and cleaner for demo
    generate_maps('data/shanghaitech/part_B_final')
    # Uncomment to run for Part A (slower)
    # generate_maps('data/shanghaitech/part_A_final')
