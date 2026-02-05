
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import scipy.spatial
from scipy.ndimage import gaussian_filter

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    
    # Adaptive kernel for crowded scenes (Part A), Fixed for sparse (Part B)
    # Heuristic: if > 100 points, assume crowded and use adaptive
    # Otherwise fixed often works well or adaptive is fast enough.
    
    if gt_count > 0:
        leafsize = 2048
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
        distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. 
        
        # Optimization: Apply filter only on a patch
        sigma = min(sigma, 100) # clip
        kernel_size = int(round(sigma * 4)) * 2 + 1
        
        # If kernel is huge, falling back to full image filter might be safer but slower.
        # But usually sigma is small (e.g. 5-10).
        
        x, y = int(pt[0]), int(pt[1])
        
        # Generate gaussian kernel
        k_size = kernel_size
        k_half = k_size // 2
        
        # Create grid for kernel
        kx = np.arange(k_size) - k_half
        ky = np.arange(k_size) - k_half
        kxx, kyy = np.meshgrid(kx, ky)
        kernel = np.exp(-(kxx**2 + kyy**2) / (2 * sigma**2))
        kernel = kernel / (2 * np.pi * sigma**2) # Normalize? Density sum should be 1 per point ideally?
        # Actually CSRNet usually strictly sums to 1 per point.
        # Normalize kernel so sum is 1
        kernel = kernel / np.sum(kernel)
        
        # Place kernel on density map
        r_min = max(0, y - k_half)
        r_max = min(gt.shape[0], y + k_half + 1)
        c_min = max(0, x - k_half)
        c_max = min(gt.shape[1], x + k_half + 1)
        
        # Indices in kernel
        kr_min = max(0, k_half - (y - 0))
        kr_max = k_size - max(0, (y + k_half + 1) - gt.shape[0])
        kc_min = max(0, k_half - (x - 0))
        kc_max = k_size - max(0, (x + k_half + 1) - gt.shape[1])
        
        # density[r_min:r_max, c_min:c_max] += kernel[kr_min:kr_max, kc_min:kc_max]
        # Careful with shapes
        
        patch = kernel[kr_min:kr_min + (r_max-r_min), kc_min:kc_min + (c_max-c_min)]
        density[r_min:r_max, c_min:c_max] += patch
        
    return density

def save_density_map(img_path, gt_path, output_path):
    img = Image.open(img_path).convert('RGB')
    mat = io.loadmat(gt_path)
    
    if 'image_info' in mat:
        points = mat['image_info'][0,0]['location'][0,0]
    elif 'location' in mat:
        points = mat['location']
    else:
        points = []
        
    k = np.zeros((img.size[1], img.size[0]))
    for p in points:
        x, y = int(p[0]), int(p[1])
        if y < img.size[1] and x < img.size[0]:
             k[y,x]=1
             
    dmap = gaussian_filter_density(k)
    
    with h5py.File(output_path, 'w') as hf:
            hf['density'] = dmap
            
    return dmap
