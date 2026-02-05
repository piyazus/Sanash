
import torch
import os
import random
from model import CSRNet
from dataset import CrowdDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CSRNet()
    model.load_state_dict(torch.load('best_weights.pth', map_location=device))
    model.to(device)
    model.eval()
    
    root = '../../data/shanghaitech/part_B_final'
    val_set = CrowdDataset(os.path.join(root, 'test_data'), train=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    
    mae = 0.0
    mse = 0.0
    
    # Visualization sample
    idx_to_vis = random.randint(0, len(val_set)-1)
    
    with torch.no_grad():
        for i, (img, target) in enumerate(val_loader):
            img = img.to(device)
            target = target.to(device)
            
            output = model(img)
            
            pred_count = torch.sum(output).item()
            gt_count = torch.sum(target).item()
            
            mae += abs(pred_count - gt_count)
            mse += (pred_count - gt_count)**2
            
            if i == idx_to_vis:
                # Save vis
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                # Show Original Image (need to inverse transform if normalized)
                # Here we just loaded it as tensor
                img_np = img[0].cpu().permute(1,2,0).numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                
                ax[0].imshow(img_np)
                ax[0].set_title(f"Input Image (GT: {gt_count:.1f})")
                
                ax[1].imshow(target[0,0].cpu().numpy(), cmap='jet')
                ax[1].set_title("GT Density Map")
                
                ax[2].imshow(output[0,0].cpu().numpy(), cmap='jet')
                ax[2].set_title(f"Pred Density Map (Count: {pred_count:.1f})")
                
                plt.savefig('density_map_examples.png')
                plt.close()
                
    mae = mae / len(val_loader)
    mse = (mse / len(val_loader))**0.5
    
    print(f"Evaluation Results - MAE: {mae:.2f} MSE: {mse:.2f}")
    with open('results.txt', 'a') as f:
        f.write(f"Eval MAE: {mae:.2f}\nEval MSE: {mse:.2f}\n")

if __name__ == '__main__':
    evaluate()
