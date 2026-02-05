
import torch
import os
import random
from model import P2PNet
from dataset import P2PDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = P2PNet()
    model.load_state_dict(torch.load('best_weights.pth', map_location=device))
    model.to(device)
    model.eval()
    
    root = '../../data/shanghaitech/part_B_final'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_set = P2PDataset(os.path.join(root, 'test_data'), transform=transform)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    mae = 0.0
    
    # Visual sample
    idx_vis = random.randint(0, len(val_set)-1)
    
    with torch.no_grad():
        for i, (img, targets) in enumerate(val_loader):
            img = img.to(device)
            # targets is list of dicts
            
            pred_points_b, pred_logits_b = model(img) # B x N x 2, B x N x 1
            prob = torch.sigmoid(pred_logits_b)
            
            for b in range(len(targets)):
                gt_points = targets[b]['points']
                gt_count = len(gt_points)
                
                # Threshold
                mask = prob[b] > 0.5
                pred_pts_filtered = pred_points_b[b][mask.squeeze()]
                pred_count = len(pred_pts_filtered)
                
                mae += abs(pred_count - gt_count)
                
                if i == idx_vis:
                    # Viz
                    # Denormalize image
                    img_np = img[b].cpu().permute(1,2,0).numpy()
                    img_np = (img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
                    img_np = (img_np * 255).astype('uint8')
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # OpenCV uses BGR
                    
                    # Draw GT
                    for p in gt_points:
                        cv2.circle(img_np, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1) # Green = GT
                        
                    # Draw Pred
                    for p in pred_pts_filtered:
                        cv2.circle(img_np, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1) # Red = Pred
                        
                    cv2.putText(img_np, f"GT: {gt_count} Pred: {pred_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    cv2.imwrite('prediction_visualizations.png', img_np)
                    
    mae = mae / len(val_set)
    print(f"Eval MAE: {mae:.2f}")
    with open('results.txt', 'a') as f:
        f.write(f"Final MAE: {mae:.2f}\n")

if __name__ == '__main__':
    evaluate()
