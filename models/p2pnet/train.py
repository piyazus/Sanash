
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np
from torchvision import transforms

from model import P2PNet
from dataset import P2PDataset
from loss import P2PLoss

def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets

def train_p2pnet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparams
    lr = 1e-4
    epochs = 100
    batch_size = 2 
    
    # Dataset (Part B)
    root = '../../data/shanghaitech/part_B_final'
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_set = P2PDataset(os.path.join(root, 'train_data'), transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    val_set = P2PDataset(os.path.join(root, 'test_data'), transform=transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = P2PNet().to(device)
    criterion = P2PLoss(w_cls=1.0, w_reg=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    start_epoch = 0
    best_mae = 1e9
    
    # Resume logic
    checkpoint_path = 'checkpoint.pth'
    if os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_mae = checkpoint.get('best_mae', 1e9)
        print(f"Resuming from epoch {start_epoch}")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        
        for img, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            img = img.to(device)
            pred_points, pred_logits = model(img)
            loss = criterion(pred_points, pred_logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {train_loss/len(train_loader):.4f}")
        
        # Validation
        model.eval()
        mae = 0.0
        with torch.no_grad():
            for img, targets in val_loader:
                img = img.to(device)
                pred_points, pred_logits = model(img)
                prob = torch.sigmoid(pred_logits)
                for b in range(len(targets)):
                    mask = prob[b] > 0.5
                    pred_count = mask.sum().item()
                    gt_count = len(targets[b]['points'])
                    mae += abs(pred_count - gt_count)
                    
        mae = mae / len(val_set)
        print(f"Val MAE: {mae:.2f}")
        
        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_mae': best_mae
        }, checkpoint_path)
        
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), 'best_weights.pth')
            print(f"Saved best model (MAE: {mae:.2f})")
            
    with open('results.txt', 'w') as f:
        f.write(f"Best MAE: {best_mae:.2f}\n")

if __name__ == '__main__':
    train_p2pnet()
