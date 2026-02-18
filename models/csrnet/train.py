
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from model import CSRNet
from dataset import CrowdDataset

def train_csrnet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparams
    lr = 1e-5
    epochs = 100
    batch_size = 1
    
    # Dataset
    root = '../../data/shanghaitech/part_B_final'
    train_set = CrowdDataset(os.path.join(root, 'train_data'))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    val_set = CrowdDataset(os.path.join(root, 'test_data'))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    model = CSRNet().to(device)
    criterion = nn.MSELoss(reduction="sum").to(device)
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
        
        for img, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            img = img.to(device)
            target = target.to(device)
            
            output = model(img)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {train_loss/len(train_loader):.4f}")
        
        # Validation
        model.eval()
        mae = 0.0
        mse = 0.0
        
        with torch.no_grad():
            for img, target in val_loader:
                img = img.to(device)
                target = target.to(device)
                output = model(img)
                pred_count = torch.sum(output).item()
                gt_count = torch.sum(target).item()
                mae += abs(pred_count - gt_count)
                mse += (pred_count - gt_count)**2
                
        mae = mae / len(val_loader)
        mse = (mse / len(val_loader))**0.5
        
        print(f"Val MAE: {mae:.2f} MSE: {mse:.2f}")
        
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
        f.write(f"Final MSE: {mse:.2f}\n")

if __name__ == '__main__':
    train_csrnet()
