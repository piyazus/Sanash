
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import time
import os

# Import models
# (assuming paths setup or relative)
import sys
sys.path.append(os.path.join(os.getcwd(), 'models', 'csrnet'))
sys.path.append(os.path.join(os.getcwd(), 'models', 'p2pnet'))

from ultralytics import YOLO
from models.csrnet.model import CSRNet
from models.p2pnet.model import P2PNet

def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running {args.model} on {args.source} using {device}...")
    
    # Load Model
    model = None
    if args.model == 'yolov8':
        # Load best.pt if exists, else pretrained
        weights = 'models/yolov8/runs/detect/yolov8n_crowd/weights/best.pt'
        if not os.path.exists(weights):
            print(f"Warning: {weights} not found, using yolov8n.pt")
            weights = 'yolov8n.pt'
        model = YOLO(weights)
        
    elif args.model == 'csrnet':
        model = CSRNet().to(device)
        weights = 'models/csrnet/best_weights.pth'
        if os.path.exists(weights):
            model.load_state_dict(torch.load(weights, map_location=device))
        else:
            print("Warning: CSRNet weights not found, using random init")
        model.eval()
        
    elif args.model == 'p2pnet':
        model = P2PNet().to(device)
        weights = 'models/p2pnet/best_weights.pth'
        if os.path.exists(weights):
            model.load_state_dict(torch.load(weights, map_location=device))
        else:
            print("Warning: P2PNet weights not found, using random init")
        model.eval()
        
    # Open Source
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        # Try as image
        if os.path.exists(args.source):
            img = cv2.imread(args.source)
            if img is None:
                print("Error: Could not read source.")
                return
            # Process single image
            process_frame(img, model, args, device)
            return
        else:
            print("Error: Source not found.")
            return
            
    # Video Loop
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, 'output.mp4')
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        vis_frame, count = process_frame(frame, model, args, device)
        writer.write(vis_frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames. Count: {count:.1f}")
            
    end_time = time.time()
    print(f"Finished. Average FPS: {frame_count/(end_time-start_time):.2f}")
    cap.release()
    writer.release()

def process_frame(frame, model, args, device):
    # frame is BGR (OpenCV)
    vis_frame = frame.copy()
    count = 0
    
    if args.model == 'yolov8':
        results = model.predict(frame, verbose=False)
        count = len(results[0].boxes)
        vis_frame = results[0].plot()
        
    elif args.model == 'csrnet':
        # Preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            
        count = torch.sum(output).item()
        
        # Visualize Density Map (heatmap overlay)
        dmap = output[0,0].cpu().numpy()
        dmap = cv2.resize(dmap, (frame.shape[1], frame.shape[0]))
        dmap = (dmap - dmap.min()) / (dmap.max() - dmap.min() + 1e-5)
        dmap = np.uint8(255 * dmap)
        heatmap = cv2.applyColorMap(dmap, cv2.COLORMAP_JET)
        vis_frame = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        
        cv2.putText(vis_frame, f"Count: {count:.1f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    elif args.model == 'p2pnet':
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_points, pred_logits = model(input_tensor)
            prob = torch.sigmoid(pred_logits)
            mask = prob[0] > 0.5
            points = pred_points[0][mask.squeeze()].cpu().numpy()
            
        count = len(points)
        
        # Visualize points
        for p in points:
            cv2.circle(vis_frame, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
            
        cv2.putText(vis_frame, f"Count: {count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    return vis_frame, count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['yolov8', 'csrnet', 'p2pnet'])
    parser.add_argument('--source', type=str, required=True, help='Path to video or image')
    parser.add_argument('--output', type=str, default='inference_results', help='Output directory')
    args = parser.parse_args()
    
    run_inference(args)
