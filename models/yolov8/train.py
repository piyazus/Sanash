
from ultralytics import YOLO
import os

def train_yolo():
    # Load model
    model = YOLO('yolov8n.pt')  # Nano model for speed
    
    # Train
    # Assuming script runs from models/yolov8/
    # data.yaml path needs to be correct relative to here or absolute
    
    data_path = os.path.abspath('data.yaml')
    
    print(f"Starting training with data config: {data_path}")

    # Check for existing run to resume
    # Expected layout from Ultralytics: runs/detect/<run_name>/weights/last.pt
    last_pt = 'runs/detect/yolov8n_crowd_overnight/weights/last.pt'
    if os.path.exists(last_pt):
        print(f"Resuming from {last_pt}...")
        model = YOLO(last_pt)
        results = model.train(resume=True)
    else:
        results = model.train(
            data=data_path, 
            epochs=100,        # Increased for overnight training
            imgsz=640, 
            batch=8, 
            project='runs/detect', 
            name='yolov8n_crowd_overnight',
            exist_ok=True
        )
    
    # Evaluate
    metrics = model.val()
    
    print("Training Complete.")
    print(f"mAP50-95: {metrics.box.map}")

if __name__ == '__main__':
    train_yolo()
