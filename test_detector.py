"""
Test script for the Bus Vision Person Detector
Downloads a sample video and runs detection
"""

import cv2
import urllib.request
import os
import sys

def download_sample_video():
    """Download a sample video with people for testing."""
    output_path = "input/sample_pedestrians.mp4"
    
    # Create input directory if needed
    os.makedirs("input", exist_ok=True)
    
    if os.path.exists(output_path):
        print(f"Sample video already exists: {output_path}")
        return output_path
    
    # Sample video URL (pedestrian walking video from Pexels - free to use)
    # This is a short clip perfect for testing person detection
    video_url = "https://www.pexels.com/download/video/3209828/?fps=25.0&h=1080&w=1920"
    
    print(f"Downloading sample video...")
    print("This may take a moment...")
    
    try:
        # Create a request with headers to avoid 403
        request = urllib.request.Request(
            video_url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        
        with urllib.request.urlopen(request, timeout=60) as response:
            with open(output_path, 'wb') as out_file:
                out_file.write(response.read())
        
        print(f"Downloaded: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Could not download sample video: {e}")
        print("\nAlternative: Creating a test video using webcam...")
        return create_webcam_test_video()

def create_webcam_test_video():
    """Create a short test video from webcam if download fails."""
    output_path = "input/webcam_test.mp4"
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam available and download failed.")
        print("\nPlease manually place a video file in the 'input' folder.")
        return None
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Recording 5 seconds from webcam...")
    print("Press 'q' to stop early")
    
    frame_count = 0
    max_frames = int(fps * 5)  # 5 seconds
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        writer.write(frame)
        cv2.imshow("Recording...", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    print(f"Created test video: {output_path}")
    return output_path

def run_detector(video_path):
    """Run the person detector on the video."""
    from bus_tracker.detector import PersonDetector
    
    print("\n" + "="*60)
    print("INITIALIZING PERSON DETECTOR")
    print("="*60)
    
    # Initialize detector
    detector = PersonDetector(
        frame_skip=2,      # Process every 2nd frame for speed
        batch_size=4,      # Batch processing
        enable_profiling=True
    )
    
    print(f"\nModel: {detector.model_path}")
    print(f"Device: {'GPU' if detector.use_gpu else 'CPU'}")
    print(f"Frame Skip: {detector.frame_skip}")
    print(f"Batch Size: {detector.batch_size}")
    
    # Process video
    print("\n" + "="*60)
    print("PROCESSING VIDEO")
    print("="*60)
    print(f"Input: {video_path}")
    print("Press 'q' to stop, 'p' to pause\n")
    
    output_path = "output/videos/test_output.mp4"
    os.makedirs("output/videos", exist_ok=True)
    
    results = detector.process_video(
        video_path=video_path,
        output_path=output_path,
        show_dashboard=True  # Show live visualization
    )
    
    # Print results
    print("\n" + "="*60)
    print("DETECTION RESULTS")
    print("="*60)
    print(f"Frames Processed: {results['total_frames_processed']}")
    print(f"Unique People Detected: {results['unique_people_count']}")
    print(f"Processing Time: {results['processing_duration']:.1f}s")
    print(f"Average FPS: {results['average_fps']:.1f}")
    print(f"Output Video: {output_path}")
    print("="*60)
    
    return results

def main():
    print("="*60)
    print("BUS VISION - PERSON DETECTOR TEST")
    print("="*60)
    
    # Step 1: Get a video to test with
    video_path = None
    
    # Check if video path provided as argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            video_path = None
    
    # Check input folder for existing videos
    if video_path is None:
        input_dir = "input"
        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(input_dir, f)
                    print(f"Found existing video: {video_path}")
                    break
    
    # Download sample if no video found
    if video_path is None:
        video_path = download_sample_video()
    
    if video_path is None:
        print("\nNo video available for testing.")
        print("Please place a video file in the 'input' folder.")
        return
    
    # Step 2: Run detector
    try:
        results = run_detector(video_path)
        print("\n✅ Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
