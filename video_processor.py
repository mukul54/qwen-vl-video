#!/usr/bin/env python
# coding=utf-8

"""
Test script to debug video processing with Qwen2-VL.
"""

import os
import torch
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from transformers import Qwen2VLProcessor

def test_video_processing(video_path: str):
    """Test video processing with detailed debug info."""
    print(f"\nTesting video: {video_path}")
    
    # Check if file exists
    if not os.path.exists(video_path):
        print("Error: Video file not found")
        return
    
    print(f"File size: {os.path.getsize(video_path)} bytes")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
        
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info:")
    print(f"- Total frames: {total_frames}")
    print(f"- FPS: {fps}")
    print(f"- Resolution: {width}x{height}")
    
    try:
        # Sample frames
        frames = []
        frame_indices = [0] if total_frames < 2 else np.linspace(0, total_frames-1, 8, dtype=int)
        
        for idx in frame_indices:
            print(f"\nProcessing frame {idx}...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Error: Could not read frame {idx}")
                continue
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"Frame shape: {frame.shape}")
            
            frames.append(frame)
            print(f"Frame {idx} processed successfully")
        
        cap.release()
        
        if not frames:
            print("Error: No frames extracted")
            return
        
        print(f"\nExtracted {len(frames)} frames successfully")
        
        # Load processor
        print("\nLoading Qwen2-VL processor...")
        processor = Qwen2VLProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            trust_remote_code=True
        )
        
        # Process frames
        print("Processing frames with Qwen2-VL processor...")
        inputs = processor.image_processor(images=frames, return_tensors="pt")
        
        print("\nProcessor output keys:", inputs.keys())
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"{key} shape: {value.shape}")
        
        print("\nProcessing completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if cap is not None:
            cap.release()

def main():
    # Test with one video
    video_dir = "/l/users/mukul.ranjan/video_data/videos"
    test_video = os.path.join(video_dir, "34422271.mp4")  # Using first video from your list
    
    success = test_video_processing(test_video)
    print(f"\nTest {'succeeded' if success else 'failed'}")

if __name__ == "__main__":
    main()