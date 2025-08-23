# -*- coding: utf-8 -*-
"""
Stage 3 - 2D Pose Estimation using easy_ViTPose

This script is adapted to use the easy_ViTPose library. It takes tracking
results (a .npz file with bounding boxes) and the source images to perform
2D pose estimation with a variety of ViT-Pose models.
"""
import sys
import os

# Add the project's root directory to the Python path
# This assumes your script is in the '/content/' directory
project_root = '/content/easy_ViTPose'
sys.path.insert(0, project_root)

# Now your regular imports will work
from easy_ViTPose.inference import VitInference
# --- 1. Imports and Setup ---
import cv2
import numpy as np
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from easy_ViTPose.inference import VitInference

# --- 2. Visualization and Helper Functions ---

class FrameSource:
    def __init__(self, source_path_str):
        self.source_path = Path(source_path_str)
        self.is_video = self.source_path.is_file()
        self.frame_files = []
        self.cap = None
        self.width, self.height = 0, 0
        self.fps = 30

        if self.is_video:
            self.cap = cv2.VideoCapture(source_path_str)
            if not self.cap.isOpened(): raise IOError(f"Could not open video file: {source_path_str}")
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            image_extensions = ['*.png', '*.jpg', '*.jpeg']
            self.frame_files = sorted([p for ext in image_extensions for p in self.source_path.glob(ext)])
            if not self.frame_files: raise IOError(f"No images found in folder: {source_path_str}")
            first_frame = cv2.imread(str(self.frame_files[0]))
            if first_frame is None: raise IOError(f"Could not read first image: {self.frame_files[0]}")
            self.height, self.width, _ = first_frame.shape
        print(f"Source dimensions: {self.width}x{self.height}, FPS: {self.fps:.2f}")

    def get_frame(self, frame_index):
        frame_bgr = None
        if self.is_video:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame_bgr = self.cap.read()
            if not success: return None
        else:
            if 0 <= frame_index < len(self.frame_files):
                frame_bgr = cv2.imread(str(self.frame_files[frame_index]))
            else:
                return None
        
        return frame_bgr

    def release(self):
        if self.is_video and self.cap:
            self.cap.release()

def run_pose_estimation(npz_path, source_path, output_npz_path, model_path, model_name, dataset, viz_output_path, device, viz_confidence):
    try:
        track_data = np.load(npz_path)
        frame_indices = track_data['frame_indices']
        # Bboxes are loaded in (x1, y1, x2, y2) format
        bboxes_voc = track_data['bboxes']
    except Exception as e:
        print(f"Error loading .npz file: {e}")
        return

    # Initialize the VitInference model
    try:
        model = VitInference(model=model_path, model_name=model_name, dataset=dataset, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    frame_source = FrameSource(source_path)
    video_writer = None
    if viz_output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(viz_output_path, fourcc, frame_source.fps, (frame_source.width, frame_source.height))

    all_keypoints_combined = []
    progress_bar = tqdm(total=len(frame_indices), desc="Estimating Poses")

    for i in range(len(frame_indices)):
        frame_idx = frame_indices[i]
        bbox_voc = bboxes_voc[i] # Bbox in (x1, y1, x2, y2) format

        frame_bgr = frame_source.get_frame(frame_idx)
        
        if frame_bgr is None:
            # If frame is not found, append zeros and continue
            num_keypoints = 17 # Default to COCO 17 keypoints
            if dataset == 'coco_25': num_keypoints = 25
            elif dataset == 'wholebody': num_keypoints = 133
            all_keypoints_combined.append(np.zeros((num_keypoints, 3), dtype=np.float32))
            progress_bar.update(1)
            continue
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Prepare bounding box for inference_with_bboxes
        # The method expects a score, so we add a dummy score of 1.0
        bbox_with_score = np.append(bbox_voc, 1.0).reshape(1, 5)

        # Run inference using the provided bounding box
        keypoints_data = model.inference_with_bboxes(frame_rgb, bbox_with_score)
        
        # Extract the keypoints for the first (and only) person
        person_id = list(keypoints_data.keys())[0]
        keypoints_with_scores = keypoints_data[person_id]
        all_keypoints_combined.append(keypoints_with_scores)

        if video_writer:
            # Use the built-in draw method from the model for visualization
            viz_frame = model.draw(confidence_threshold=viz_confidence)
            video_writer.write(cv2.cvtColor(viz_frame, cv2.COLOR_RGB2BGR))
            
        progress_bar.update(1)

    progress_bar.close()

    np.savez_compressed(output_npz_path, frame_indices=frame_indices, keypoints=np.array(all_keypoints_combined, dtype=np.float32))
    print(f"Successfully saved keypoints to {output_npz_path}")

    frame_source.release()
    if video_writer:
        video_writer.release()
        print(f"Successfully saved visualization to {viz_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 2D pose estimation using the easy-ViTPose library.")
    parser.add_argument("--npz_path", required=True, help="Path to the input .npz file from the tracking script.")
    parser.add_argument("--source", required=True, help="Path to the original input video file or image folder.")
    parser.add_argument("--output_npz", required=True, help="Path to save the output pose keypoints .npz file.")
    parser.add_argument("--model", required=True, help="Path to the ViT-Pose model checkpoint (.pth, .onnx, .engine).")
    parser.add_argument("--model-name", type=str, required=True, choices=['s', 'b', 'l', 'h'], help="Model size: 's', 'b', 'l', or 'h'.")
    parser.add_argument("--dataset", type=str, default='coco', help="Dataset the model was trained on (e.g., 'coco', 'wholebody').")
    parser.add_argument("--viz", type=str, default=None, help="Optional: Path to save a visualization video.")
    parser.add_argument("--viz_conf", type=float, default=0.5, help="Confidence threshold for drawing keypoints.")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    run_pose_estimation(
        npz_path=args.npz_path,
        source_path=args.source,
        output_npz_path=args.output_npz,
        model_path=args.model,
        model_name=args.model_name,
        dataset=args.dataset,
        viz_output_path=args.viz,
        device=device,
        viz_confidence=args.viz_conf
    )