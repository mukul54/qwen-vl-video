#!/usr/bin/env python
# coding=utf-8

"""
Script for evaluating fine-tuned Qwen2-VL model on video understanding tasks.
"""

import os
import json
import torch
import evaluate
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2VLProcessor
from transformers import Qwen2VLForConditionalGeneration
from peft import PeftModel
import argparse
import cv2
import numpy as np
from typing import List, Optional, Dict, Union

class VideoModelEvaluator:
    def __init__(
        self,
        base_model_path: str,
        adapter_path: str,
        video_dir: str,
        processor_path: str = None,
        frame_size: tuple = (224, 224)  # Default frame size
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.video_dir = Path(video_dir)
        self.frame_size = frame_size
        
        print(f"\nUsing device: {self.device}")
        print(f"Video directory: {self.video_dir}")
        print(f"Frame size: {self.frame_size}")
        
        # Load base model
        print("Loading model...")
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "device_map": "auto",  # Enable automatic device mapping
            "low_cpu_mem_usage": True,
        }
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_path,
            **model_kwargs
        )
        
        # Load LoRA adapter
        if adapter_path:
            print("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                device_map="auto"  # Enable automatic device mapping for adapter
            )
            
        # Move to device after loading adapter
        self.model.to(self.device)
        
        # Load tokenizer and processor
        print("Loading tokenizer and processor...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        self.processor = Qwen2VLProcessor.from_pretrained(
            processor_path or base_model_path,
            trust_remote_code=True
        )
        
        # Set generation config
        self.model.config.max_new_tokens = 128
        self.model.config.temperature = 0.7
        self.model.config.top_p = 0.7
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame to ensure consistent format.
        
        Args:
            frame (np.ndarray): Input frame in BGR format
            
        Returns:
            np.ndarray: Preprocessed frame
        """
        try:
            # Convert BGR to RGB if needed
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame
            frame = cv2.resize(frame, self.frame_size)
            
            # Convert to float32 and normalize
            frame = frame.astype(np.float32) / 255.0
            
            return frame
            
        except Exception as e:
            print(f"Error in frame preprocessing: {str(e)}")
            raise
    
    def extract_frames(self, video_path: str, num_frames: int = 8) -> np.ndarray:
        """
        Extract and preprocess frames from video file.
        
        Args:
            video_path (str): Path to video file
            num_frames (int): Number of frames to extract
            
        Returns:
            np.ndarray: Stack of preprocessed frames
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 1:
                raise ValueError(f"No frames in video: {video_path}")
            
            # Sample frame indices
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
            print(f"Extracting {num_frames} frames from total {total_frames} frames")
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    try:
                        frame = self.preprocess_frame(frame)
                        frames.append(frame)
                    except Exception as e:
                        print(f"Error preprocessing frame {idx}: {str(e)}")
                        continue
            
            if not frames:
                raise ValueError(f"No valid frames extracted from: {video_path}")
            
            # Stack frames and check shape
            frames = np.stack(frames)
            print(f"Extracted frames shape: {frames.shape}, dtype: {frames.dtype}")
            
            return frames
            
        finally:
            cap.release()
    
    def process_video(self, video_path: str, question: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Process video frames and text for the model.
        
        Args:
            video_path (str): Path to video file
            question (str): Question text
            
        Returns:
            Optional[Dict[str, torch.Tensor]]: Processed inputs for the model
        """
        try:
            # Extract frames
            frames = self.extract_frames(video_path)
            
            # Add a <video_pad> token for each batch of features
            num_frames = frames.shape[0] * frames.shape[1] * frames.shape[2] // 64  # Calculate number of video tokens needed
            video_tokens = '<video_pad>' * num_frames
            
            # Format using Qwen's chat template with correct number of video tokens
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that describes video content accurately and concisely."
                },
                {
                    "role": "user", 
                    "content": video_tokens + question
                }
            ]
            
            text_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            try:
                # Process frames and text together
                encoding = self.processor(
                    text=text_input,
                    videos=frames,
                    return_tensors="pt"
                )
                
                print(f"Processed input shapes: {[(k, v.shape) for k, v in encoding.items()]}")
                return encoding
                
            except Exception as e:
                print(f"Error in processor for {video_path}: {str(e)}")
                return None
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return None
    
    def generate_response(self, question: str, video_path: str) -> str:
        """
        Generate response for a video-question pair.
        
        Args:
            question (str): Question text
            video_path (str): Path to video file
            
        Returns:
            str: Generated response or error message
        """
        try:
            # Get full video path
            full_video_path = self.video_dir / video_path.replace('videos/', '')
            if not full_video_path.exists():
                print(f"\nVideo not found: {full_video_path}")
                return "[Video Not Found]"
            
            print(f"\nProcessing video: {full_video_path}")
            
            # Process video and text
            inputs = self.process_video(full_video_path, question)
            if inputs is None:
                return "[Video Processing Error]"
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            try:
                # Set up generation config
                gen_config = {
                    "max_new_tokens": 150,
                    "do_sample": True,
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id
                }
                
                print("\nGenerating response...")
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **gen_config
                    )
                
                # Print raw output for debugging
                raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                print(f"Raw response: {raw_response}")
                
                # Decode and clean response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Decoded response: {response}")
                
                # Clean up response
                response = response.replace(question, "").strip()
                if response.startswith("<|im_start|>"):
                    response = response.split("<|im_end|>")[0].replace("<|im_start|>", "").strip()
                
                print(f"Final cleaned response: {response}")
                
                if not response:
                    print("Warning: Empty response after processing")
                    return "[Empty Response]"
                
                return response
                
            except Exception as e:
                print(f"\nError generating response: {str(e)}")
                return "[Generation Error]"
            
        except Exception as e:
            print(f"\nError processing {video_path}: {str(e)}")
            return "[Error]"
    
    def evaluate_dataset(self, test_data_path: str, num_samples: int = None) -> dict:
        """
        Evaluate model on test dataset.
        
        Args:
            test_data_path (str): Path to test dataset JSON
            num_samples (int, optional): Number of samples to evaluate
            
        Returns:
            dict: Evaluation results and metrics
        """
        # Load test data
        try:
            with open(test_data_path, 'r') as f:
                test_data = json.load(f)
        except Exception as e:
            print(f"Error loading test data: {str(e)}")
            return {}
        
        # Sample subset if specified
        if num_samples and num_samples < len(test_data):
            import random
            random.seed(42)
            test_data = random.sample(test_data, num_samples)
        
        predictions = []
        references = []
        results = []
        
        # Generate predictions
        print(f"\nProcessing {len(test_data)} samples...")
        
        for item in tqdm(test_data, desc="Evaluating"):
            try:
                # Get question and reference
                question = item['instruction']
                reference = item['output']
                video_path = item['video']
                
                # Generate prediction
                prediction = self.generate_response(question, video_path)
                
                print("\nExample:")
                print(f"Video: {video_path}")
                print(f"Question: {question}")
                print(f"Prediction: {prediction}")
                print(f"Reference: {reference}")
                
                # Only include valid predictions for metrics
                if prediction not in ["[Video Not Found]", "[Error]", "[Video Processing Error]", "[Generation Error]", ""]:
                    predictions.append(prediction)
                    references.append(reference)
                
                # Store all results
                results.append({
                    'question': question,
                    'prediction': prediction,
                    'reference': reference,
                    'video': video_path
                })
                
            except Exception as e:
                print(f"Error processing item: {str(e)}")
                continue
        
        # Calculate metrics if we have valid predictions
        metrics = {}
        if predictions:
            print(f"\nCalculating metrics on {len(predictions)} valid predictions...")
            try:
                rouge = evaluate.load('rouge')
                metrics['rouge'] = rouge.compute(predictions=predictions, references=references)
            except Exception as e:
                print(f"Error computing ROUGE: {e}")
                
            try:
                bleu = evaluate.load('bleu')
                metrics['bleu'] = bleu.compute(predictions=predictions, references=[[r] for r in references])['bleu']
            except Exception as e:
                print(f"Error computing BLEU: {e}")
                
            try:
                meteor = evaluate.load('meteor')
                metrics['meteor'] = meteor.compute(predictions=predictions, references=references)['meteor']
            except Exception as e:
                print(f"Error computing METEOR: {e}")
        else:
            print("\nNo valid predictions for metric calculation!")
        
        return {
            'metrics': metrics,
            'results': results,
            'valid_predictions': len(predictions),
            'total_samples': len(test_data)
        }

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned video model")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                      help="Path to base model")
    parser.add_argument("--adapter-path", type=str, required=True,
                      help="Path to trained LoRA adapter")
    parser.add_argument("--test-data", type=str, required=True,
                      help="Path to test dataset JSON")
    parser.add_argument("--video-dir", type=str, required=True,
                      help="Directory containing the videos")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Directory to save evaluation results")
    parser.add_argument("--num-samples", type=int, default=10,
                      help="Number of samples to evaluate (default: 10)")
    parser.add_argument("--frame-size", type=int, nargs=2, default=[224, 224],
                      help="Frame size (height width) for preprocessing (default: 224 224)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = VideoModelEvaluator(
        base_model_path=args.base_model,
        adapter_path=args.adapter_path,
        video_dir=args.video_dir,
        frame_size=tuple(args.frame_size)
    )
    
    # Run evaluation
    results = evaluator.evaluate_dataset(args.test_data, args.num_samples)
    
    # Save results
    output_file = output_dir / "eval_results.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        return
    
    # Print summary
    print("\nEvaluation Summary:")
    print("-" * 50)
    print(f"Total samples processed: {results['total_samples']}")
    print(f"Valid predictions: {results['valid_predictions']}")
    
    if results.get('metrics'):
        print("\nMetrics:")
        for metric, value in results['metrics'].items():
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"{metric}-{k}: {v:.4f}")
            else:
                print(f"{metric}: {value:.4f}")
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()