#!/usr/bin/env python
# coding=utf-8

"""
Script to download WebVid-10M dataset from Hugging Face and prepare it for LLaMA Factory.
"""

import os
import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests
from datasets import load_dataset

# Instruction templates for video understanding
INSTRUCTION_TEMPLATES = [
    # Description prompts
    {"question": "Please describe what is happening in this video.", "prefix": "In this video, "},
    {"question": "Can you explain the main activity in this video?", "prefix": "The main activity shows "},
    {"question": "What's happening in this footage?", "prefix": "This footage shows "},
    
    # Analysis prompts
    {"question": "What do you observe in this video?", "prefix": "I observe "},
    {"question": "What actions are taking place in this clip?", "prefix": "The actions include "},
    
    # Focus prompts
    {"question": "What is the most notable aspect of this video?", "prefix": "The most notable aspect is "},
    {"question": "What's the key element in this video?", "prefix": "The key element is "}
]

class WebVidDownloader:
    """Downloader for WebVid dataset from Hugging Face."""
    
    def __init__(
        self,
        output_dir: str,
        num_videos: int = 5000,
        num_workers: int = 4
    ):
        self.output_dir = Path(output_dir)
        self.video_dir = self.output_dir / "videos"
        self.num_videos = num_videos
        self.num_workers = num_workers
        
        # Create directories
        self.video_dir.mkdir(parents=True, exist_ok=True)
    
    def download_video(self, video_url: str, video_id: str) -> bool:
        """Download a single video."""
        output_path = self.video_dir / f"{video_id}.mp4"
        
        if output_path.exists():
            return True
            
        try:
            response = requests.get(video_url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Verify content type
            content_type = response.headers.get('content-type', '')
            if 'video' not in content_type and 'application' not in content_type:
                return False
            
            # Save video
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Verify file size
            if output_path.stat().st_size < 10000:  # Less than 10KB is probably an error
                output_path.unlink()
                return False
                
            return True
            
        except Exception as e:
            print(f"\nError downloading {video_id}: {str(e)}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def create_conversation(self, video_id: str, caption: str) -> dict:
        """Create an instructional conversation for a video."""
        # Select random template
        template = random.choice(INSTRUCTION_TEMPLATES)
        
        # Clean up caption
        caption = caption.strip()
        if caption.endswith('.'):
            caption = caption[:-1]
        
        # Add prefix and format
        if not caption.lower().startswith(template['prefix'].lower()):
            caption = template['prefix'] + caption.lower()
        
        if not caption.endswith('.'):
            caption += '.'
            
        # Create conversation
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"<video>{template['question']}"
                },
                {
                    "role": "assistant",
                    "content": caption
                }
            ],
            "videos": [f"videos/{video_id}.mp4"]
        }
    
    def prepare_dataset(self):
        """Download videos and prepare instruction-tuned data."""
        print("Loading WebVid dataset from Hugging Face...")
        dataset = load_dataset("TempoFunk/webvid-10M", split="train")
        
        # Sample videos
        if len(dataset) > self.num_videos:
            indices = random.sample(range(len(dataset)), self.num_videos)
            dataset = dataset.select(indices)
        
        # Download videos
        print(f"\nDownloading {len(dataset)} videos...")
        success_count = 0
        failed_videos = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for item in dataset:
                futures.append(
                    executor.submit(
                        self.download_video,
                        item['contentUrl'],
                        str(item['videoid'])
                    )
                )
            
            for i, future in enumerate(tqdm(futures, total=len(futures), desc="Downloading")):
                if future.result():
                    success_count += 1
                else:
                    failed_videos.append(str(dataset[i]['videoid']))
        
        # Prepare instruction-tuned data
        print("\nCreating instruction-tuned dataset...")
        formatted_data = []
        
        for item in dataset:
            video_id = str(item['videoid'])
            if video_id not in failed_videos:
                conversation = self.create_conversation(
                    video_id,
                    item['name']
                )
                formatted_data.append(conversation)
        
        # Save formatted data
        output_file = self.output_dir / "webvid_llama.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
        # Create training configuration
        config = {
            "model_name_or_path": "Qwen/Qwen2-VL-7B-Instruct",
            "template": "qwen2_vl",
            "trust_remote_code": True,
            
            # Training method
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "lora",
            "freeze_vision_tower": True,
            "train_mm_proj_only": True,
            
            # Dataset
            "dataset": str(output_file),
            "cutoff_len": 2048,
            "preprocessing_num_workers": 4,
            
            # LoRA settings
            "lora_rank": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.05,
            "lora_target": "q_proj,v_proj",
            
            # Output settings
            "output_dir": str(self.output_dir / "model"),
            "logging_steps": 10,
            "save_steps": 100,
            "save_total_limit": 3,
            
            # Training parameters
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2e-4,
            "num_train_epochs": 3,
            "warmup_ratio": 0.1,
            "bf16": True,
            
            # Validation
            "val_size": 0.1,
            "evaluation_strategy": "steps",
            "eval_steps": 100,
            
            # System settings
            "gradient_checkpointing": True
        }
        
        config_file = self.output_dir / "train_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            import yaml
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        
        print(f"\nSuccessfully downloaded {success_count} videos")
        print(f"Failed to download {len(failed_videos)} videos")
        print(f"\nFiles created:")
        print(f"1. Videos: {self.video_dir}")
        print(f"2. Dataset: {output_file}")
        print(f"3. Training config: {config_file}")
        if formatted_data:
            print(f"\nExample conversation:")
            print(json.dumps(formatted_data[0], indent=2))
        print(f"\nTo start training, run:")
        print(f"llamafactory-cli train {config_file}")

def main():
    # Configuration
    output_dir = "/l/users/mukul.ranjan/video_data"
    num_videos = 5000  # Adjust based on your needs
    num_workers = 4   # Adjust based on your CPU and network
    
    downloader = WebVidDownloader(
        output_dir=output_dir,
        num_videos=num_videos,
        num_workers=num_workers
    )
    
    downloader.prepare_dataset()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")