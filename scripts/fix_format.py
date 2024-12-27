#!/usr/bin/env python
# coding=utf-8

"""
Script to fix data format for LLaMA Factory training.
"""

import json
import yaml
from pathlib import Path

def format_data_for_training(input_path: str, output_path: str):
    """Format data in alpaca format that LLaMA Factory expects."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        try:
            # Extract messages
            user_msg = next(msg for msg in item['messages'] if msg['role'] == 'user')
            assistant_msg = next(msg for msg in item['messages'] if msg['role'] == 'assistant')
            
            # Format in alpaca style
            formatted_item = {
                "instruction": user_msg['content'],
                "input": "",
                "output": assistant_msg['content'],
                "video": item['videos'][0]  # Take first video path
            }
            formatted_data.append(formatted_item)
        except Exception as e:
            print(f"Skipping malformed item: {str(e)}")
    
    # Save formatted data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)
    
    return len(formatted_data)

def main():
    # Setup paths
    code_dir = Path("/home/mukul.ranjan/projects/video-vlm/qwen-vl-video")
    data_dir = code_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    base_dir = Path("/l/users/mukul.ranjan/video_data")
    
    # Format data
    input_path = base_dir / "webvid_llama.json"
    output_path = base_dir / "webvid_formatted.json"
    
    num_samples = format_data_for_training(str(input_path), str(output_path))
    print(f"Successfully formatted {num_samples} samples")
    
    # Create dataset info in the code directory
    dataset_info = {
        "webvid_formatted": {
            "file_name": str(output_path),
            "type": "alpaca"  # Important: specify the format type
        }
    }
    
    with open(data_dir / "dataset_info.json", 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Create training config
    train_config = {
        # Model configuration
        "model_name_or_path": "Qwen/Qwen2-VL-7B-Instruct",
        "template": "alpaca",  # Use alpaca template for this format
        "trust_remote_code": True,
        
        # Training method
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "freeze_vision_tower": True,
        "train_mm_proj_only": True,
        
        # Dataset
        "dataset": "webvid_formatted",
        "dataset_dir": str(data_dir),  # Point to local data directory
        "cutoff_len": 2048,
        
        # LoRA settings
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "lora_target": "q_proj,v_proj",
        
        # Output settings
        "output_dir": str(base_dir / "model"),
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 3,
        
        # Training parameters
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "num_train_epochs": 3,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        
        # Mixed precision training
        "bf16": True,
        
        # System settings
        "seed": 42,
        "neftune_noise_alpha": 5,
        "gradient_checkpointing": True
    }
    
    config_path = base_dir / "train_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(train_config, f, sort_keys=False, allow_unicode=True)
    
    print("\nCreated files:")
    print(f"1. Formatted dataset: {output_path}")
    print(f"2. Dataset info: {data_dir}/dataset_info.json")
    print(f"3. Training config: {config_path}")
    print("\nTo start training, run:")
    print(f"llamafactory-cli train {config_path}")

if __name__ == "__main__":
    main()