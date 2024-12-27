#!/usr/bin/env python
# coding=utf-8

"""
Training script for video understanding using Qwen2-VL and LLaMA Factory.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass

@dataclass
class VideoTrainingConfig:
    """Training configuration for video fine-tuning."""
    
    # Paths
    config_base_dir: str = "./configs"
    base_dir: str = "/l/users/mukul.ranjan/video_data"
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    dataset_path: str = "webvid_llama.json"
    
    # Model config
    template: str = "qwen2_vl"
    max_length: int = 2048
    
    # Training method
    finetuning_type: str = "lora"  # Efficient fine-tuning with LoRA
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target: str = "q_proj,v_proj"
    
    # Training settings
    batch_size: int = 4
    gradient_accumulation: int = 4  # Effective batch size = 16
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    
    # Performance
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"
    gradient_checkpointing: bool = True
    
    # Saving & Evaluation
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    val_size: float = 0.1

def create_training_config(config: VideoTrainingConfig) -> dict:
    """Create LLaMA Factory training configuration."""
    base_dir = Path(config.base_dir)
    
    return {
        # Model configuration
        "model_name_or_path": config.model_name,
        "template": config.template,
        "trust_remote_code": True,
        
        # Training method
        "stage": "sft",
        "do_train": True,
        "finetuning_type": config.finetuning_type,
        "freeze_vision_tower": True,
        "train_mm_proj_only": False,
        
        # Dataset settings
        "dataset": str(base_dir / config.dataset_path),
        "cutoff_len": config.max_length,
        "preprocessing_num_workers": 4,
        
        # LoRA settings
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "lora_target": config.lora_target,
        
        # Output settings
        "output_dir": str(base_dir / "model"),
        "logging_steps": 10,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "plot_loss": True,
        "overwrite_output_dir": True,
        
        # Training parameters
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": config.warmup_ratio,
        
        # Validation
        "val_size": config.val_size,
        "evaluation_strategy": "steps",
        "eval_steps": config.eval_steps,
        
        # Performance settings
        "bf16": config.mixed_precision == "bf16",
        "fp16": config.mixed_precision == "fp16",
        "gradient_checkpointing": config.gradient_checkpointing,
        "ddp_timeout": 180000000,
        
        # System settings
        "seed": 42,
        "double_quant": False,
        "neftune_noise_alpha": 5,  # Enable NEFTune for better convergence
    }

def create_deepspeed_config() -> dict:
    """Create DeepSpeed ZeRO-3 configuration."""
    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto"
        },
        
        "bf16": {
            "enabled": True
        },
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": "auto"
            }
        },
        
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
                "total_num_steps": "auto"
            }
        }
    }

def main():
    # Create base configuration
    config = VideoTrainingConfig()
    
    # Create necessary directories
    base_dir = Path(config.base_dir)
    config_base_dir = Path(config.config_base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training config
    train_config = create_training_config(config)
    train_config_path = base_dir / "train_config.yaml"
    
    with open(train_config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(train_config, f, sort_keys=False, allow_unicode=True)
    
    # Generate DeepSpeed config
    ds_config = create_deepspeed_config()
    ds_config_path = base_dir / "ds_config.json"
    
    with open(ds_config_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(ds_config, f, indent=2)
    
    print("\nConfiguration files created:")
    print(f"1. Training config: {train_config_path}")
    print(f"2. DeepSpeed config: {ds_config_path}")
    print("\nTo start single-GPU training:")
    print(f"llamafactory-cli train {train_config_path}")
    print("\nTo start multi-GPU training with DeepSpeed:")
    print(f"deepspeed --num_gpus=8 -m llamafactory.train --train_config={train_config_path} --deepspeed={ds_config_path}")

if __name__ == "__main__":
    main()