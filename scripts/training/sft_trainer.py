"""
IMI SFT (Supervised Fine-Tuning) Trainer
Optimized for medical reasoning with multi-GPU support
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class IMITrainingConfig:
    """Configuration for IMI medical model training."""
    
    # Model
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    output_dir: str = "outputs/imi-medical"
    
    # Training mode: "full", "lora", "qlora"
    training_mode: str = "qlora"
    
    # Sequence length (8192 for long medical documents)
    max_seq_length: int = 4096
    
    # Batch size (per device)
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    
    # Learning rate
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    
    # Training duration
    num_train_epochs: int = 3
    max_steps: int = -1
    
    # Optimizer
    optim: str = "paged_adamw_8bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Precision
    bf16: bool = True
    fp16: bool = False
    
    # LoRA config
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # QLoRA (4-bit quantization)
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # Checkpointing
    gradient_checkpointing: bool = True
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 500
    
    # Logging
    logging_steps: int = 10
    report_to: str = "wandb"
    
    # Data
    train_data_path: str = "data/processed/sft_train.jsonl"
    eval_data_path: str = "data/processed/sft_val.jsonl"
    
    # Misc
    seed: int = 42
    dataloader_num_workers: int = 4
    group_by_length: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class IMISFTTrainer:
    """
    Supervised Fine-Tuning trainer for IMI medical model.
    Supports full fine-tuning, LoRA, and QLoRA.
    """
    
    def __init__(self, config: IMITrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def setup_tokenizer(self):
        """Initialize tokenizer."""
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right",
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
    
    def setup_model(self):
        """Initialize model with appropriate configuration."""
        logger.info(f"Loading model: {self.config.model_name}")
        logger.info(f"Training mode: {self.config.training_mode}")
        
        # Quantization config for QLoRA
        bnb_config = None
        if self.config.training_mode == "qlora" and self.config.load_in_4bit:
            compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            )
            logger.info("Using 4-bit quantization (QLoRA)")
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if self.config.bf16 else torch.float16,
            "device_map": "auto",
        }
        
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        
        # Apply LoRA if not full fine-tuning
        if self.config.training_mode in ["lora", "qlora"]:
            if self.config.training_mode == "qlora":
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=self.config.gradient_checkpointing,
                )
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        logger.info("Model loaded successfully")
    
    def load_data(self) -> tuple:
        """Load training and evaluation datasets."""
        logger.info("Loading datasets...")
        
        def load_jsonl(path: str) -> Dataset:
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            return Dataset.from_list(data)
        
        train_dataset = load_jsonl(self.config.train_data_path)
        eval_dataset = load_jsonl(self.config.eval_data_path)
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Eval samples: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def format_chat(self, example: Dict) -> str:
        """Format example into chat template."""
        messages = example.get("messages", [])
        
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        
        # Fallback: simple format
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                text += f"[INST] {content} [/INST]"
            else:
                text += f" {content}</s>"
        
        return text
    
    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Dataset):
        """Setup SFT trainer."""
        logger.info("Setting up trainer...")
        
        # Training arguments
        training_args = SFTConfig(
            output_dir=self.config.output_dir,
            
            # Batch size
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            # Learning rate
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            
            # Duration
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps,
            
            # Optimizer
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            
            # Precision
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            
            # Checkpointing
            gradient_checkpointing=self.config.gradient_checkpointing,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            
            # Evaluation
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            
            # Logging
            logging_steps=self.config.logging_steps,
            report_to=self.config.report_to,
            
            # Misc
            seed=self.config.seed,
            dataloader_num_workers=self.config.dataloader_num_workers,
            group_by_length=self.config.group_by_length,
            
            # SFT specific
            max_seq_length=self.config.max_seq_length,
            packing=False,
            dataset_text_field="text",
        )
        
        # Format dataset
        def format_example(example):
            return {"text": self.format_chat(example)}
        
        train_dataset = train_dataset.map(format_example)
        eval_dataset = eval_dataset.map(format_example)
        
        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        logger.info("Trainer setup complete")
    
    def train(self):
        """Run training."""
        logger.info("Starting training...")
        
        # Train
        train_result = self.trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        self.trainer.save_model()
        
        # Save metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        logger.info("Training complete!")
        return metrics
    
    def run(self):
        """Full training pipeline."""
        # Setup
        self.setup_tokenizer()
        self.setup_model()
        
        # Load data
        train_dataset, eval_dataset = self.load_data()
        
        # Setup trainer
        self.setup_trainer(train_dataset, eval_dataset)
        
        # Train
        metrics = self.train()
        
        return metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="IMI SFT Training")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3", help="Base model")
    parser.add_argument("--output-dir", default="outputs/imi-medical", help="Output directory")
    parser.add_argument("--mode", choices=["full", "lora", "qlora"], default="qlora", help="Training mode")
    parser.add_argument("--train-data", default="data/processed/sft_train.jsonl", help="Training data")
    parser.add_argument("--eval-data", default="data/processed/sft_val.jsonl", help="Evaluation data")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()
    
    # Create config
    config = IMITrainingConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        training_mode=args.mode,
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lora_r=args.lora_r,
        report_to="none" if args.no_wandb else "wandb",
    )
    
    # Run training
    trainer = IMISFTTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
