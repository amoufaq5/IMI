"""
IMI ORPO (Odds Ratio Preference Optimization) Trainer
Combines SFT and preference alignment in a single training stage
More efficient than DPO as it doesn't require a reference model
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import ORPOTrainer, ORPOConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class IMIORPOConfig:
    """Configuration for ORPO training."""
    
    # Model
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    output_dir: str = "outputs/imi-medical-orpo"
    
    # Training mode
    training_mode: str = "qlora"
    
    # ORPO specific
    beta: float = 0.1  # Weight for odds ratio loss
    
    # Sequence length
    max_length: int = 2048
    max_prompt_length: int = 1024
    
    # Batch size
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    
    # Learning rate
    learning_rate: float = 8e-6
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    
    # Training duration
    num_train_epochs: int = 3
    max_steps: int = -1
    
    # Optimizer
    optim: str = "paged_adamw_8bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Precision
    bf16: bool = True
    
    # LoRA config
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # QLoRA
    load_in_4bit: bool = True
    
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
    train_data_path: str = "data/processed/dpo_train.jsonl"
    
    # Misc
    seed: int = 42


class IMIORPOTrainer:
    """
    ORPO trainer - combines SFT and preference alignment.
    Advantage: No need for reference model, more memory efficient.
    """
    
    def __init__(self, config: IMIORPOConfig):
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
            padding_side="left",
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def setup_model(self):
        """Initialize model."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        bnb_config = None
        if self.config.training_mode == "qlora" and self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }
        
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        
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
        
        logger.info("Model setup complete")
    
    def load_data(self) -> Dataset:
        """Load preference data."""
        logger.info(f"Loading data from: {self.config.train_data_path}")
        
        data = []
        with open(self.config.train_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append({
                    "prompt": item["prompt"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                })
        
        dataset = Dataset.from_list(data)
        logger.info(f"Loaded {len(dataset)} examples")
        
        return dataset
    
    def setup_trainer(self, train_dataset: Dataset):
        """Setup ORPO trainer."""
        logger.info("Setting up ORPO trainer...")
        
        training_args = ORPOConfig(
            output_dir=self.config.output_dir,
            
            # ORPO specific
            beta=self.config.beta,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
            
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
            
            # Checkpointing
            gradient_checkpointing=self.config.gradient_checkpointing,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            
            # Evaluation
            eval_strategy="no",
            
            # Logging
            logging_steps=self.config.logging_steps,
            report_to=self.config.report_to,
            
            # Misc
            seed=self.config.seed,
            remove_unused_columns=False,
        )
        
        self.trainer = ORPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )
        
        logger.info("ORPO trainer setup complete")
    
    def train(self):
        """Run ORPO training."""
        logger.info("Starting ORPO training...")
        
        train_result = self.trainer.train()
        
        logger.info("Saving final model...")
        self.trainer.save_model()
        
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        logger.info("ORPO training complete!")
        return metrics
    
    def run(self):
        """Full ORPO training pipeline."""
        self.setup_tokenizer()
        self.setup_model()
        
        train_dataset = self.load_data()
        self.setup_trainer(train_dataset)
        
        metrics = self.train()
        return metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="IMI ORPO Training")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3", help="Base model")
    parser.add_argument("--output-dir", default="outputs/imi-medical-orpo", help="Output directory")
    parser.add_argument("--train-data", default="data/processed/dpo_train.jsonl", help="Training data")
    parser.add_argument("--beta", type=float, default=0.1, help="ORPO beta")
    parser.add_argument("--lr", type=float, default=8e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()
    
    config = IMIORPOConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        train_data_path=args.train_data,
        beta=args.beta,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        report_to="none" if args.no_wandb else "wandb",
    )
    
    trainer = IMIORPOTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
