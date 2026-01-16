"""
UMI Medical Model Fine-Tuning Pipeline
Fine-tunes Mistral-7B on medical domain using LoRA/QLoRA
"""

import os
import json
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer


@dataclass
class UMITrainingConfig:
    """Configuration for UMI model fine-tuning."""
    
    # Model
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    output_dir: str = "models/umi-medical-v1"
    
    # Data
    train_data: str = "data/training/train.jsonl"
    val_data: str = "data/training/val.jsonl"
    max_seq_length: int = 2048
    
    # LoRA Configuration
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # Quantization (for QLoRA)
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    
    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Optimization
    optim: str = "paged_adamw_32bit"
    fp16: bool = False
    bf16: bool = True
    max_grad_norm: float = 0.3
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Misc
    seed: int = 42
    report_to: str = "tensorboard"


class MedicalChatFormatter:
    """Formats medical QA data for Mistral chat fine-tuning."""
    
    SYSTEM_PROMPT = """You are UMI, a medical AI assistant. You provide accurate, evidence-based medical information while following the ASMETHOD protocol for patient consultations.

Key principles:
1. Patient safety is the top priority
2. Identify danger signs requiring immediate medical attention
3. Provide clear, actionable recommendations
4. Always recommend professional consultation for serious conditions
5. Be empathetic and clear in communication

You are NOT a replacement for professional medical advice."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def format_chat(self, example: Dict[str, Any]) -> str:
        """Format a single example for training."""
        messages = example.get("messages", [])
        
        # Add system prompt
        full_messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ] + messages
        
        # Apply chat template
        formatted = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        return formatted
    
    def format_instruction(self, example: Dict[str, Any]) -> str:
        """Format instruction-style data."""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        if input_text:
            user_content = f"{instruction}\n\nContext: {input_text}"
        else:
            user_content = instruction
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )


class UMIFineTuner:
    """
    Fine-tuning pipeline for UMI medical model.
    """
    
    def __init__(self, config: UMITrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def setup_quantization(self) -> Optional[BitsAndBytesConfig]:
        """Configure 4-bit quantization for QLoRA."""
        if not self.config.use_4bit:
            return None
        
        compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
        
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.config.use_nested_quant,
        )
    
    def load_model(self) -> None:
        """Load base model and tokenizer."""
        print(f"Loading base model: {self.config.base_model}")
        
        # Quantization config
        bnb_config = self.setup_quantization()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Prepare for k-bit training
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Disable cache for training
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        
        print(f"Model loaded. Parameters: {self.model.num_parameters():,}")
    
    def setup_lora(self) -> None:
        """Configure and apply LoRA adapters."""
        print("Setting up LoRA adapters...")
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable, total = self.model.get_nb_trainable_parameters()
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    
    def load_data(self) -> tuple:
        """Load and preprocess training data."""
        print("Loading training data...")
        
        # Load datasets
        train_dataset = load_dataset(
            "json",
            data_files=self.config.train_data,
            split="train",
        )
        
        val_dataset = None
        if os.path.exists(self.config.val_data):
            val_dataset = load_dataset(
                "json",
                data_files=self.config.val_data,
                split="train",
            )
        
        print(f"Train samples: {len(train_dataset)}")
        if val_dataset:
            print(f"Val samples: {len(val_dataset)}")
        
        # Format data
        formatter = MedicalChatFormatter(self.tokenizer)
        
        def format_sample(example):
            if "messages" in example:
                text = formatter.format_chat(example)
            else:
                text = formatter.format_instruction(example)
            return {"text": text}
        
        train_dataset = train_dataset.map(format_sample)
        if val_dataset:
            val_dataset = val_dataset.map(format_sample)
        
        return train_dataset, val_dataset
    
    def setup_trainer(self, train_dataset, val_dataset) -> None:
        """Configure the SFT trainer."""
        print("Setting up trainer...")
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            optim=self.config.optim,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if val_dataset else None,
            evaluation_strategy="steps" if val_dataset else "no",
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True if val_dataset else False,
            report_to=self.config.report_to,
            seed=self.config.seed,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            packing=False,
        )
    
    def train(self) -> None:
        """Run the training loop."""
        print("=" * 60)
        print("Starting training...")
        print("=" * 60)
        
        # Train
        self.trainer.train()
        
        # Save final model
        print("Saving final model...")
        self.trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save training config
        config_path = Path(self.config.output_dir) / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        print(f"Model saved to: {self.config.output_dir}")
    
    def run(self) -> None:
        """Execute the full fine-tuning pipeline."""
        # Load model
        self.load_model()
        
        # Setup LoRA
        self.setup_lora()
        
        # Load data
        train_dataset, val_dataset = self.load_data()
        
        # Setup trainer
        self.setup_trainer(train_dataset, val_dataset)
        
        # Train
        self.train()


def merge_lora_weights(
    base_model: str,
    lora_model: str,
    output_dir: str,
) -> None:
    """
    Merge LoRA weights with base model for deployment.
    Creates a standalone model without adapter overhead.
    """
    from peft import PeftModel
    
    print(f"Merging LoRA weights...")
    print(f"Base model: {base_model}")
    print(f"LoRA model: {lora_model}")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Load and merge LoRA
    model = PeftModel.from_pretrained(model, lora_model)
    model = model.merge_and_unload()
    
    # Save merged model
    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Merge complete!")


def main():
    """Run the fine-tuning pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="UMI Medical Model Fine-Tuning")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA weights after training")
    parser.add_argument("--base-model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--output-dir", type=str, default="models/umi-medical-v1")
    parser.add_argument("--train-data", type=str, default="data/training/train.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = UMITrainingConfig(**config_dict)
    else:
        config = UMITrainingConfig(
            base_model=args.base_model,
            output_dir=args.output_dir,
            train_data=args.train_data,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    
    print("=" * 60)
    print("UMI Medical Model Fine-Tuning Pipeline")
    print("=" * 60)
    print(f"Base Model: {config.base_model}")
    print(f"Output: {config.output_dir}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Batch Size: {config.per_device_train_batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print("=" * 60)
    
    # Run fine-tuning
    fine_tuner = UMIFineTuner(config)
    fine_tuner.run()
    
    # Optionally merge weights
    if args.merge:
        merged_dir = f"{config.output_dir}-merged"
        merge_lora_weights(
            base_model=config.base_model,
            lora_model=config.output_dir,
            output_dir=merged_dir,
        )


if __name__ == "__main__":
    main()
