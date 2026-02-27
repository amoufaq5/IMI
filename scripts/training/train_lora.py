"""
LoRA Training Pipeline for IMI Medical LLM

Complete training pipeline using PEFT (Parameter-Efficient Fine-Tuning):
- Loads Meditron base model
- Configures LoRA adapters
- Trains on domain-specific data
- Saves adapters for inference
"""
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ADAPTERS_DIR = PROJECT_ROOT / "adapters"


@dataclass
class AdapterTrainingConfig:
    """Configuration for adapter training"""
    name: str
    description: str
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    learning_rate: float = 1e-4
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    max_seq_length: int = 4096
    weight_decay: float = 0.01


# Adapter configurations
ADAPTER_CONFIGS = {
    "patient_triage": AdapterTrainingConfig(
        name="patient_triage",
        description="Patient symptom assessment and triage",
        lora_r=32,
        lora_alpha=64,
        learning_rate=1e-4,
        num_epochs=3,
    ),
    "clinical_pharmacist": AdapterTrainingConfig(
        name="clinical_pharmacist",
        description="Drug interactions and medication safety",
        lora_r=32,
        lora_alpha=64,
        learning_rate=1e-4,
        num_epochs=3,
    ),
    "clinical_decision": AdapterTrainingConfig(
        name="clinical_decision",
        description="Clinical decision support for doctors",
        lora_r=32,
        lora_alpha=64,
        learning_rate=5e-5,
        num_epochs=4,
    ),
    "education": AdapterTrainingConfig(
        name="education",
        description="Medical education and USMLE preparation",
        lora_r=32,
        lora_alpha=64,
        learning_rate=1e-4,
        num_epochs=3,
    ),
    "regulatory_qa": AdapterTrainingConfig(
        name="regulatory_qa",
        description="Pharmaceutical regulatory and QA",
        lora_r=32,
        lora_alpha=64,
        learning_rate=1e-4,
        num_epochs=3,
    ),
    "research": AdapterTrainingConfig(
        name="research",
        description="Medical research and literature synthesis",
        lora_r=32,
        lora_alpha=64,
        learning_rate=5e-5,
        num_epochs=4,
    ),
}


class InstructionDataset(Dataset):
    """Dataset for instruction-tuning"""
    
    def __init__(
        self,
        data_path: Path,
        tokenizer,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load data
        if data_path.exists():
            with open(data_path) as f:
                self.examples = json.load(f)
            logger.info(f"Loaded {len(self.examples)} examples from {data_path}")
        else:
            logger.warning(f"Data file not found: {data_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format as instruction
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        # Build prompt
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        full_text = prompt + output
        
        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Create labels (mask prompt tokens)
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_length = prompt_tokens["input_ids"].shape[1]
        
        labels = tokenized["input_ids"].clone()
        labels[0, :prompt_length] = -100  # Mask prompt
        labels[labels == self.tokenizer.pad_token_id] = -100  # Mask padding
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }


class LoRATrainer:
    """Trainer for LoRA adapters"""
    
    def __init__(
        self,
        base_model_name: str = "epfl-llm/meditron-70b",
        use_4bit: bool = True,
        use_8bit: bool = False,
        gpu_id: Optional[int] = None,
    ):
        self.base_model_name = base_model_name
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.gpu_id = gpu_id
        self.model = None
        self.tokenizer = None
    
    def load_base_model(self):
        """Load the base model with quantization"""
        logger.info(f"Loading base model: {self.base_model_name}")
        
        # Quantization config
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.use_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Device map: target specific GPU if set, otherwise auto-distribute
        if self.gpu_id is not None:
            device_map = {"":  self.gpu_id}
        else:
            device_map = "auto"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Prepare for training
        if bnb_config:
            self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info("Base model loaded successfully")
    
    def setup_lora(self, config: AdapterTrainingConfig):
        """Setup LoRA configuration"""
        logger.info(f"Setting up LoRA for {config.name}")
        
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                   f"({100 * trainable_params / total_params:.2f}%)")
    
    def load_data(self, adapter_name: str, config: AdapterTrainingConfig):
        """Load training and validation data"""
        # Check multiple data sources
        data_sources = [
            DATA_DIR / "train" / f"{adapter_name}_train.json",
            DATA_DIR / "processed" / adapter_name / f"{adapter_name}.json",
            DATA_DIR / "synthetic" / f"{adapter_name}_synthetic.json",
        ]
        
        train_examples = []
        for source in data_sources:
            if source.exists():
                with open(source) as f:
                    train_examples.extend(json.load(f))
                logger.info(f"Loaded data from {source}")
        
        if not train_examples:
            logger.warning(f"No training data found for {adapter_name}")
            return None, None
        
        # Split into train/val
        split_idx = int(len(train_examples) * 0.9)
        
        # Save temporary files for dataset
        train_path = DATA_DIR / "temp" / f"{adapter_name}_train_temp.json"
        val_path = DATA_DIR / "temp" / f"{adapter_name}_val_temp.json"
        train_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(train_path, 'w') as f:
            json.dump(train_examples[:split_idx], f)
        with open(val_path, 'w') as f:
            json.dump(train_examples[split_idx:], f)
        
        train_dataset = InstructionDataset(train_path, self.tokenizer, config.max_seq_length)
        val_dataset = InstructionDataset(val_path, self.tokenizer, config.max_seq_length)
        
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train(
        self,
        adapter_name: str,
        config: Optional[AdapterTrainingConfig] = None,
        resume_from: Optional[str] = None,
    ):
        """Train a LoRA adapter"""
        if config is None:
            config = ADAPTER_CONFIGS.get(adapter_name)
            if config is None:
                raise ValueError(f"Unknown adapter: {adapter_name}")
        
        logger.info("=" * 60)
        logger.info(f"Training adapter: {config.name}")
        logger.info(f"Description: {config.description}")
        logger.info("=" * 60)
        
        # Load base model if not loaded
        if self.model is None:
            self.load_base_model()
        
        # Setup LoRA
        self.setup_lora(config)
        
        # Load data
        train_dataset, val_dataset = self.load_data(adapter_name, config)
        
        if train_dataset is None or len(train_dataset) == 0:
            logger.error("No training data available")
            return None
        
        # Output directory
        output_dir = ADAPTERS_DIR / adapter_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            evaluation_strategy="steps" if val_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if val_dataset else False,
            report_to="none",
            fp16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            metric_for_best_model="eval_loss" if val_dataset else None,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Starting training...")
        
        if resume_from:
            trainer.train(resume_from_checkpoint=resume_from)
        else:
            trainer.train()
        
        # Save adapter
        logger.info(f"Saving adapter to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        config_path = output_dir / "adapter_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "name": config.name,
                "description": config.description,
                "base_model": self.base_model_name,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "training_epochs": config.num_epochs,
            }, f, indent=2)
        
        logger.info("Training complete!")
        return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapters for IMI")
    parser.add_argument(
        "--adapter",
        choices=list(ADAPTER_CONFIGS.keys()) + ["all"],
        required=True,
        help="Adapter to train",
    )
    parser.add_argument(
        "--base-model",
        default="epfl-llm/meditron-70b",
        help="Base model name or path",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization",
    )
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Use 8-bit quantization",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU ID for this training job (for parallel multi-GPU training)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Train all adapters in parallel across GPUs (requires --adapter all)",
    )
    parser.add_argument(
        "--resume-from",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate",
    )
    
    args = parser.parse_args()
    
    # Determine adapters to train
    if args.adapter == "all":
        adapters = list(ADAPTER_CONFIGS.keys())
    else:
        adapters = [args.adapter]
    
    # ─── Parallel multi-GPU training ───────────────────────────────
    if args.parallel and len(adapters) > 1:
        import subprocess, sys
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            logger.warning("Only 1 GPU detected — falling back to sequential training")
        else:
            logger.info(f"Launching parallel training across {num_gpus} GPUs for {len(adapters)} adapters")
            processes = []
            for idx, adapter_name in enumerate(adapters):
                gpu_id = idx % num_gpus
                cmd = [
                    sys.executable, __file__,
                    "--adapter", adapter_name,
                    "--base-model", args.base_model,
                    "--gpu", str(gpu_id),
                ]
                if args.use_8bit:
                    cmd.append("--use-8bit")
                if args.epochs:
                    cmd.extend(["--epochs", str(args.epochs)])
                if args.batch_size:
                    cmd.extend(["--batch-size", str(args.batch_size)])
                if args.learning_rate:
                    cmd.extend(["--learning-rate", str(args.learning_rate)])
                
                logger.info(f"  GPU {gpu_id} ← {adapter_name}")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                proc = subprocess.Popen(cmd, env=env)
                processes.append((adapter_name, proc))
            
            # Wait for all to finish
            failed = []
            for adapter_name, proc in processes:
                rc = proc.wait()
                if rc != 0:
                    failed.append(adapter_name)
                    logger.error(f"Training FAILED for {adapter_name} (exit code {rc})")
                else:
                    logger.info(f"Training COMPLETE for {adapter_name}")
            
            if failed:
                logger.error(f"Failed adapters: {failed}")
                raise SystemExit(1)
            
            logger.info("All parallel training jobs completed successfully!")
            return
    
    # ─── Single-GPU / sequential training ──────────────────────────
    trainer = LoRATrainer(
        base_model_name=args.base_model,
        use_4bit=args.use_4bit and not args.use_8bit,
        use_8bit=args.use_8bit,
        gpu_id=args.gpu,
    )
    
    for adapter_name in adapters:
        config = ADAPTER_CONFIGS[adapter_name]
        
        # Apply overrides
        if args.epochs:
            config.num_epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.learning_rate:
            config.learning_rate = args.learning_rate
        
        trainer.train(
            adapter_name=adapter_name,
            config=config,
            resume_from=args.resume_from,
        )
        
        # Reset LoRA weights but keep base model loaded to avoid
        # reloading the full 70B model from scratch for each adapter
        if hasattr(trainer, 'model') and trainer.model is not None:
            try:
                from peft import PeftModel
                if isinstance(trainer.model, PeftModel):
                    trainer.model = trainer.model.get_base_model()
                else:
                    trainer.model = None
            except Exception:
                trainer.model = None


if __name__ == "__main__":
    main()
