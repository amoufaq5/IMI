"""
UMI Advanced Distributed Fine-Tuning Pipeline
Optimized for 4xH100 GPUs with DeepSpeed/FSDP support
Full fine-tune on Mistral-7B with 8192 sequence length
"""

import os
import json
import math
import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    get_scheduler,
    set_seed,
)
from transformers.integrations import deepspeed
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class UMIDistributedConfig:
    """
    Configuration for distributed fine-tuning on 4xH100 GPUs.
    Optimized for maximum throughput and memory efficiency.
    """
    
    # Model Configuration
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    output_dir: str = "models/umi-medical-v2"
    model_revision: str = "main"
    trust_remote_code: bool = True
    
    # Sequence Length - 8192 for long medical documents
    max_seq_length: int = 8192
    
    # Training Mode: "full" or "qlora"
    training_mode: str = "full"  # full fine-tune for 7B on 4xH100
    
    # Data Configuration
    train_data_dir: str = "data/training"
    knowledge_base_dir: str = "data/knowledge_base"
    max_train_samples: Optional[int] = None  # None = use all data
    preprocessing_num_workers: int = 16
    
    # Full Fine-Tune Settings (for 4xH100)
    # With 4xH100 (320GB VRAM), we can do full fine-tune on 7B
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch = 4*4*4 = 64
    
    # Learning Rate & Schedule
    learning_rate: float = 2e-5  # Lower for full fine-tune
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 = use epochs
    
    # Optimizer Settings
    optim: str = "adamw_torch_fused"  # Fused AdamW for H100
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Precision & Memory
    bf16: bool = True  # H100 excels at BF16
    fp16: bool = False
    tf32: bool = True  # Enable TF32 for H100
    
    # Gradient Checkpointing - saves memory
    gradient_checkpointing: bool = True
    
    # DeepSpeed Configuration
    use_deepspeed: bool = True
    deepspeed_stage: int = 2  # ZeRO Stage 2 for full fine-tune
    deepspeed_offload: bool = False  # No offload needed with 4xH100
    
    # FSDP Configuration (alternative to DeepSpeed)
    use_fsdp: bool = False
    fsdp_sharding_strategy: str = "FULL_SHARD"
    
    # Flash Attention 2 - critical for 8192 seq length
    use_flash_attention_2: bool = True
    
    # Logging & Checkpointing
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Fault Tolerance & Recovery
    save_on_each_node: bool = False
    resume_from_checkpoint: Optional[str] = None
    ignore_data_skip: bool = False
    
    # Reporting
    report_to: List[str] = field(default_factory=lambda: ["tensorboard", "wandb"])
    run_name: Optional[str] = None
    
    # QLoRA Settings (if training_mode == "qlora")
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    
    # Seed
    seed: int = 42
    
    def __post_init__(self):
        if self.run_name is None:
            self.run_name = f"umi-{self.training_mode}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def create_deepspeed_config(config: UMIDistributedConfig) -> Dict[str, Any]:
    """
    Create DeepSpeed configuration optimized for 4xH100.
    """
    ds_config = {
        "train_micro_batch_size_per_gpu": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_clipping": config.max_grad_norm,
        
        "bf16": {
            "enabled": config.bf16,
        },
        
        "zero_optimization": {
            "stage": config.deepspeed_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.learning_rate,
                "betas": [config.adam_beta1, config.adam_beta2],
                "eps": config.adam_epsilon,
                "weight_decay": config.weight_decay,
            }
        },
        
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": config.learning_rate,
                "warmup_num_steps": "auto",
                "total_num_steps": "auto",
            }
        },
        
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": True,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
        },
        
        "wall_clock_breakdown": False,
        "steps_per_print": config.logging_steps,
    }
    
    # ZeRO Stage 3 settings
    if config.deepspeed_stage == 3:
        ds_config["zero_optimization"].update({
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        })
        
        if config.deepspeed_offload:
            ds_config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }
            ds_config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": True,
            }
    
    return ds_config


class MedicalDataProcessor:
    """
    Process and combine medical data from multiple sources for training.
    """
    
    SYSTEM_PROMPT = """You are UMI (Universal Medical Intelligence), an advanced medical AI assistant trained on comprehensive medical knowledge. You provide accurate, evidence-based medical information while following best practices for patient safety.

Key principles:
1. Patient safety is the top priority - identify danger signs requiring immediate attention
2. Provide clear, evidence-based information from medical literature
3. Recommend professional consultation for diagnosis and treatment decisions
4. Be empathetic, clear, and thorough in explanations
5. Cite sources when possible (drug databases, clinical guidelines, research)

You are a medical information assistant, NOT a replacement for professional medical care."""

    def __init__(self, tokenizer, max_seq_length: int = 8192):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def format_chat_message(self, messages: List[Dict[str, str]]) -> str:
        """Format messages using the model's chat template."""
        full_messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + messages
        
        try:
            formatted = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback formatting
            formatted = f"<s>[INST] {self.SYSTEM_PROMPT}\n\n"
            for msg in messages:
                if msg["role"] == "user":
                    formatted += f"{msg['content']} [/INST] "
                else:
                    formatted += f"{msg['content']}</s><s>[INST] "
            formatted = formatted.rstrip("<s>[INST] ")
        
        return formatted
    
    def format_qa_pair(self, question: str, answer: str, source: str = "") -> str:
        """Format a QA pair for training."""
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return self.format_chat_message(messages)
    
    def load_all_training_data(self, config: UMIDistributedConfig) -> Dataset:
        """Load and combine all training data sources."""
        all_texts = []
        
        # Load from training directory
        train_dir = Path(config.train_data_dir)
        if train_dir.exists():
            for jsonl_file in train_dir.glob("**/*.jsonl"):
                logger.info(f"Loading training data from: {jsonl_file}")
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                item = json.loads(line)
                                text = self._process_training_item(item)
                                if text and len(text) > 100:
                                    all_texts.append({"text": text})
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    logger.warning(f"Error loading {jsonl_file}: {e}")
        
        # Load from knowledge base (converted to QA format)
        kb_dir = Path(config.knowledge_base_dir)
        if kb_dir.exists():
            for source_dir in kb_dir.iterdir():
                if source_dir.is_dir():
                    for jsonl_file in source_dir.glob("*.jsonl"):
                        logger.info(f"Loading knowledge base: {jsonl_file}")
                        try:
                            texts = self._load_knowledge_base_file(jsonl_file)
                            all_texts.extend(texts)
                        except Exception as e:
                            logger.warning(f"Error loading {jsonl_file}: {e}")
        
        logger.info(f"Total training samples: {len(all_texts)}")
        
        if not all_texts:
            raise ValueError("No training data found!")
        
        # Create dataset
        dataset = Dataset.from_list(all_texts)
        
        # Limit samples if specified
        if config.max_train_samples and len(dataset) > config.max_train_samples:
            dataset = dataset.shuffle(seed=config.seed).select(range(config.max_train_samples))
        
        return dataset
    
    def _process_training_item(self, item: Dict[str, Any]) -> Optional[str]:
        """Process a single training item."""
        # Chat format
        if "messages" in item:
            messages = item["messages"]
            if isinstance(messages, list) and len(messages) >= 2:
                return self.format_chat_message(messages)
        
        # Instruction format
        if "instruction" in item:
            question = item["instruction"]
            if item.get("input"):
                question = f"{question}\n\nContext: {item['input']}"
            answer = item.get("output", item.get("response", ""))
            if answer:
                return self.format_qa_pair(question, answer)
        
        # QA format
        if "question" in item and "answer" in item:
            return self.format_qa_pair(item["question"], item["answer"])
        
        # Raw text
        if "text" in item:
            return item["text"]
        
        return None
    
    def _load_knowledge_base_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Load knowledge base file and convert to training format."""
        texts = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    title = item.get("title", "")
                    content = item.get("content", "")
                    
                    if not title or not content or len(content) < 100:
                        continue
                    
                    # Create QA pair from knowledge base entry
                    question = f"What can you tell me about {title}?"
                    answer = content[:4000]  # Limit answer length
                    
                    text = self.format_qa_pair(question, answer)
                    texts.append({"text": text})
                    
                except json.JSONDecodeError:
                    continue
        
        return texts


class UMIDistributedTrainer:
    """
    Distributed fine-tuning trainer for 4xH100 GPUs.
    Supports both DeepSpeed and FSDP.
    """
    
    def __init__(self, config: UMIDistributedConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Enable TF32 for H100
        if config.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def setup_model(self) -> None:
        """Load and configure the model for distributed training."""
        logger.info(f"Loading model: {self.config.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=self.config.trust_remote_code,
            model_max_length=self.config.max_seq_length,
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Model configuration
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": torch.bfloat16 if self.config.bf16 else torch.float16,
            "use_cache": False,  # Disable for training
        }
        
        # Enable Flash Attention 2
        if self.config.use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 enabled")
        
        # QLoRA quantization
        if self.config.training_mode == "qlora" and self.config.use_4bit:
            from transformers import BitsAndBytesConfig
            
            compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
            )
            logger.info("4-bit quantization enabled for QLoRA")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            **model_kwargs,
        )
        
        # Prepare for QLoRA if needed
        if self.config.training_mode == "qlora":
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.gradient_checkpointing,
            )
            self._setup_lora()
        
        # Enable gradient checkpointing for full fine-tune
        elif self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    def _setup_lora(self) -> None:
        """Configure LoRA adapters for QLoRA training."""
        logger.info("Setting up LoRA adapters...")
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def setup_training(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None) -> None:
        """Configure the trainer with DeepSpeed or FSDP."""
        logger.info("Setting up distributed training...")
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # DeepSpeed config
        deepspeed_config = None
        if self.config.use_deepspeed:
            deepspeed_config = create_deepspeed_config(self.config)
            ds_config_path = output_dir / "deepspeed_config.json"
            with open(ds_config_path, 'w') as f:
                json.dump(deepspeed_config, f, indent=2)
            logger.info(f"DeepSpeed config saved to: {ds_config_path}")
        
        # FSDP config
        fsdp_config = None
        if self.config.use_fsdp and not self.config.use_deepspeed:
            fsdp_config = {
                "fsdp_transformer_layer_cls_to_wrap": ["MistralDecoderLayer"],
                "fsdp_backward_prefetch": "backward_pre",
                "fsdp_forward_prefetch": True,
                "fsdp_use_orig_params": True,
                "fsdp_cpu_ram_efficient_loading": True,
                "fsdp_sync_module_states": True,
            }
        
        # Training arguments
        training_args = SFTConfig(
            output_dir=str(output_dir),
            run_name=self.config.run_name,
            
            # Batch size & accumulation
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            # Training duration
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps,
            
            # Learning rate
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            
            # Optimizer
            optim=self.config.optim,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            adam_epsilon=self.config.adam_epsilon,
            max_grad_norm=self.config.max_grad_norm,
            
            # Precision
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            tf32=self.config.tf32,
            
            # Gradient checkpointing
            gradient_checkpointing=self.config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            
            # Logging & saving
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            evaluation_strategy=self.config.evaluation_strategy if eval_dataset else "no",
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end if eval_dataset else False,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            
            # Distributed training
            deepspeed=deepspeed_config if self.config.use_deepspeed else None,
            fsdp=self.config.fsdp_sharding_strategy if fsdp_config else None,
            fsdp_config=fsdp_config,
            
            # Reporting
            report_to=self.config.report_to,
            
            # Misc
            seed=self.config.seed,
            data_seed=self.config.seed,
            dataloader_num_workers=self.config.preprocessing_num_workers,
            dataloader_pin_memory=True,
            remove_unused_columns=True,
            
            # SFT specific
            max_seq_length=self.config.max_seq_length,
            packing=True,  # Pack sequences for efficiency
            dataset_text_field="text",
        )
        
        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        logger.info("Trainer setup complete")
    
    def train(self) -> None:
        """Run the training loop with fault tolerance."""
        logger.info("=" * 70)
        logger.info("STARTING DISTRIBUTED TRAINING")
        logger.info(f"Mode: {self.config.training_mode.upper()}")
        logger.info(f"GPUs: {torch.cuda.device_count()}")
        logger.info(f"Sequence Length: {self.config.max_seq_length}")
        logger.info(f"Effective Batch Size: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps * max(1, torch.cuda.device_count())}")
        logger.info("=" * 70)
        
        # Resume from checkpoint if specified
        resume_from = self.config.resume_from_checkpoint
        if resume_from is None:
            # Check for latest checkpoint
            checkpoints = list(Path(self.config.output_dir).glob("checkpoint-*"))
            if checkpoints:
                latest = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
                resume_from = str(latest)
                logger.info(f"Resuming from checkpoint: {resume_from}")
        
        try:
            # Train
            train_result = self.trainer.train(resume_from_checkpoint=resume_from)
            
            # Save final model
            logger.info("Saving final model...")
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            # Save training metrics
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            self.trainer.save_state()
            
            # Save config
            config_path = Path(self.config.output_dir) / "training_config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2, default=str)
            
            logger.info(f"Training complete! Model saved to: {self.config.output_dir}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Save checkpoint on failure
            try:
                emergency_path = Path(self.config.output_dir) / "emergency_checkpoint"
                self.trainer.save_model(str(emergency_path))
                logger.info(f"Emergency checkpoint saved to: {emergency_path}")
            except:
                pass
            raise
    
    def run(self) -> None:
        """Execute the full training pipeline."""
        # Setup model
        self.setup_model()
        
        # Load data
        data_processor = MedicalDataProcessor(self.tokenizer, self.config.max_seq_length)
        train_dataset = data_processor.load_all_training_data(self.config)
        
        # Split for evaluation (5%)
        split = train_dataset.train_test_split(test_size=0.05, seed=self.config.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Eval samples: {len(eval_dataset)}")
        
        # Setup trainer
        self.setup_training(train_dataset, eval_dataset)
        
        # Train
        self.train()


def merge_lora_weights(
    base_model: str,
    lora_model: str,
    output_dir: str,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
) -> None:
    """Merge LoRA weights with base model for deployment."""
    from peft import PeftModel
    
    logger.info("Merging LoRA weights...")
    logger.info(f"Base model: {base_model}")
    logger.info(f"LoRA model: {lora_model}")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Load and merge LoRA
    model = PeftModel.from_pretrained(model, lora_model)
    model = model.merge_and_unload()
    
    # Save
    logger.info(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Push to hub
    if push_to_hub and hub_model_id:
        logger.info(f"Pushing to Hub: {hub_model_id}")
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
    
    logger.info("Merge complete!")


def main():
    """Main entry point for distributed training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="UMI Distributed Fine-Tuning")
    
    # Model
    parser.add_argument("--base-model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--output-dir", type=str, default="models/umi-medical-v2")
    
    # Training mode
    parser.add_argument("--mode", type=str, choices=["full", "qlora"], default="full")
    
    # Data
    parser.add_argument("--train-data-dir", type=str, default="data/training")
    parser.add_argument("--knowledge-base-dir", type=str, default="data/knowledge_base")
    parser.add_argument("--max-samples", type=int, default=None)
    
    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    
    # Distributed
    parser.add_argument("--deepspeed", action="store_true", default=True)
    parser.add_argument("--deepspeed-stage", type=int, default=2)
    parser.add_argument("--fsdp", action="store_true", default=False)
    parser.add_argument("--no-flash-attn", action="store_true", default=False)
    
    # Resume
    parser.add_argument("--resume", type=str, default=None)
    
    # Merge
    parser.add_argument("--merge", action="store_true", help="Merge LoRA weights after training")
    
    # Config file
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = UMIDistributedConfig(**config_dict)
    else:
        config = UMIDistributedConfig(
            base_model=args.base_model,
            output_dir=args.output_dir,
            training_mode=args.mode,
            train_data_dir=args.train_data_dir,
            knowledge_base_dir=args.knowledge_base_dir,
            max_train_samples=args.max_samples,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            max_seq_length=args.max_seq_length,
            use_deepspeed=args.deepspeed and not args.fsdp,
            deepspeed_stage=args.deepspeed_stage,
            use_fsdp=args.fsdp,
            use_flash_attention_2=not args.no_flash_attn,
            resume_from_checkpoint=args.resume,
        )
    
    # Print config
    logger.info("=" * 70)
    logger.info("UMI DISTRIBUTED FINE-TUNING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Base Model: {config.base_model}")
    logger.info(f"Training Mode: {config.training_mode}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Sequence Length: {config.max_seq_length}")
    logger.info(f"Epochs: {config.num_train_epochs}")
    logger.info(f"Batch Size (per device): {config.per_device_train_batch_size}")
    logger.info(f"Gradient Accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"Learning Rate: {config.learning_rate}")
    logger.info(f"DeepSpeed: {config.use_deepspeed} (Stage {config.deepspeed_stage})")
    logger.info(f"FSDP: {config.use_fsdp}")
    logger.info(f"Flash Attention 2: {config.use_flash_attention_2}")
    logger.info("=" * 70)
    
    # Run training
    trainer = UMIDistributedTrainer(config)
    trainer.run()
    
    # Merge if QLoRA
    if args.merge and config.training_mode == "qlora":
        merged_dir = f"{config.output_dir}-merged"
        merge_lora_weights(
            base_model=config.base_model,
            lora_model=config.output_dir,
            output_dir=merged_dir,
        )


if __name__ == "__main__":
    main()
