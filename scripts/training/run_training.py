"""
IMI Master Training Pipeline
Complete pipeline: Data Ingestion → Processing → SFT → DPO/ORPO
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "data_ingestion"))


def run_ingestion(args):
    """Run data ingestion."""
    print("\n" + "=" * 70)
    print("STAGE 1: DATA INGESTION")
    print("=" * 70)
    
    from scrape_all import run_all_scrapers
    
    asyncio.run(run_all_scrapers(
        output_base=args.data_dir + "/raw",
        pubmed=not args.skip_pubmed,
        datasets=not args.skip_datasets,
        pubmed_articles_per_topic=args.pubmed_articles,
        max_samples_per_dataset=args.max_samples,
    ))


def run_processing(args):
    """Run data processing."""
    print("\n" + "=" * 70)
    print("STAGE 2: DATA PROCESSING")
    print("=" * 70)
    
    from data_processor import DataProcessor
    
    processor = DataProcessor(
        raw_data_dir=args.data_dir + "/raw",
        output_dir=args.data_dir + "/processed",
    )
    
    processor.process_all(max_samples_per_source=args.max_samples)
    processor.save(format="chat")


def run_sft(args):
    """Run SFT training."""
    print("\n" + "=" * 70)
    print("STAGE 3: SUPERVISED FINE-TUNING (SFT)")
    print("=" * 70)
    
    from sft_trainer import IMISFTTrainer, IMITrainingConfig
    
    config = IMITrainingConfig(
        model_name=args.model,
        output_dir=args.output_dir + "/sft",
        training_mode=args.mode,
        train_data_path=args.data_dir + "/processed/sft_train.jsonl",
        eval_data_path=args.data_dir + "/processed/sft_val.jsonl",
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lora_r=args.lora_r,
        report_to="wandb" if args.wandb else "none",
    )
    
    trainer = IMISFTTrainer(config)
    trainer.run()


def run_dpo(args):
    """Run DPO training."""
    print("\n" + "=" * 70)
    print("STAGE 4: DIRECT PREFERENCE OPTIMIZATION (DPO)")
    print("=" * 70)
    
    from dpo_trainer import IMIDPOTrainer, IMIDPOConfig
    
    config = IMIDPOConfig(
        model_name=args.output_dir + "/sft",  # Use SFT model
        output_dir=args.output_dir + "/dpo",
        train_data_path=args.data_dir + "/processed/dpo_train.jsonl",
        beta=args.dpo_beta,
        learning_rate=args.dpo_lr,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        report_to="wandb" if args.wandb else "none",
    )
    
    trainer = IMIDPOTrainer(config)
    trainer.run()


def run_orpo(args):
    """Run ORPO training (alternative to SFT+DPO)."""
    print("\n" + "=" * 70)
    print("STAGE 3-4: ORPO (Combined SFT + Preference)")
    print("=" * 70)
    
    from orpo_trainer import IMIORPOTrainer, IMIORPOConfig
    
    config = IMIORPOConfig(
        model_name=args.model,
        output_dir=args.output_dir + "/orpo",
        train_data_path=args.data_dir + "/processed/dpo_train.jsonl",
        beta=args.dpo_beta,
        learning_rate=args.orpo_lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lora_r=args.lora_r,
        report_to="wandb" if args.wandb else "none",
    )
    
    trainer = IMIORPOTrainer(config)
    trainer.run()


def main():
    parser = argparse.ArgumentParser(
        description="IMI Medical LLM Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with defaults
  python run_training.py --all
  
  # Only data ingestion and processing
  python run_training.py --ingest --process
  
  # Only SFT training (data already processed)
  python run_training.py --sft --model mistralai/Mistral-7B-Instruct-v0.3
  
  # SFT + DPO pipeline
  python run_training.py --sft --dpo
  
  # ORPO (combined SFT + preference, more efficient)
  python run_training.py --orpo
  
  # Full pipeline with custom settings
  python run_training.py --all --model meta-llama/Llama-3-8B-Instruct --epochs 5 --lr 1e-4
        """
    )
    
    # Pipeline stages
    stage_group = parser.add_argument_group("Pipeline Stages")
    stage_group.add_argument("--all", action="store_true", help="Run full pipeline (ingest → process → sft → dpo)")
    stage_group.add_argument("--ingest", action="store_true", help="Run data ingestion")
    stage_group.add_argument("--process", action="store_true", help="Run data processing")
    stage_group.add_argument("--sft", action="store_true", help="Run SFT training")
    stage_group.add_argument("--dpo", action="store_true", help="Run DPO training (after SFT)")
    stage_group.add_argument("--orpo", action="store_true", help="Run ORPO training (alternative to SFT+DPO)")
    
    # Data settings
    data_group = parser.add_argument_group("Data Settings")
    data_group.add_argument("--data-dir", default="data", help="Data directory")
    data_group.add_argument("--skip-pubmed", action="store_true", help="Skip PubMed scraping")
    data_group.add_argument("--skip-datasets", action="store_true", help="Skip HuggingFace datasets")
    data_group.add_argument("--pubmed-articles", type=int, default=200, help="PubMed articles per topic")
    data_group.add_argument("--max-samples", type=int, help="Max samples per data source")
    
    # Model settings
    model_group = parser.add_argument_group("Model Settings")
    model_group.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3", help="Base model")
    model_group.add_argument("--output-dir", default="outputs/imi-medical", help="Output directory")
    model_group.add_argument("--mode", choices=["full", "lora", "qlora"], default="qlora", help="Training mode")
    
    # Training settings
    train_group = parser.add_argument_group("Training Settings")
    train_group.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length")
    train_group.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    train_group.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    train_group.add_argument("--lr", type=float, default=2e-4, help="SFT learning rate")
    train_group.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_group.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    
    # RL settings
    rl_group = parser.add_argument_group("RL Settings (DPO/ORPO)")
    rl_group.add_argument("--dpo-beta", type=float, default=0.1, help="DPO/ORPO beta")
    rl_group.add_argument("--dpo-lr", type=float, default=5e-5, help="DPO learning rate")
    rl_group.add_argument("--orpo-lr", type=float, default=8e-6, help="ORPO learning rate")
    
    # Misc
    misc_group = parser.add_argument_group("Misc")
    misc_group.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    misc_group.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Default to --all if no stage specified
    if not any([args.all, args.ingest, args.process, args.sft, args.dpo, args.orpo]):
        print("No stage specified. Use --help for options.")
        print("Running --all by default...")
        args.all = True
    
    # Expand --all
    if args.all:
        args.ingest = True
        args.process = True
        args.sft = True
        args.dpo = True
    
    # Print configuration
    print("=" * 70)
    print("IMI MEDICAL LLM TRAINING PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"\nStages to run:")
    print(f"  - Data Ingestion: {args.ingest}")
    print(f"  - Data Processing: {args.process}")
    print(f"  - SFT Training: {args.sft}")
    print(f"  - DPO Training: {args.dpo}")
    print(f"  - ORPO Training: {args.orpo}")
    print(f"\nModel: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"Mode: {args.mode}")
    
    # Run stages
    try:
        if args.ingest:
            run_ingestion(args)
        
        if args.process:
            run_processing(args)
        
        if args.orpo:
            # ORPO is alternative to SFT+DPO
            run_orpo(args)
        else:
            if args.sft:
                run_sft(args)
            
            if args.dpo:
                run_dpo(args)
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"Finished: {datetime.now().isoformat()}")
        print(f"Output directory: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        raise


if __name__ == "__main__":
    main()
