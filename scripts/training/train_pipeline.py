#!/usr/bin/env python3
"""
UMI Master Training Pipeline
Continuous training with auto-recovery, data ingestion integration, and fault tolerance.

Features:
- Automatic data ingestion before training
- Checkpoint recovery on failure
- Continuous training mode (ingest -> train -> repeat)
- Multi-GPU support (DeepSpeed/FSDP)
- Experiment tracking with W&B/TensorBoard
- Health monitoring and alerts
"""

import os
import sys
import json
import time
import signal
import asyncio
import subprocess
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class PipelineState(str, Enum):
    IDLE = "idle"
    INGESTING = "ingesting"
    CONVERTING = "converting"
    TRAINING = "training"
    EVALUATING = "evaluating"
    CHECKPOINTING = "checkpointing"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass
class PipelineConfig:
    """Configuration for the training pipeline."""
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    checkpoint_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "checkpoints")
    
    # Model
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    max_seq_length: int = 8192
    
    # Training
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    
    # Distributed
    num_gpus: int = 4
    use_deepspeed: bool = True
    deepspeed_stage: int = 2
    
    # Pipeline
    run_ingestion: bool = True
    continuous_mode: bool = False
    ingestion_interval_hours: int = 24
    max_retries: int = 3
    retry_delay_seconds: int = 60
    
    # Data sources to ingest
    data_sources: List[str] = field(default_factory=lambda: [
        "pubmed", "drugs", "trials", "rxnorm", "who",
        "kaggle", "medlineplus", "opentargets", "umls",
        "snomed", "orphanet", "disgenet", "chembl", "uniprot"
    ])
    
    # API Keys
    kaggle_api_key: Optional[str] = None
    umls_api_key: Optional[str] = None
    disgenet_api_key: Optional[str] = None
    
    def __post_init__(self):
        self.kaggle_api_key = self.kaggle_api_key or os.environ.get("KAGGLE_KEY")
        self.umls_api_key = self.umls_api_key or os.environ.get("UMLS_API_KEY")
        self.disgenet_api_key = self.disgenet_api_key or os.environ.get("DISGENET_API_KEY")


@dataclass
class PipelineStatus:
    """Current pipeline status."""
    state: PipelineState = PipelineState.IDLE
    current_step: str = ""
    progress: float = 0.0
    start_time: Optional[datetime] = None
    last_checkpoint: Optional[str] = None
    last_ingestion: Optional[datetime] = None
    total_samples_ingested: int = 0
    total_samples_trained: int = 0
    current_epoch: int = 0
    current_loss: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        d["start_time"] = self.start_time.isoformat() if self.start_time else None
        d["last_ingestion"] = self.last_ingestion.isoformat() if self.last_ingestion else None
        return d


class TrainingPipeline:
    """
    Master training pipeline with auto-recovery.
    Orchestrates data ingestion, conversion, and distributed training.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.status = PipelineStatus()
        self._shutdown_requested = False
        self._setup_signal_handlers()
        self._setup_directories()
        self._load_state()
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n[PIPELINE] Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True
    
    def _setup_directories(self):
        """Create necessary directories."""
        for d in [self.config.data_dir, self.config.output_dir, self.config.checkpoint_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        (self.config.data_dir / "knowledge_base").mkdir(exist_ok=True)
        (self.config.data_dir / "training").mkdir(exist_ok=True)
    
    def _state_file(self) -> Path:
        return self.config.checkpoint_dir / "pipeline_state.json"
    
    def _load_state(self):
        """Load pipeline state from checkpoint."""
        state_file = self._state_file()
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                self.status.last_checkpoint = data.get("last_checkpoint")
                self.status.last_ingestion = datetime.fromisoformat(data["last_ingestion"]) if data.get("last_ingestion") else None
                self.status.total_samples_ingested = data.get("total_samples_ingested", 0)
                self.status.total_samples_trained = data.get("total_samples_trained", 0)
                print(f"[PIPELINE] Loaded state from {state_file}")
            except Exception as e:
                print(f"[PIPELINE] Warning: Could not load state: {e}")
    
    def _save_state(self):
        """Save pipeline state to checkpoint."""
        state_file = self._state_file()
        try:
            with open(state_file, "w") as f:
                json.dump(self.status.to_dict(), f, indent=2)
        except Exception as e:
            print(f"[PIPELINE] Warning: Could not save state: {e}")
    
    def _log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def _should_run_ingestion(self) -> bool:
        """Check if data ingestion should run."""
        if not self.config.run_ingestion:
            return False
        
        if self.status.last_ingestion is None:
            return True
        
        elapsed = datetime.now() - self.status.last_ingestion
        return elapsed > timedelta(hours=self.config.ingestion_interval_hours)
    
    async def run_ingestion(self) -> bool:
        """Run data ingestion pipeline."""
        self.status.state = PipelineState.INGESTING
        self.status.current_step = "Data Ingestion"
        self._log("Starting data ingestion...")
        
        try:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "data_ingestion" / "ingest_all.py"),
                "--output", str(self.config.data_dir / "knowledge_base"),
            ]
            
            if self.config.data_sources:
                cmd.extend(["--only"] + self.config.data_sources)
            
            if self.config.kaggle_api_key:
                cmd.extend(["--kaggle-key", self.config.kaggle_api_key])
            if self.config.umls_api_key:
                cmd.extend(["--umls-key", self.config.umls_api_key])
            if self.config.disgenet_api_key:
                cmd.extend(["--disgenet-key", self.config.disgenet_api_key])
            
            self._log(f"Running: {' '.join(cmd[:5])}...")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
            
            while True:
                if self._shutdown_requested:
                    process.terminate()
                    return False
                
                line = await process.stdout.readline()
                if not line:
                    break
                print(f"  [INGEST] {line.decode().strip()}")
            
            await process.wait()
            
            if process.returncode == 0:
                self.status.last_ingestion = datetime.now()
                self._log("Data ingestion completed successfully")
                return True
            else:
                self._log(f"Data ingestion failed with code {process.returncode}", "ERROR")
                return False
                
        except Exception as e:
            self._log(f"Data ingestion error: {e}", "ERROR")
            self.status.errors.append({
                "stage": "ingestion",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            return False
    
    async def run_conversion(self) -> bool:
        """Convert ingested data to training format."""
        self.status.state = PipelineState.CONVERTING
        self.status.current_step = "Data Conversion"
        self._log("Converting data to training format...")
        
        try:
            converter_script = PROJECT_ROOT / "scripts" / "training" / "convert_ingested_data.py"
            
            if not converter_script.exists():
                self._log("Converter script not found, skipping conversion", "WARNING")
                return True
            
            cmd = [
                sys.executable,
                str(converter_script),
                "--input", str(self.config.data_dir / "knowledge_base"),
                "--output", str(self.config.data_dir / "training"),
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
            
            while True:
                if self._shutdown_requested:
                    process.terminate()
                    return False
                
                line = await process.stdout.readline()
                if not line:
                    break
                print(f"  [CONVERT] {line.decode().strip()}")
            
            await process.wait()
            
            if process.returncode == 0:
                self._log("Data conversion completed successfully")
                return True
            else:
                self._log(f"Data conversion failed with code {process.returncode}", "ERROR")
                return False
                
        except Exception as e:
            self._log(f"Data conversion error: {e}", "ERROR")
            return False
    
    async def run_training(self) -> bool:
        """Run distributed training."""
        self.status.state = PipelineState.TRAINING
        self.status.current_step = "Model Training"
        self._log(f"Starting training on {self.config.num_gpus} GPUs...")
        
        try:
            training_script = PROJECT_ROOT / "scripts" / "training" / "fine_tune_distributed.py"
            
            if not training_script.exists():
                self._log("Training script not found!", "ERROR")
                return False
            
            # Build command
            if self.config.use_deepspeed:
                cmd = [
                    "deepspeed",
                    "--num_gpus", str(self.config.num_gpus),
                    str(training_script),
                    "--deepspeed",
                ]
            else:
                cmd = [
                    "torchrun",
                    f"--nproc_per_node={self.config.num_gpus}",
                    str(training_script),
                    "--fsdp",
                ]
            
            # Add training arguments
            cmd.extend([
                "--model_name", self.config.base_model,
                "--output_dir", str(self.config.output_dir / "model"),
                "--max_seq_length", str(self.config.max_seq_length),
                "--num_epochs", str(self.config.num_epochs),
                "--batch_size", str(self.config.batch_size),
                "--gradient_accumulation_steps", str(self.config.gradient_accumulation_steps),
                "--learning_rate", str(self.config.learning_rate),
            ])
            
            # Resume from checkpoint if available
            if self.status.last_checkpoint:
                checkpoint_path = Path(self.status.last_checkpoint)
                if checkpoint_path.exists():
                    cmd.extend(["--resume_from_checkpoint", str(checkpoint_path)])
                    self._log(f"Resuming from checkpoint: {checkpoint_path}")
            
            self._log(f"Running: {cmd[0]} ... (full command logged)")
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(self.config.num_gpus))
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
            
            while True:
                if self._shutdown_requested:
                    self._log("Shutdown requested, saving checkpoint...")
                    process.send_signal(signal.SIGINT)
                    await asyncio.sleep(30)
                    if process.returncode is None:
                        process.terminate()
                    return False
                
                line = await process.stdout.readline()
                if not line:
                    break
                
                line_str = line.decode().strip()
                print(f"  [TRAIN] {line_str}")
                
                # Parse training progress
                if "loss" in line_str.lower():
                    try:
                        if "loss:" in line_str:
                            loss_str = line_str.split("loss:")[1].split()[0]
                            self.status.current_loss = float(loss_str.strip(","))
                    except:
                        pass
                
                if "epoch" in line_str.lower():
                    try:
                        if "epoch:" in line_str:
                            epoch_str = line_str.split("epoch:")[1].split()[0]
                            self.status.current_epoch = int(float(epoch_str.strip(",")))
                    except:
                        pass
                
                if "checkpoint" in line_str.lower() and "saved" in line_str.lower():
                    # Extract checkpoint path
                    self.status.last_checkpoint = str(self.config.output_dir / "model" / "checkpoint-latest")
                    self._save_state()
            
            await process.wait()
            
            if process.returncode == 0:
                self._log("Training completed successfully!")
                self.status.last_checkpoint = str(self.config.output_dir / "model")
                return True
            else:
                self._log(f"Training failed with code {process.returncode}", "ERROR")
                return False
                
        except Exception as e:
            self._log(f"Training error: {e}", "ERROR")
            traceback.print_exc()
            self.status.errors.append({
                "stage": "training",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            return False
    
    async def run_evaluation(self) -> bool:
        """Run model evaluation."""
        self.status.state = PipelineState.EVALUATING
        self.status.current_step = "Model Evaluation"
        self._log("Running model evaluation...")
        
        try:
            eval_script = PROJECT_ROOT / "scripts" / "training" / "evaluate.py"
            
            if not eval_script.exists():
                self._log("Evaluation script not found, skipping", "WARNING")
                return True
            
            cmd = [
                sys.executable,
                str(eval_script),
                "--model_path", str(self.config.output_dir / "model"),
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
            
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                print(f"  [EVAL] {line.decode().strip()}")
            
            await process.wait()
            return process.returncode == 0
            
        except Exception as e:
            self._log(f"Evaluation error: {e}", "ERROR")
            return False
    
    async def run_once(self) -> bool:
        """Run the pipeline once (ingest -> convert -> train -> evaluate)."""
        self.status.start_time = datetime.now()
        self.status.errors = []
        
        self._log("=" * 60)
        self._log("UMI TRAINING PIPELINE - Starting")
        self._log("=" * 60)
        
        try:
            # Step 1: Data Ingestion (if needed)
            if self._should_run_ingestion():
                success = await self.run_ingestion()
                if not success and not self._shutdown_requested:
                    self._log("Ingestion failed, continuing with existing data...", "WARNING")
                
                if self._shutdown_requested:
                    return False
                
                # Step 2: Data Conversion
                success = await self.run_conversion()
                if not success and not self._shutdown_requested:
                    self._log("Conversion failed, continuing with existing data...", "WARNING")
            else:
                self._log("Skipping ingestion (recent data available)")
            
            if self._shutdown_requested:
                return False
            
            # Step 3: Training
            for attempt in range(self.config.max_retries):
                if self._shutdown_requested:
                    return False
                
                success = await self.run_training()
                
                if success:
                    break
                elif attempt < self.config.max_retries - 1:
                    self._log(f"Training failed, retrying in {self.config.retry_delay_seconds}s (attempt {attempt + 2}/{self.config.max_retries})...")
                    await asyncio.sleep(self.config.retry_delay_seconds)
            
            if not success:
                self.status.state = PipelineState.FAILED
                self._log("Training failed after all retries", "ERROR")
                return False
            
            if self._shutdown_requested:
                return False
            
            # Step 4: Evaluation
            await self.run_evaluation()
            
            self.status.state = PipelineState.COMPLETED
            self._save_state()
            
            elapsed = datetime.now() - self.status.start_time
            self._log("=" * 60)
            self._log(f"PIPELINE COMPLETED in {elapsed}")
            self._log("=" * 60)
            
            return True
            
        except Exception as e:
            self.status.state = PipelineState.FAILED
            self._log(f"Pipeline error: {e}", "ERROR")
            traceback.print_exc()
            return False
    
    async def run_continuous(self):
        """Run the pipeline continuously."""
        self._log("Starting continuous training mode...")
        
        cycle = 0
        while not self._shutdown_requested:
            cycle += 1
            self._log(f"\n{'='*60}")
            self._log(f"CONTINUOUS MODE - Cycle {cycle}")
            self._log(f"{'='*60}\n")
            
            success = await self.run_once()
            
            if self._shutdown_requested:
                break
            
            if success:
                self._log(f"Cycle {cycle} completed. Waiting for next ingestion interval...")
                
                # Wait until next ingestion is due
                wait_seconds = self.config.ingestion_interval_hours * 3600
                for _ in range(wait_seconds):
                    if self._shutdown_requested:
                        break
                    await asyncio.sleep(1)
            else:
                self._log(f"Cycle {cycle} failed. Retrying in {self.config.retry_delay_seconds}s...")
                await asyncio.sleep(self.config.retry_delay_seconds)
        
        self._log("Continuous mode stopped.")
        self._save_state()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="UMI Master Training Pipeline")
    
    # Mode
    parser.add_argument("--continuous", action="store_true", help="Run in continuous mode")
    parser.add_argument("--no-ingest", action="store_true", help="Skip data ingestion")
    
    # Model
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    
    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num-gpus", type=int, default=4)
    
    # Distributed
    parser.add_argument("--use-fsdp", action="store_true", help="Use FSDP instead of DeepSpeed")
    parser.add_argument("--deepspeed-stage", type=int, default=2, choices=[2, 3])
    
    # Data sources
    parser.add_argument("--sources", type=str, nargs="+", help="Data sources to ingest")
    
    # API Keys
    parser.add_argument("--kaggle-key", type=str)
    parser.add_argument("--umls-key", type=str)
    parser.add_argument("--disgenet-key", type=str)
    
    # Paths
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "outputs"))
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        base_model=args.model,
        max_seq_length=args.max_seq_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_gpus=args.num_gpus,
        use_deepspeed=not args.use_fsdp,
        deepspeed_stage=args.deepspeed_stage,
        run_ingestion=not args.no_ingest,
        continuous_mode=args.continuous,
        data_sources=args.sources if args.sources else None,
        kaggle_api_key=args.kaggle_key,
        umls_api_key=args.umls_key,
        disgenet_api_key=args.disgenet_key,
        output_dir=Path(args.output_dir),
    )
    
    if config.data_sources is None:
        config.data_sources = [
            "pubmed", "drugs", "trials", "rxnorm", "who",
            "kaggle", "medlineplus", "opentargets", "umls",
            "snomed", "orphanet", "disgenet", "chembl", "uniprot"
        ]
    
    pipeline = TrainingPipeline(config)
    
    if args.continuous:
        asyncio.run(pipeline.run_continuous())
    else:
        success = asyncio.run(pipeline.run_once())
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
