"""
Domain Adapters for Medical LLM

LoRA/QLoRA adapters for domain-specific fine-tuning:
- Patient triage
- Clinical pharmacist reasoning
- Regulatory QA assistant
- Research synthesis
- Medical education
"""
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path
import logging

from peft import LoraConfig, TaskType, get_peft_model

logger = logging.getLogger(__name__)


class AdapterType(str, Enum):
    """Domain adapter types"""
    PATIENT_TRIAGE = "patient_triage"
    CLINICAL_PHARMACIST = "clinical_pharmacist"
    REGULATORY_QA = "regulatory_qa"
    RESEARCH = "research"
    EDUCATION = "education"
    GENERAL_MEDICAL = "general_medical"


class AdapterConfig:
    """Configuration for LoRA adapters"""
    
    DEFAULT_LORA_CONFIG = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    
    ADAPTER_CONFIGS: Dict[AdapterType, Dict[str, Any]] = {
        AdapterType.PATIENT_TRIAGE: {
            "r": 16,
            "lora_alpha": 32,
            "description": "Optimized for patient symptom assessment and triage",
            "training_focus": ["symptom_analysis", "urgency_assessment", "referral_decision"],
        },
        AdapterType.CLINICAL_PHARMACIST: {
            "r": 32,
            "lora_alpha": 64,
            "description": "Optimized for drug information and clinical pharmacy",
            "training_focus": ["drug_interactions", "dosing", "patient_counseling"],
        },
        AdapterType.REGULATORY_QA: {
            "r": 16,
            "lora_alpha": 32,
            "description": "Optimized for pharmaceutical QA and regulatory compliance",
            "training_focus": ["gmp_compliance", "documentation", "regulatory_guidance"],
        },
        AdapterType.RESEARCH: {
            "r": 32,
            "lora_alpha": 64,
            "description": "Optimized for research synthesis and literature review",
            "training_focus": ["literature_synthesis", "study_design", "data_interpretation"],
        },
        AdapterType.EDUCATION: {
            "r": 16,
            "lora_alpha": 32,
            "description": "Optimized for medical education and USMLE prep",
            "training_focus": ["concept_explanation", "question_answering", "clinical_correlation"],
        },
        AdapterType.GENERAL_MEDICAL: {
            "r": 16,
            "lora_alpha": 32,
            "description": "General medical knowledge adapter",
            "training_focus": ["medical_qa", "disease_info", "treatment_overview"],
        },
    }


class DomainAdapter:
    """
    Domain-specific LoRA adapter manager
    
    Handles loading, switching, and managing domain adapters
    for specialized medical tasks.
    """
    
    def __init__(self, base_adapter_path: Optional[str] = None):
        self.base_path = Path(base_adapter_path) if base_adapter_path else Path("/models/adapters")
        self.loaded_adapters: Dict[AdapterType, str] = {}
        self.current_adapter: Optional[AdapterType] = None
    
    def get_adapter_path(self, adapter_type: AdapterType) -> Path:
        """Get the path for an adapter"""
        return self.base_path / adapter_type.value
    
    def get_lora_config(self, adapter_type: AdapterType) -> LoraConfig:
        """Get LoRA configuration for an adapter type"""
        config = AdapterConfig.ADAPTER_CONFIGS.get(
            adapter_type,
            AdapterConfig.DEFAULT_LORA_CONFIG
        )
        
        return LoraConfig(
            r=config.get("r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=AdapterConfig.DEFAULT_LORA_CONFIG["lora_dropout"],
            bias=AdapterConfig.DEFAULT_LORA_CONFIG["bias"],
            task_type=AdapterConfig.DEFAULT_LORA_CONFIG["task_type"],
            target_modules=AdapterConfig.DEFAULT_LORA_CONFIG["target_modules"],
        )
    
    def create_adapter_for_training(
        self,
        base_model,
        adapter_type: AdapterType,
    ):
        """Create a new adapter for training"""
        lora_config = self.get_lora_config(adapter_type)
        model = get_peft_model(base_model, lora_config)
        
        logger.info(f"Created {adapter_type.value} adapter for training")
        logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")
        
        return model
    
    def get_adapter_info(self, adapter_type: AdapterType) -> Dict[str, Any]:
        """Get information about an adapter"""
        config = AdapterConfig.ADAPTER_CONFIGS.get(adapter_type, {})
        return {
            "type": adapter_type.value,
            "description": config.get("description", ""),
            "training_focus": config.get("training_focus", []),
            "lora_r": config.get("r", 16),
            "lora_alpha": config.get("lora_alpha", 32),
            "path": str(self.get_adapter_path(adapter_type)),
            "is_loaded": adapter_type in self.loaded_adapters,
        }
    
    def list_available_adapters(self) -> List[Dict[str, Any]]:
        """List all available adapters"""
        return [self.get_adapter_info(at) for at in AdapterType]
    
    def select_adapter_for_task(self, task_description: str) -> AdapterType:
        """Select the best adapter for a given task"""
        task_lower = task_description.lower()
        
        if any(kw in task_lower for kw in ["triage", "symptom", "emergency", "urgent"]):
            return AdapterType.PATIENT_TRIAGE
        
        if any(kw in task_lower for kw in ["drug", "medication", "interaction", "dose", "pharmacy"]):
            return AdapterType.CLINICAL_PHARMACIST
        
        if any(kw in task_lower for kw in ["qa", "regulatory", "compliance", "gmp", "fda", "validation"]):
            return AdapterType.REGULATORY_QA
        
        if any(kw in task_lower for kw in ["research", "study", "trial", "literature", "patent"]):
            return AdapterType.RESEARCH
        
        if any(kw in task_lower for kw in ["usmle", "exam", "study", "learn", "education", "student"]):
            return AdapterType.EDUCATION
        
        return AdapterType.GENERAL_MEDICAL


class AdapterTrainingConfig:
    """Configuration for adapter training"""
    
    def __init__(
        self,
        adapter_type: AdapterType,
        output_dir: str,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_ratio: float = 0.03,
        logging_steps: int = 10,
        save_steps: int = 100,
        eval_steps: int = 100,
    ):
        self.adapter_type = adapter_type
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
    
    def to_training_arguments(self):
        """Convert to HuggingFace TrainingArguments"""
        from transformers import TrainingArguments
        
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            fp16=True,
            optim="paged_adamw_32bit",
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="tensorboard",
        )
