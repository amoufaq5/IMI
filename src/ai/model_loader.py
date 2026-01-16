"""
UMI Model Loader
Handles loading and managing AI models for inference
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    model_name: str
    use_4bit: bool = True
    use_8bit: bool = False
    device_map: str = "auto"
    torch_dtype: str = "float16"
    trust_remote_code: bool = True
    max_memory: Optional[Dict[int, str]] = None
    load_in_low_memory: bool = False


class ModelLoader:
    """
    Handles loading and caching of AI models.
    Supports both base models and fine-tuned LoRA adapters.
    """
    
    _instance = None
    _models: Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer]] = {}
    
    def __new__(cls):
        """Singleton pattern for model caching."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"model_loader_initialized", device=self.device)
    
    def get_quantization_config(self, config: ModelConfig) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration for memory-efficient loading."""
        if not config.use_4bit and not config.use_8bit:
            return None
        
        if config.use_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, config.torch_dtype),
                bnb_4bit_use_double_quant=True,
            )
        
        if config.use_8bit:
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        return None
    
    def load_model(
        self,
        config: ModelConfig,
        lora_path: Optional[str] = None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a model with optional LoRA adapters.
        
        Args:
            config: Model configuration
            lora_path: Optional path to LoRA adapter weights
        
        Returns:
            Tuple of (model, tokenizer)
        """
        cache_key = f"{config.model_name}:{lora_path or 'base'}"
        
        # Return cached model if available
        if cache_key in self._models:
            logger.info("using_cached_model", model=cache_key)
            return self._models[cache_key]
        
        logger.info("loading_model", model=config.model_name, lora=lora_path)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Get quantization config
        quant_config = self.get_quantization_config(config)
        
        # Determine torch dtype
        torch_dtype = getattr(torch, config.torch_dtype)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=quant_config,
            device_map=config.device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=config.trust_remote_code,
            max_memory=config.max_memory,
            low_cpu_mem_usage=config.load_in_low_memory,
        )
        
        # Load LoRA adapters if provided
        if lora_path and Path(lora_path).exists():
            logger.info("loading_lora_adapters", path=lora_path)
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_path)
        
        # Set to evaluation mode
        model.eval()
        
        # Cache the model
        self._models[cache_key] = (model, tokenizer)
        
        logger.info(
            "model_loaded",
            model=config.model_name,
            parameters=model.num_parameters(),
            device=self.device,
        )
        
        return model, tokenizer
    
    def load_umi_medical_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load the UMI fine-tuned medical model."""
        
        # Check for fine-tuned model first
        umi_model_path = Path("models/umi-medical-v1")
        lora_path = str(umi_model_path) if umi_model_path.exists() else None
        
        config = ModelConfig(
            model_name=settings.llm_model_name,
            use_4bit=True,
            torch_dtype="float16",
        )
        
        return self.load_model(config, lora_path)
    
    def load_embedding_model(self):
        """Load the embedding model for RAG."""
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = settings.embedding_model
            logger.info("loading_embedding_model", model=model_name)
            
            model = SentenceTransformer(model_name)
            
            if torch.cuda.is_available():
                model = model.to("cuda")
            
            return model
        
        except ImportError:
            logger.warning("sentence_transformers_not_installed")
            return None
    
    def unload_model(self, model_name: str) -> None:
        """Unload a model from memory."""
        keys_to_remove = [k for k in self._models if model_name in k]
        
        for key in keys_to_remove:
            model, _ = self._models.pop(key)
            del model
            torch.cuda.empty_cache()
            logger.info("model_unloaded", model=key)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {"device": "cpu", "memory": "N/A"}
        
        return {
            "device": torch.cuda.get_device_name(0),
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }


class InferenceEngine:
    """
    High-level inference engine for UMI models.
    """
    
    def __init__(self):
        self.loader = ModelLoader()
        self._model = None
        self._tokenizer = None
    
    async def initialize(self) -> None:
        """Initialize the inference engine with the UMI model."""
        if self._model is None:
            self._model, self._tokenizer = self.loader.load_umi_medical_model()
    
    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: User prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            system_prompt: Optional system prompt
        
        Returns:
            Generated text
        """
        await self.initialize()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template
        formatted = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize
        inputs = self._tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self._model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        
        # Decode
        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return response.strip()
    
    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ):
        """
        Generate text with streaming output.
        
        Yields tokens as they are generated.
        """
        await self.initialize()
        
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template
        formatted = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize
        inputs = self._tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self._model.device)
        
        # Setup streamer
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        
        # Generate in thread
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "streamer": streamer,
        }
        
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield tokens
        for token in streamer:
            yield token
        
        thread.join()


# Global inference engine instance
inference_engine = InferenceEngine()
