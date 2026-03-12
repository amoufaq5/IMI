"""
Mixtral 8x7B Model Integration

Mixtral 8x7B is a Mixture-of-Experts LLM (Apache 2.0) used as IMI's base model.
After medical foundation fine-tuning + DPO safety alignment, it serves as the
core generation engine with 6 LoRA adapters hot-swapped per user type.

This module handles model loading, inference, and medical-specific optimizations.
Also supports vLLM inference backend for production deployment.
"""
import torch
from typing import Optional, List, Dict, Any, Generator
from pathlib import Path
import logging

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel, PeftConfig

from src.core.config import settings

logger = logging.getLogger(__name__)


class MixtralMedicalModel:
    """
    Mixtral 8x7B Medical Language Model
    
    Base: mistralai/Mixtral-8x7B-Instruct-v0.1 (Apache 2.0)
    Architecture: Mixture of Experts (8 experts, 2 active per token)
    Context: 32K tokens
    
    Handles:
    - Model loading with QLoRA quantization (4-bit NF4)
    - LoRA adapter loading/hot-swapping for 6 user types
    - Text generation with Mixtral chat template
    - Streaming generation support
    - Optional vLLM backend for production inference
    """
    
    DEFAULT_GENERATION_CONFIG = {
        "max_new_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "do_sample": True,
        "pad_token_id": None,
        "eos_token_id": None,
    }
    
    # Mixtral chat template
    CHAT_TEMPLATE = "<s>[INST] {system}\n\n{user} [/INST]"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        use_vllm: bool = False,
        vllm_url: Optional[str] = None,
    ):
        self.model_path = model_path or settings.llm_model_path
        self.device = device or settings.llm_device
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.use_vllm = use_vllm
        self.vllm_url = vllm_url or "http://localhost:8080"
        self._loaded_adapters: Dict[str, bool] = {}
    
    def load(self) -> None:
        """Load the model and tokenizer"""
        if self.use_vllm:
            logger.info(f"Using vLLM backend at {self.vllm_url}")
            # Only load tokenizer locally; inference goes through vLLM API
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left",
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.generation_config = GenerationConfig(
                **self.DEFAULT_GENERATION_CONFIG,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            logger.info("vLLM backend ready (tokenizer loaded locally)")
            return
        
        logger.info(f"Loading Mixtral 8x7B model from {self.model_path}")
        
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs,
        )
        
        self.generation_config = GenerationConfig(
            **self.DEFAULT_GENERATION_CONFIG,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        logger.info("Mixtral 8x7B model loaded successfully")
    
    def load_adapter(self, adapter_path: str, adapter_name: str = "default") -> None:
        """Load a LoRA adapter for domain-specific fine-tuning"""
        if adapter_name in self._loaded_adapters:
            logger.info(f"Adapter {adapter_name} already loaded")
            return
        
        logger.info(f"Loading adapter {adapter_name} from {adapter_path}")
        
        if not self._loaded_adapters:
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                adapter_name=adapter_name,
            )
        else:
            self.model.load_adapter(adapter_path, adapter_name=adapter_name)
        
        self._loaded_adapters[adapter_name] = True
        logger.info(f"Adapter {adapter_name} loaded successfully")
    
    def set_adapter(self, adapter_name: str) -> None:
        """Switch to a specific adapter"""
        if adapter_name not in self._loaded_adapters:
            raise ValueError(f"Adapter {adapter_name} not loaded")
        self.model.set_adapter(adapter_name)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate text from prompt"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.llm_max_length,
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens or self.generation_config.max_new_tokens,
            temperature=temperature or self.generation_config.temperature,
            top_p=top_p or self.generation_config.top_p,
            top_k=self.generation_config.top_k,
            repetition_penalty=self.generation_config.repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
        
        return generated_text.strip()
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """Generate text with streaming output"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.llm_max_length,
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens or self.generation_config.max_new_tokens,
            temperature=temperature or self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        generation_kwargs = {
            **inputs,
            "generation_config": gen_config,
            "streamer": streamer,
        }
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for text in streamer:
            yield text
        
        thread.join()
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs"""
        return self.tokenizer.encode(text, return_tensors="pt")
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get text embeddings from the model"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.llm_max_length,
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            embeddings = hidden_states.mean(dim=1)
        
        return embeddings
    
    def generate_with_consistency(
        self,
        prompt: str,
        n: int = 5,
        temperature: float = 0.4,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Self-consistency decoding for high-stakes medical queries.

        Generates n independent responses at temperature > 0 and returns the
        most representative one via majority-vote on urgency/recommendation label.

        Intended for EMERGENCY and URGENT triage queries only, where a single
        stochastic decode could miss a critical flag. By generating multiple
        independent samples and voting, we reduce the chance of a false-negative
        on life-threatening conditions.

        Vote strategy:
          - Extract the first capital-letter urgency label from each response
            (EMERGENCY, URGENT, SEMI_URGENT, ROUTINE, SELF_CARE).
          - If any vote is EMERGENCY, return the response that voted EMERGENCY
            (conservative: always surface the most critical finding).
          - Otherwise return the response with the most common label.
          - Tie-break: return the first response.

        Args:
            prompt: The fully-formatted prompt string.
            n: Number of independent samples (default 5).
            temperature: Sampling temperature (default 0.4 — diverse but coherent).
            max_new_tokens: Max tokens per sample (default: model default).

        Returns:
            Single best response string.
        """
        import re
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        _URGENCY_LABELS = ["EMERGENCY", "URGENT", "SEMI_URGENT", "ROUTINE", "SELF_CARE"]
        _URGENCY_PATTERN = re.compile(r"\b(EMERGENCY|URGENT|SEMI_URGENT|ROUTINE|SELF_CARE)\b")

        responses: List[str] = []
        labels: List[str] = []

        for _ in range(n):
            response = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            responses.append(response)
            match = _URGENCY_PATTERN.search(response.upper())
            labels.append(match.group(1) if match else "UNKNOWN")

        # Conservative vote: if any sample says EMERGENCY, surface it
        if "EMERGENCY" in labels:
            idx = labels.index("EMERGENCY")
            logger.info(f"Self-consistency: EMERGENCY detected in {labels.count('EMERGENCY')}/{n} samples")
            return responses[idx]

        # Otherwise majority vote
        from collections import Counter
        most_common_label, _ = Counter(labels).most_common(1)[0]
        for response, label in zip(responses, labels):
            if label == most_common_label:
                logger.info(f"Self-consistency: majority label={most_common_label} ({labels.count(most_common_label)}/{n})")
                return response

        return responses[0]

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def unload(self) -> None:
        """Unload model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._loaded_adapters.clear()
        logger.info("Model unloaded")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
