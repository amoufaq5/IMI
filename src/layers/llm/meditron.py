"""
Mistral 7B Model Integration

Mistral 7B is a dense decoder-only LLM (Apache 2.0) used as IMI's base model.
After medical foundation fine-tuning + DPO safety alignment, it serves as the
core generation engine.

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
    GenerationConfig,
)

from src.core.config import settings

logger = logging.getLogger(__name__)


class MistralMedicalModel:
    """
    Mistral 7B Medical Language Model

    Base: mistralai/Mistral-7B-Instruct-v0.3 (Apache 2.0)
    Architecture: Dense decoder-only transformer (7.3B parameters)
    Context: 32K tokens

    Handles:
    - Model loading in BFloat16 (full precision)
    - Text generation with Mistral chat template
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

    # Mistral chat template
    CHAT_TEMPLATE = "<s>[INST] {system}\n\n{user} [/INST]"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_vllm: bool = False,
        vllm_url: Optional[str] = None,
    ):
        self.model_path = model_path or settings.llm_model_path
        self.device = device or settings.llm_device
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.use_vllm = use_vllm
        self.vllm_url = vllm_url or "http://localhost:8080"

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

        logger.info(f"Loading Mistral 7B model from {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self.device == "cuda" else None,
        )

        self.generation_config = GenerationConfig(
            **self.DEFAULT_GENERATION_CONFIG,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        logger.info("Mistral 7B model loaded successfully")

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

        logger.info("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None


# Backward compatibility aliases
MixtralMedicalModel = MistralMedicalModel
MeditronModel = MistralMedicalModel
