"""
Adapter Evaluation Script

Evaluates trained LoRA adapters on test sets:
- Perplexity measurement
- Task-specific metrics
- Generation quality assessment
"""
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ADAPTERS_DIR = PROJECT_ROOT / "adapters"


class AdapterEvaluator:
    """Evaluates trained adapters"""
    
    def __init__(
        self,
        base_model_name: str = "epfl-llm/meditron-7b",
        adapter_path: Optional[Path] = None,
    ):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load model with adapter"""
        logger.info(f"Loading base model: {self.base_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        if self.adapter_path and self.adapter_path.exists():
            logger.info(f"Loading adapter from: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        
        self.model.eval()
    
    def calculate_perplexity(self, texts: List[str]) -> float:
        """Calculate perplexity on a set of texts"""
        total_loss = 0
        total_tokens = 0
        
        for text in tqdm(texts, desc="Calculating perplexity"):
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
                total_tokens += inputs["input_ids"].shape[1]
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def generate_response(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate a response for a prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):]
    
    def evaluate_on_test_set(self, test_path: Path) -> Dict[str, Any]:
        """Evaluate on a test set"""
        with open(test_path) as f:
            test_data = json.load(f)
        
        results = {
            "total_examples": len(test_data),
            "perplexity": 0,
            "generations": [],
        }
        
        # Calculate perplexity
        texts = []
        for example in test_data:
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", "")
            
            if input_text:
                full_text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                full_text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            texts.append(full_text)
        
        results["perplexity"] = self.calculate_perplexity(texts)
        
        # Generate samples
        sample_indices = list(range(min(10, len(test_data))))
        for idx in sample_indices:
            example = test_data[idx]
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            expected = example.get("output", "")
            
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            
            generated = self.generate_response(prompt)
            
            results["generations"].append({
                "prompt": prompt[:200] + "...",
                "expected": expected[:200] + "...",
                "generated": generated[:200] + "...",
            })
        
        return results
    
    def run_evaluation(self, adapter_name: str) -> Dict[str, Any]:
        """Run full evaluation for an adapter"""
        logger.info(f"Evaluating adapter: {adapter_name}")
        
        # Find test data
        test_paths = [
            DATA_DIR / "train" / f"{adapter_name}_val.json",
            DATA_DIR / "processed" / adapter_name,
            DATA_DIR / "synthetic" / f"{adapter_name}_synthetic.json",
        ]
        
        test_path = None
        for path in test_paths:
            if path.exists():
                test_path = path
                break
        
        if not test_path:
            logger.warning(f"No test data found for {adapter_name}")
            return {}
        
        # Load model
        self.adapter_path = ADAPTERS_DIR / adapter_name
        self.load_model()
        
        # Evaluate
        results = self.evaluate_on_test_set(test_path)
        
        # Save results
        results_path = ADAPTERS_DIR / adapter_name / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Perplexity: {results['perplexity']:.2f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained adapters")
    parser.add_argument("--adapter", required=True, help="Adapter name to evaluate")
    parser.add_argument("--base-model", default="epfl-llm/meditron-7b", help="Base model")
    
    args = parser.parse_args()
    
    evaluator = AdapterEvaluator(base_model_name=args.base_model)
    evaluator.run_evaluation(args.adapter)


if __name__ == "__main__":
    main()
