"""
IMI Pharma App - Drug Discovery & Research Assistant
For pharmaceutical researchers and drug developers
"""

import json
import os
from pathlib import Path
from typing import Optional

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch


class PharmaAssistant:
    """
    Pharmaceutical research assistant powered by IMI medical LLM.
    Specialized for drug discovery, clinical trials, and regulatory information.
    """
    
    SYSTEM_PROMPT = """You are IMI Pharma, an AI assistant specialized in pharmaceutical research and drug development. You help with:

1. **Drug Discovery**: Mechanism of action, target identification, lead optimization
2. **Clinical Trials**: Trial design, endpoints, patient selection, regulatory requirements
3. **Drug Interactions**: Pharmacokinetics, drug-drug interactions, contraindications
4. **Regulatory Affairs**: FDA/EMA guidelines, approval pathways, documentation
5. **Literature Review**: Summarizing research, identifying trends, competitive analysis

Always provide evidence-based information with appropriate caveats. Cite sources when possible.
For safety-critical information, recommend consulting official sources and qualified professionals."""

    def __init__(
        self,
        model_path: str = "outputs/imi-medical/sft",
        device: str = "auto",
    ):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.pipe = None
    
    def load_model(self):
        """Load the fine-tuned model."""
        print(f"Loading model from: {self.model_path}")
        
        # Check if it's a PEFT model
        peft_config_path = Path(self.model_path) / "adapter_config.json"
        
        if peft_config_path.exists():
            with open(peft_config_path) as f:
                peft_config = json.load(f)
            base_model_name = peft_config.get("base_model_name_or_path")
            
            print(f"Loading base model: {base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
            )
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        print("Model loaded successfully!")
    
    def generate(
        self,
        query: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate response to pharmaceutical query."""
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"[INST] {self.SYSTEM_PROMPT}\n\n{query} [/INST]"
        
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        response = outputs[0]["generated_text"]
        
        # Extract only the assistant's response
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        return response
    
    def drug_interaction_check(self, drugs: list) -> str:
        """Check for potential drug interactions."""
        query = f"""Analyze potential drug interactions between the following medications:
{', '.join(drugs)}

Please provide:
1. Known interactions and their severity
2. Mechanism of interaction
3. Clinical recommendations
4. Monitoring parameters"""
        
        return self.generate(query)
    
    def clinical_trial_design(self, indication: str, phase: str) -> str:
        """Help design a clinical trial."""
        query = f"""Help design a Phase {phase} clinical trial for {indication}.

Please provide recommendations for:
1. Primary and secondary endpoints
2. Patient inclusion/exclusion criteria
3. Sample size considerations
4. Study duration
5. Safety monitoring requirements
6. Regulatory considerations"""
        
        return self.generate(query)
    
    def mechanism_analysis(self, drug_name: str) -> str:
        """Analyze drug mechanism of action."""
        query = f"""Provide a detailed analysis of the mechanism of action for {drug_name}.

Include:
1. Molecular target(s)
2. Binding characteristics
3. Downstream signaling effects
4. Therapeutic implications
5. Potential off-target effects"""
        
        return self.generate(query)


def create_gradio_app(assistant: PharmaAssistant) -> gr.Blocks:
    """Create Gradio interface for Pharma app."""
    
    with gr.Blocks(title="IMI Pharma - Drug Research Assistant", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 💊 IMI Pharma
        ### AI-Powered Drug Discovery & Research Assistant
        
        Specialized assistance for pharmaceutical research, clinical trials, and drug development.
        """)
        
        with gr.Tabs():
            # General Query Tab
            with gr.Tab("💬 Research Query"):
                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask about drug mechanisms, clinical trials, regulatory requirements...",
                            lines=4,
                        )
                        with gr.Row():
                            temp_slider = gr.Slider(0.1, 1.0, value=0.7, label="Temperature")
                            max_tokens = gr.Slider(256, 2048, value=1024, step=128, label="Max Tokens")
                        query_btn = gr.Button("🔍 Get Answer", variant="primary")
                    
                    with gr.Column(scale=4):
                        query_output = gr.Textbox(label="Response", lines=20)
                
                query_btn.click(
                    fn=assistant.generate,
                    inputs=[query_input, max_tokens, temp_slider],
                    outputs=query_output,
                )
            
            # Drug Interactions Tab
            with gr.Tab("⚠️ Drug Interactions"):
                gr.Markdown("### Check for potential drug-drug interactions")
                
                drugs_input = gr.Textbox(
                    label="Enter medications (comma-separated)",
                    placeholder="e.g., Warfarin, Aspirin, Omeprazole",
                )
                interaction_btn = gr.Button("🔬 Check Interactions", variant="primary")
                interaction_output = gr.Textbox(label="Interaction Analysis", lines=15)
                
                def check_interactions(drugs_str):
                    drugs = [d.strip() for d in drugs_str.split(",")]
                    return assistant.drug_interaction_check(drugs)
                
                interaction_btn.click(
                    fn=check_interactions,
                    inputs=drugs_input,
                    outputs=interaction_output,
                )
            
            # Clinical Trial Design Tab
            with gr.Tab("📋 Trial Design"):
                gr.Markdown("### Clinical Trial Design Assistant")
                
                with gr.Row():
                    indication_input = gr.Textbox(label="Indication/Disease", placeholder="e.g., Type 2 Diabetes")
                    phase_dropdown = gr.Dropdown(
                        choices=["1", "1/2", "2", "2/3", "3", "4"],
                        label="Trial Phase",
                        value="2",
                    )
                
                trial_btn = gr.Button("📝 Generate Trial Design", variant="primary")
                trial_output = gr.Textbox(label="Trial Design Recommendations", lines=20)
                
                trial_btn.click(
                    fn=assistant.clinical_trial_design,
                    inputs=[indication_input, phase_dropdown],
                    outputs=trial_output,
                )
            
            # Mechanism Analysis Tab
            with gr.Tab("🧬 Mechanism Analysis"):
                gr.Markdown("### Drug Mechanism of Action Analysis")
                
                drug_input = gr.Textbox(label="Drug Name", placeholder="e.g., Metformin")
                moa_btn = gr.Button("🔬 Analyze Mechanism", variant="primary")
                moa_output = gr.Textbox(label="Mechanism Analysis", lines=20)
                
                moa_btn.click(
                    fn=assistant.mechanism_analysis,
                    inputs=drug_input,
                    outputs=moa_output,
                )
        
        gr.Markdown("""
        ---
        ⚠️ **Disclaimer**: This tool is for research purposes only. Always verify information with official sources 
        and consult qualified professionals for clinical decisions.
        """)
    
    return app


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="IMI Pharma App")
    parser.add_argument("--model", default="outputs/imi-medical/sft", help="Model path")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()
    
    # Initialize assistant
    assistant = PharmaAssistant(model_path=args.model)
    assistant.load_model()
    
    # Create and launch app
    app = create_gradio_app(assistant)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
