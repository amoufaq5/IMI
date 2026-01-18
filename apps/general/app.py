"""
IMI General App - Health Information Assistant
For general users seeking health information and guidance
"""

import json
import os
from pathlib import Path
from typing import Optional

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch


class GeneralHealthAssistant:
    """
    General health assistant powered by IMI medical LLM.
    Designed for non-medical users seeking health information.
    """
    
    SYSTEM_PROMPT = """You are IMI Health, a friendly AI health information assistant. You help general users understand health topics in plain language.

Your role:
1. **Health Education**: Explain medical conditions, treatments, and prevention in simple terms
2. **Symptom Information**: Help users understand symptoms (NOT diagnose)
3. **Medication Guidance**: Explain how medications work and general precautions
4. **Wellness Tips**: Provide evidence-based lifestyle and prevention advice
5. **Healthcare Navigation**: Help users understand when to seek care

IMPORTANT GUIDELINES:
- Always use plain, non-technical language
- Never diagnose conditions or prescribe treatments
- Encourage users to consult healthcare providers for personal medical advice
- Be empathetic and supportive
- Provide balanced, evidence-based information
- Include appropriate disclaimers for health-related queries

Remember: You provide information, not medical advice. Always recommend consulting a healthcare provider for personal health concerns."""

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
        
        peft_config_path = Path(self.model_path) / "adapter_config.json"
        
        if peft_config_path.exists():
            with open(peft_config_path) as f:
                peft_config = json.load(f)
            base_model_name = peft_config.get("base_model_name_or_path")
            
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
    ) -> str:
        """Generate response to health query."""
        
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
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        response = outputs[0]["generated_text"]
        
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        # Add disclaimer for health queries
        disclaimer = "\n\n---\n*This information is for educational purposes only and is not a substitute for professional medical advice. Please consult a healthcare provider for personal medical concerns.*"
        
        return response + disclaimer
    
    def explain_condition(self, condition: str) -> str:
        """Explain a health condition in plain language."""
        query = f"""Please explain {condition} in simple, easy-to-understand terms.

Include:
1. What it is (simple explanation)
2. Common symptoms
3. What causes it
4. How it's typically treated
5. When to see a doctor
6. Prevention tips (if applicable)

Use everyday language that anyone can understand."""
        
        return self.generate(query)
    
    def medication_info(self, medication: str) -> str:
        """Provide general medication information."""
        query = f"""Please provide general information about {medication}.

Include:
1. What it's commonly used for
2. How it generally works
3. Common side effects to be aware of
4. General precautions
5. Important interactions to know about
6. Questions to ask your doctor or pharmacist

Remember: This is general information only. Always follow your doctor's specific instructions."""
        
        return self.generate(query)
    
    def symptom_checker(self, symptoms: str) -> str:
        """Provide information about symptoms (not diagnosis)."""
        query = f"""I'm experiencing: {symptoms}

Please help me understand:
1. What these symptoms might generally indicate
2. When these symptoms typically require immediate medical attention
3. What information I should prepare for a doctor visit
4. General self-care measures while waiting to see a doctor
5. Red flags that mean I should seek emergency care

IMPORTANT: This is NOT a diagnosis. Please see a healthcare provider for proper evaluation."""
        
        return self.generate(query, max_new_tokens=1200)
    
    def wellness_tips(self, topic: str) -> str:
        """Provide wellness and prevention tips."""
        query = f"""Please provide evidence-based wellness tips about: {topic}

Include:
1. Why this matters for health
2. Practical, actionable tips
3. Common myths vs. facts
4. How to get started
5. When to seek professional guidance

Keep the advice practical and achievable for everyday life."""
        
        return self.generate(query)
    
    def prepare_for_appointment(self, reason: str) -> str:
        """Help prepare for a doctor's appointment."""
        query = f"""I have a doctor's appointment for: {reason}

Please help me prepare:
1. Questions I should ask my doctor
2. Information I should bring or have ready
3. Symptoms or changes to track before the visit
4. What to expect during the appointment
5. How to make the most of my time with the doctor

This will help me have a productive conversation with my healthcare provider."""
        
        return self.generate(query)


def create_gradio_app(assistant: GeneralHealthAssistant) -> gr.Blocks:
    """Create Gradio interface for General Health app."""
    
    with gr.Blocks(title="IMI Health - Your Health Information Assistant", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🏥 IMI Health
        ### Your Friendly Health Information Assistant
        
        Get clear, easy-to-understand health information. Remember: This is for information only - 
        always consult a healthcare provider for personal medical advice.
        """)
        
        with gr.Tabs():
            # Ask a Question Tab
            with gr.Tab("💬 Ask a Question"):
                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="Your Health Question",
                            placeholder="Ask any health-related question in plain language...",
                            lines=4,
                        )
                        query_btn = gr.Button("🔍 Get Information", variant="primary")
                    
                    with gr.Column(scale=4):
                        query_output = gr.Textbox(label="Response", lines=20)
                
                query_btn.click(
                    fn=assistant.generate,
                    inputs=query_input,
                    outputs=query_output,
                )
            
            # Understand a Condition Tab
            with gr.Tab("📖 Understand a Condition"):
                gr.Markdown("### Learn about health conditions in plain language")
                
                condition_input = gr.Textbox(
                    label="Condition Name",
                    placeholder="e.g., High blood pressure, Diabetes, Anxiety",
                )
                condition_btn = gr.Button("📚 Explain", variant="primary")
                condition_output = gr.Textbox(label="Explanation", lines=20)
                
                condition_btn.click(
                    fn=assistant.explain_condition,
                    inputs=condition_input,
                    outputs=condition_output,
                )
            
            # Medication Information Tab
            with gr.Tab("💊 Medication Info"):
                gr.Markdown("### Learn about medications")
                
                med_input = gr.Textbox(
                    label="Medication Name",
                    placeholder="e.g., Ibuprofen, Metformin, Lisinopril",
                )
                med_btn = gr.Button("💊 Get Info", variant="primary")
                med_output = gr.Textbox(label="Medication Information", lines=20)
                
                med_btn.click(
                    fn=assistant.medication_info,
                    inputs=med_input,
                    outputs=med_output,
                )
            
            # Symptom Information Tab
            with gr.Tab("🩺 Symptom Information"):
                gr.Markdown("""### Understand your symptoms
                
                ⚠️ **Important**: This is NOT a diagnosis tool. If you're experiencing severe symptoms, 
                chest pain, difficulty breathing, or other emergencies, call emergency services immediately.
                """)
                
                symptom_input = gr.Textbox(
                    label="Describe Your Symptoms",
                    placeholder="e.g., Headache for 3 days, mild fever, fatigue...",
                    lines=4,
                )
                symptom_btn = gr.Button("🔍 Get Information", variant="primary")
                symptom_output = gr.Textbox(label="Symptom Information", lines=20)
                
                symptom_btn.click(
                    fn=assistant.symptom_checker,
                    inputs=symptom_input,
                    outputs=symptom_output,
                )
            
            # Wellness Tips Tab
            with gr.Tab("🌱 Wellness Tips"):
                gr.Markdown("### Evidence-based wellness advice")
                
                wellness_topic = gr.Dropdown(
                    choices=[
                        "Better sleep",
                        "Stress management",
                        "Healthy eating",
                        "Exercise and fitness",
                        "Heart health",
                        "Mental wellness",
                        "Weight management",
                        "Quitting smoking",
                        "Reducing alcohol",
                        "Healthy aging",
                    ],
                    label="Choose a Topic",
                    value="Better sleep",
                )
                wellness_btn = gr.Button("🌱 Get Tips", variant="primary")
                wellness_output = gr.Textbox(label="Wellness Tips", lines=20)
                
                wellness_btn.click(
                    fn=assistant.wellness_tips,
                    inputs=wellness_topic,
                    outputs=wellness_output,
                )
            
            # Prepare for Appointment Tab
            with gr.Tab("📋 Prepare for Doctor Visit"):
                gr.Markdown("### Get ready for your healthcare appointment")
                
                appt_reason = gr.Textbox(
                    label="Reason for Visit",
                    placeholder="e.g., Annual checkup, Back pain, Follow-up for diabetes",
                )
                appt_btn = gr.Button("📋 Prepare", variant="primary")
                appt_output = gr.Textbox(label="Appointment Preparation Guide", lines=20)
                
                appt_btn.click(
                    fn=assistant.prepare_for_appointment,
                    inputs=appt_reason,
                    outputs=appt_output,
                )
        
        gr.Markdown("""
        ---
        ⚠️ **Important Disclaimer**: 
        
        This tool provides general health information for educational purposes only. It is NOT a substitute 
        for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician 
        or other qualified health provider with any questions you may have regarding a medical condition.
        
        **In case of emergency, call your local emergency number immediately.**
        """)
    
    return app


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="IMI General Health App")
    parser.add_argument("--model", default="outputs/imi-medical/sft", help="Model path")
    parser.add_argument("--port", type=int, default=7862, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()
    
    assistant = GeneralHealthAssistant(model_path=args.model)
    assistant.load_model()
    
    app = create_gradio_app(assistant)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
