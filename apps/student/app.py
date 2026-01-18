"""
IMI Student App - Medical Education Assistant
For medical students, residents, and healthcare learners
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch


class StudentAssistant:
    """
    Medical education assistant powered by IMI medical LLM.
    Specialized for learning, exam prep, and clinical reasoning.
    """
    
    SYSTEM_PROMPT = """You are IMI Student, an AI tutor specialized in medical education. You help medical students and healthcare learners with:

1. **Concept Explanation**: Break down complex medical topics into understandable parts
2. **Clinical Reasoning**: Guide through differential diagnosis and clinical decision-making
3. **Exam Preparation**: USMLE, COMLEX, NCLEX, and other medical board exam prep
4. **Case Studies**: Analyze clinical cases with step-by-step reasoning
5. **Memory Aids**: Provide mnemonics and learning strategies

Teaching approach:
- Use the Socratic method when appropriate
- Provide clear explanations with clinical relevance
- Include high-yield points for exams
- Encourage critical thinking over memorization
- Always explain the "why" behind medical concepts"""

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
        self.quiz_history: List[Dict] = []
    
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
        """Generate educational response."""
        
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
        
        return response
    
    def explain_concept(self, topic: str, level: str = "medical_student") -> str:
        """Explain a medical concept at appropriate level."""
        level_prompts = {
            "pre_med": "Explain for a pre-med student with basic biology knowledge",
            "medical_student": "Explain for a medical student (MS1-MS2 level)",
            "resident": "Explain for a resident with clinical experience",
            "attending": "Explain at attending physician level with nuances",
        }
        
        query = f"""{level_prompts.get(level, level_prompts['medical_student'])}:

Topic: {topic}

Please provide:
1. Core concept explanation
2. Pathophysiology (if applicable)
3. Clinical relevance
4. High-yield points for exams
5. Common misconceptions
6. Memory aids or mnemonics (if helpful)"""
        
        return self.generate(query)
    
    def differential_diagnosis(self, presentation: str) -> str:
        """Guide through differential diagnosis."""
        query = f"""A patient presents with: {presentation}

Guide me through the differential diagnosis process:

1. What are the key features in this presentation?
2. What is your differential diagnosis (most likely to least likely)?
3. What additional history would you want?
4. What physical exam findings would you look for?
5. What initial workup would you order?
6. What are the "can't miss" diagnoses?

Use clinical reasoning and explain your thought process."""
        
        return self.generate(query, max_new_tokens=1500)
    
    def generate_quiz(self, topic: str, num_questions: int = 5, style: str = "usmle") -> str:
        """Generate practice questions."""
        query = f"""Generate {num_questions} {style.upper()}-style practice questions about: {topic}

For each question:
1. Present a clinical vignette or scenario
2. Provide 5 answer choices (A-E)
3. After all questions, provide:
   - Correct answers
   - Detailed explanations for each
   - Key learning points
   - Related high-yield facts

Make questions progressively more challenging."""
        
        return self.generate(query, max_new_tokens=2000)
    
    def case_study(self, case_description: str) -> str:
        """Analyze a clinical case."""
        query = f"""Clinical Case:
{case_description}

Please analyze this case:

1. **Summary**: Key features of the presentation
2. **Problem List**: Active problems to address
3. **Differential Diagnosis**: With reasoning
4. **Workup**: Tests to order and expected findings
5. **Management Plan**: Initial treatment approach
6. **Teaching Points**: What this case teaches us
7. **Follow-up**: What to monitor

Walk through your clinical reasoning step by step."""
        
        return self.generate(query, max_new_tokens=1500)
    
    def study_plan(self, exam: str, weeks_until: int, weak_areas: str = "") -> str:
        """Create a study plan."""
        query = f"""Create a study plan for {exam} with {weeks_until} weeks remaining.

{"Weak areas to focus on: " + weak_areas if weak_areas else ""}

Please provide:
1. Week-by-week breakdown
2. Daily study schedule template
3. Resource recommendations
4. Practice question strategy
5. Review and retention techniques
6. Test-taking strategies
7. Self-care and burnout prevention tips"""
        
        return self.generate(query, max_new_tokens=1500)


def create_gradio_app(assistant: StudentAssistant) -> gr.Blocks:
    """Create Gradio interface for Student app."""
    
    with gr.Blocks(title="IMI Student - Medical Education Assistant", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 📚 IMI Student
        ### AI-Powered Medical Education Assistant
        
        Your personal tutor for medical learning, exam prep, and clinical reasoning.
        """)
        
        with gr.Tabs():
            # Ask Anything Tab
            with gr.Tab("💬 Ask Anything"):
                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask about any medical topic, concept, or clinical scenario...",
                            lines=4,
                        )
                        query_btn = gr.Button("🎓 Get Answer", variant="primary")
                    
                    with gr.Column(scale=4):
                        query_output = gr.Textbox(label="Response", lines=20)
                
                query_btn.click(
                    fn=assistant.generate,
                    inputs=query_input,
                    outputs=query_output,
                )
            
            # Concept Explainer Tab
            with gr.Tab("📖 Concept Explainer"):
                gr.Markdown("### Get clear explanations of medical concepts")
                
                with gr.Row():
                    topic_input = gr.Textbox(
                        label="Topic",
                        placeholder="e.g., Heart failure pathophysiology",
                    )
                    level_dropdown = gr.Dropdown(
                        choices=["pre_med", "medical_student", "resident", "attending"],
                        label="Your Level",
                        value="medical_student",
                    )
                
                explain_btn = gr.Button("📚 Explain", variant="primary")
                explain_output = gr.Textbox(label="Explanation", lines=20)
                
                explain_btn.click(
                    fn=assistant.explain_concept,
                    inputs=[topic_input, level_dropdown],
                    outputs=explain_output,
                )
            
            # Clinical Reasoning Tab
            with gr.Tab("🩺 Clinical Reasoning"):
                gr.Markdown("### Practice differential diagnosis")
                
                presentation_input = gr.Textbox(
                    label="Patient Presentation",
                    placeholder="e.g., 65-year-old male with sudden onset chest pain radiating to left arm, diaphoresis...",
                    lines=4,
                )
                ddx_btn = gr.Button("🔍 Analyze", variant="primary")
                ddx_output = gr.Textbox(label="Clinical Reasoning", lines=25)
                
                ddx_btn.click(
                    fn=assistant.differential_diagnosis,
                    inputs=presentation_input,
                    outputs=ddx_output,
                )
            
            # Practice Questions Tab
            with gr.Tab("📝 Practice Questions"):
                gr.Markdown("### Generate practice questions for exam prep")
                
                with gr.Row():
                    quiz_topic = gr.Textbox(label="Topic", placeholder="e.g., Cardiology, Renal physiology")
                    quiz_num = gr.Slider(1, 10, value=5, step=1, label="Number of Questions")
                    quiz_style = gr.Dropdown(
                        choices=["usmle", "comlex", "nclex", "shelf"],
                        label="Question Style",
                        value="usmle",
                    )
                
                quiz_btn = gr.Button("📋 Generate Quiz", variant="primary")
                quiz_output = gr.Textbox(label="Practice Questions", lines=30)
                
                quiz_btn.click(
                    fn=assistant.generate_quiz,
                    inputs=[quiz_topic, quiz_num, quiz_style],
                    outputs=quiz_output,
                )
            
            # Case Study Tab
            with gr.Tab("📋 Case Study"):
                gr.Markdown("### Analyze clinical cases")
                
                case_input = gr.Textbox(
                    label="Case Description",
                    placeholder="Enter a clinical case or paste from a case bank...",
                    lines=8,
                )
                case_btn = gr.Button("🔬 Analyze Case", variant="primary")
                case_output = gr.Textbox(label="Case Analysis", lines=25)
                
                case_btn.click(
                    fn=assistant.case_study,
                    inputs=case_input,
                    outputs=case_output,
                )
            
            # Study Planner Tab
            with gr.Tab("📅 Study Planner"):
                gr.Markdown("### Create a personalized study plan")
                
                with gr.Row():
                    exam_input = gr.Dropdown(
                        choices=["USMLE Step 1", "USMLE Step 2 CK", "USMLE Step 3", "COMLEX Level 1", "COMLEX Level 2", "NCLEX-RN", "MCAT", "Shelf Exams"],
                        label="Exam",
                        value="USMLE Step 1",
                    )
                    weeks_input = gr.Slider(1, 24, value=8, step=1, label="Weeks Until Exam")
                
                weak_areas_input = gr.Textbox(
                    label="Weak Areas (optional)",
                    placeholder="e.g., Biochemistry, Cardiology, Biostatistics",
                )
                plan_btn = gr.Button("📅 Create Plan", variant="primary")
                plan_output = gr.Textbox(label="Study Plan", lines=25)
                
                plan_btn.click(
                    fn=assistant.study_plan,
                    inputs=[exam_input, weeks_input, weak_areas_input],
                    outputs=plan_output,
                )
        
        gr.Markdown("""
        ---
        💡 **Tips**: 
        - Be specific in your questions for better answers
        - Use the Clinical Reasoning tab to practice diagnostic thinking
        - Generate practice questions regularly to test your knowledge
        """)
    
    return app


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="IMI Student App")
    parser.add_argument("--model", default="outputs/imi-medical/sft", help="Model path")
    parser.add_argument("--port", type=int, default=7861, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()
    
    assistant = StudentAssistant(model_path=args.model)
    assistant.load_model()
    
    app = create_gradio_app(assistant)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
