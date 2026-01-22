"""
IMI Gradio Demo Application

Web-based demo interface for the IMI platform.
"""
import gradio as gr
import asyncio
from typing import List, Tuple

from src.domains.patient import PatientService, SymptomAssessmentRequest
from src.domains.general import GeneralService
from src.domains.student import StudentService, ExamType


# Initialize services
patient_service = PatientService()
general_service = GeneralService()
student_service = StudentService()


def assess_symptoms(
    symptoms: str,
    chief_complaint: str,
    age: int,
    gender: str,
    conditions: str,
    medications: str,
) -> str:
    """Assess patient symptoms"""
    request = SymptomAssessmentRequest(
        symptoms=[s.strip() for s in symptoms.split(",") if s.strip()],
        chief_complaint=chief_complaint,
        age=age,
        gender=gender if gender != "Select" else None,
        medical_conditions=[c.strip() for c in conditions.split(",") if c.strip()],
        current_medications=[m.strip() for m in medications.split(",") if m.strip()],
    )
    
    result = asyncio.run(patient_service.assess_symptoms(request))
    
    output = f"""
## Triage Result

**Urgency Level:** {result.triage_result.get('urgency', 'Unknown')}

**Explanation:** {result.explanation}

### Next Steps
"""
    for step in result.next_steps:
        output += f"- {step}\n"
    
    if result.red_flags_detected:
        output += "\n### ⚠️ Red Flags Detected\n"
        for flag in result.red_flags_detected:
            output += f"- {flag}\n"
    
    output += f"\n---\n*{result.disclaimer}*"
    
    return output


def get_drug_info(drug_name: str) -> str:
    """Get drug information"""
    result = asyncio.run(general_service.get_drug_info(drug_name))
    
    if not result.get("found"):
        return f"Drug '{drug_name}' not found in database."
    
    output = f"""
## {result.get('name', drug_name)}

**Generic Name:** {result.get('generic_name', 'N/A')}
**Drug Class:** {result.get('drug_class', 'N/A')}
**Mechanism:** {result.get('mechanism', 'N/A')}

### Indications
"""
    for indication in result.get("indications", []):
        output += f"- {indication}\n"
    
    output += "\n### Side Effects\n"
    output += "**Common:** " + ", ".join(result.get("side_effects", {}).get("common", [])) + "\n"
    output += "**Serious:** " + ", ".join(result.get("side_effects", {}).get("serious", [])) + "\n"
    
    return output


def answer_usmle_question(question: str, options: str) -> str:
    """Answer a USMLE-style question"""
    option_list = [o.strip() for o in options.split("\n") if o.strip()]
    
    result = asyncio.run(student_service.answer_question(
        question=question,
        options=option_list if option_list else None,
        exam_type=ExamType.USMLE_STEP1,
    ))
    
    output = f"""
## Question Analysis

**Topic:** {result.get('analysis', {}).get('likely_topic', 'General')}
**Question Type:** {result.get('analysis', {}).get('question_type', 'Unknown')}

### Explanation
{result.get('explanation', 'No explanation available.')}

### High-Yield Points
"""
    for point in result.get("high_yield_points", []):
        output += f"- {point}\n"
    
    return output


# Create Gradio interface
with gr.Blocks(title="IMI - Intelligent Medical Interface") as demo:
    gr.Markdown("# IMI - Intelligent Medical Interface")
    gr.Markdown("AI-powered medical assistance platform with safety-first architecture")
    
    with gr.Tabs():
        # Patient Tab
        with gr.TabItem("Patient Assessment"):
            gr.Markdown("### Symptom Assessment")
            with gr.Row():
                with gr.Column():
                    symptoms_input = gr.Textbox(
                        label="Symptoms (comma-separated)",
                        placeholder="headache, fever, fatigue"
                    )
                    complaint_input = gr.Textbox(
                        label="Chief Complaint",
                        placeholder="What is your main concern?"
                    )
                    age_input = gr.Number(label="Age", value=30)
                    gender_input = gr.Dropdown(
                        label="Gender",
                        choices=["Select", "male", "female", "other"],
                        value="Select"
                    )
                    conditions_input = gr.Textbox(
                        label="Medical Conditions (comma-separated)",
                        placeholder="diabetes, hypertension"
                    )
                    medications_input = gr.Textbox(
                        label="Current Medications (comma-separated)",
                        placeholder="metformin, lisinopril"
                    )
                    assess_btn = gr.Button("Assess Symptoms", variant="primary")
                
                with gr.Column():
                    assessment_output = gr.Markdown(label="Assessment Result")
            
            assess_btn.click(
                assess_symptoms,
                inputs=[symptoms_input, complaint_input, age_input, gender_input, conditions_input, medications_input],
                outputs=assessment_output
            )
        
        # Drug Information Tab
        with gr.TabItem("Drug Information"):
            gr.Markdown("### Drug Lookup")
            with gr.Row():
                with gr.Column():
                    drug_input = gr.Textbox(
                        label="Drug Name",
                        placeholder="Enter drug name (e.g., ibuprofen)"
                    )
                    drug_btn = gr.Button("Get Information", variant="primary")
                
                with gr.Column():
                    drug_output = gr.Markdown(label="Drug Information")
            
            drug_btn.click(get_drug_info, inputs=drug_input, outputs=drug_output)
        
        # Student Tab
        with gr.TabItem("USMLE Practice"):
            gr.Markdown("### USMLE Question Helper")
            with gr.Row():
                with gr.Column():
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="Enter the USMLE question...",
                        lines=5
                    )
                    options_input = gr.Textbox(
                        label="Options (one per line)",
                        placeholder="A. Option 1\nB. Option 2\nC. Option 3",
                        lines=5
                    )
                    usmle_btn = gr.Button("Analyze Question", variant="primary")
                
                with gr.Column():
                    usmle_output = gr.Markdown(label="Analysis")
            
            usmle_btn.click(
                answer_usmle_question,
                inputs=[question_input, options_input],
                outputs=usmle_output
            )
    
    gr.Markdown("---")
    gr.Markdown("*This is a demo interface. Always consult healthcare professionals for medical decisions.*")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
