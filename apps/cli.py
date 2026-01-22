"""
IMI Command Line Interface

Provides CLI access to the IMI platform for testing and administration.
"""
import argparse
import asyncio
import json
from typing import Optional

from src.domains.patient import PatientService, SymptomAssessmentRequest
from src.domains.general import GeneralService
from src.layers.rule_engine import get_rule_engine_service


def assess_symptoms(args):
    """Run symptom assessment"""
    service = PatientService()
    
    request = SymptomAssessmentRequest(
        symptoms=args.symptoms.split(","),
        chief_complaint=args.complaint,
        age=args.age,
        gender=args.gender,
        medical_conditions=args.conditions.split(",") if args.conditions else [],
        current_medications=args.medications.split(",") if args.medications else [],
    )
    
    result = asyncio.run(service.assess_symptoms(request))
    print(json.dumps(result.model_dump(), indent=2, default=str))


def drug_info(args):
    """Get drug information"""
    service = GeneralService()
    result = asyncio.run(service.get_drug_info(args.drug_name))
    print(json.dumps(result, indent=2, default=str))


def disease_info(args):
    """Get disease information"""
    service = GeneralService()
    result = asyncio.run(service.get_disease_info(args.disease_name))
    print(json.dumps(result, indent=2, default=str))


def check_interactions(args):
    """Check drug interactions"""
    service = get_rule_engine_service()
    result = service.check_drug_interactions(args.drugs.split(","))
    print(json.dumps({
        "drugs": args.drugs.split(","),
        "interactions": result.interactions,
        "is_safe": result.is_safe,
        "warnings": result.warnings,
    }, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="IMI - Intelligent Medical Interface CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Symptom assessment
    assess_parser = subparsers.add_parser("assess", help="Assess symptoms")
    assess_parser.add_argument("--symptoms", "-s", required=True, help="Comma-separated symptoms")
    assess_parser.add_argument("--complaint", "-c", required=True, help="Chief complaint")
    assess_parser.add_argument("--age", "-a", type=int, required=True, help="Patient age")
    assess_parser.add_argument("--gender", "-g", default="unknown", help="Patient gender")
    assess_parser.add_argument("--conditions", help="Comma-separated medical conditions")
    assess_parser.add_argument("--medications", help="Comma-separated current medications")
    assess_parser.set_defaults(func=assess_symptoms)
    
    # Drug info
    drug_parser = subparsers.add_parser("drug", help="Get drug information")
    drug_parser.add_argument("drug_name", help="Name of the drug")
    drug_parser.set_defaults(func=drug_info)
    
    # Disease info
    disease_parser = subparsers.add_parser("disease", help="Get disease information")
    disease_parser.add_argument("disease_name", help="Name of the disease")
    disease_parser.set_defaults(func=disease_info)
    
    # Drug interactions
    interact_parser = subparsers.add_parser("interactions", help="Check drug interactions")
    interact_parser.add_argument("--drugs", "-d", required=True, help="Comma-separated drug names")
    interact_parser.set_defaults(func=check_interactions)
    
    args = parser.parse_args()
    
    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
