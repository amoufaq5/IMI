"""
UMI Clinical Services API
Endpoints for dosage calculation, lab interpretation, and ICD-10 coding
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from src.api.deps import get_current_user
from src.models.user import User
from src.services.dosage_calculator import (
    DosageCalculator,
    PatientParameters,
    HepaticFunction,
    get_dosage_calculator,
)
from src.services.lab_interpreter import (
    LabInterpreter,
    LabResult,
    get_lab_interpreter,
)
from src.services.icd10_service import (
    ICD10Service,
    get_icd10_service,
)

router = APIRouter(prefix="/clinical", tags=["Clinical Services"])


# ============== Dosage Calculator Schemas ==============

class PatientInfo(BaseModel):
    """Patient information for dose calculation."""
    age_years: float = Field(..., gt=0, description="Patient age in years")
    weight_kg: float = Field(..., gt=0, description="Patient weight in kg")
    height_cm: Optional[float] = Field(None, gt=0, description="Patient height in cm")
    sex: str = Field("unknown", description="Patient sex: male, female, or unknown")
    serum_creatinine: Optional[float] = Field(None, ge=0, description="Serum creatinine in mg/dL")
    is_dialysis: bool = Field(False, description="Is patient on dialysis")
    hepatic_function: str = Field("normal", description="Hepatic function: normal, mild, moderate, severe")
    is_pregnant: bool = Field(False, description="Is patient pregnant")
    is_lactating: bool = Field(False, description="Is patient breastfeeding")


class DoseCalculationRequest(BaseModel):
    """Request for dose calculation."""
    drug_name: str = Field(..., description="Name of the drug")
    patient: PatientInfo
    indication: Optional[str] = Field(None, description="Indication for use")


class DoseCheckRequest(BaseModel):
    """Request to check if a dose is safe."""
    drug_name: str = Field(..., description="Name of the drug")
    proposed_dose: float = Field(..., gt=0, description="Proposed dose amount")
    patient: PatientInfo


# ============== Lab Interpreter Schemas ==============

class LabResultInput(BaseModel):
    """A single lab result input."""
    test_name: str = Field(..., description="Name of the lab test")
    value: float = Field(..., description="Test result value")
    unit: Optional[str] = Field(None, description="Unit of measurement")


class LabInterpretRequest(BaseModel):
    """Request for lab interpretation."""
    results: List[LabResultInput] = Field(..., min_length=1, description="Lab results to interpret")
    sex: str = Field("unknown", description="Patient sex for reference ranges")
    age_years: Optional[float] = Field(None, description="Patient age in years")


class AnionGapRequest(BaseModel):
    """Request for anion gap calculation."""
    sodium: float = Field(..., description="Sodium level in mEq/L")
    chloride: float = Field(..., description="Chloride level in mEq/L")
    bicarbonate: float = Field(..., description="Bicarbonate level in mEq/L")


class CorrectedCalciumRequest(BaseModel):
    """Request for corrected calcium calculation."""
    total_calcium: float = Field(..., description="Total calcium in mg/dL")
    albumin: float = Field(..., description="Albumin in g/dL")


# ============== ICD-10 Schemas ==============

class ICD10SearchRequest(BaseModel):
    """Request for ICD-10 code search."""
    query: str = Field(..., min_length=2, description="Search query")
    limit: int = Field(10, ge=1, le=50, description="Maximum results to return")


class ICD10SuggestRequest(BaseModel):
    """Request for ICD-10 code suggestions."""
    symptoms: str = Field(..., description="Patient symptoms description")
    diagnoses: Optional[List[str]] = Field(None, description="Known diagnoses")


class EncounterCodingRequest(BaseModel):
    """Request for encounter coding."""
    chief_complaint: str = Field(..., description="Chief complaint")
    diagnoses: List[str] = Field(..., min_length=1, description="List of diagnoses")
    procedures: Optional[List[str]] = Field(None, description="Procedures performed")


# ============== Dosage Calculator Endpoints ==============

@router.post("/dosage/calculate")
async def calculate_dose(
    request: DoseCalculationRequest,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Calculate recommended medication dosage based on patient parameters.
    
    Considers:
    - Age and weight-based dosing
    - Renal function (CrCl calculation)
    - Hepatic function
    - Pregnancy and lactation
    - Elderly adjustments
    """
    calculator = get_dosage_calculator()
    
    # Convert hepatic function string to enum
    hepatic_map = {
        "normal": HepaticFunction.NORMAL,
        "mild": HepaticFunction.MILD,
        "moderate": HepaticFunction.MODERATE,
        "severe": HepaticFunction.SEVERE,
    }
    hepatic = hepatic_map.get(request.patient.hepatic_function.lower(), HepaticFunction.NORMAL)
    
    patient = PatientParameters(
        age_years=request.patient.age_years,
        weight_kg=request.patient.weight_kg,
        height_cm=request.patient.height_cm,
        sex=request.patient.sex,
        serum_creatinine=request.patient.serum_creatinine,
        is_dialysis=request.patient.is_dialysis,
        hepatic_function=hepatic,
        is_pregnant=request.patient.is_pregnant,
        is_lactating=request.patient.is_lactating,
    )
    
    result = calculator.calculate_dose(
        drug_name=request.drug_name,
        patient=patient,
        indication=request.indication,
    )
    
    return {
        "drug_name": result.drug_name,
        "recommended_dose": result.recommended_dose,
        "dose_unit": result.dose_unit,
        "frequency": result.frequency,
        "route": result.route,
        "max_daily_dose": result.max_daily_dose,
        "warnings": result.warnings,
        "adjustments_applied": result.adjustments_applied,
        "calculation_details": result.calculation_details,
        "confidence": result.confidence,
        "disclaimer": "This is a clinical decision support tool. Always verify dosing with official references and clinical judgment.",
    }


@router.post("/dosage/check")
async def check_dose_safety(
    request: DoseCheckRequest,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Check if a proposed dose is within safe limits for a patient.
    """
    calculator = get_dosage_calculator()
    
    hepatic_map = {
        "normal": HepaticFunction.NORMAL,
        "mild": HepaticFunction.MILD,
        "moderate": HepaticFunction.MODERATE,
        "severe": HepaticFunction.SEVERE,
    }
    hepatic = hepatic_map.get(request.patient.hepatic_function.lower(), HepaticFunction.NORMAL)
    
    patient = PatientParameters(
        age_years=request.patient.age_years,
        weight_kg=request.patient.weight_kg,
        height_cm=request.patient.height_cm,
        sex=request.patient.sex,
        serum_creatinine=request.patient.serum_creatinine,
        is_dialysis=request.patient.is_dialysis,
        hepatic_function=hepatic,
        is_pregnant=request.patient.is_pregnant,
        is_lactating=request.patient.is_lactating,
    )
    
    result = calculator.check_dose_safety(
        drug_name=request.drug_name,
        proposed_dose=request.proposed_dose,
        patient=patient,
    )
    
    return result


@router.get("/dosage/drugs")
async def list_available_drugs(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    List all drugs available in the dosage calculator database.
    """
    calculator = get_dosage_calculator()
    drugs = calculator.get_available_drugs()
    
    return {
        "count": len(drugs),
        "drugs": sorted(drugs),
    }


# ============== Lab Interpreter Endpoints ==============

@router.post("/labs/interpret")
async def interpret_lab_results(
    request: LabInterpretRequest,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Interpret laboratory test results and provide clinical guidance.
    
    Returns:
    - Status (normal, low, high, critical)
    - Urgency level
    - Possible causes
    - Recommended actions
    - Related tests to consider
    """
    interpreter = get_lab_interpreter()
    
    lab_results = [
        LabResult(
            test_name=r.test_name,
            value=r.value,
            unit=r.unit or "",
        )
        for r in request.results
    ]
    
    result = interpreter.interpret_panel(
        results=lab_results,
        sex=request.sex,
        age_years=request.age_years,
    )
    
    # Convert interpretations to dict format
    interpretations = []
    for interp in result["interpretations"]:
        interpretations.append({
            "test_name": interp.test_name,
            "value": interp.value,
            "unit": interp.unit,
            "status": interp.status.value,
            "urgency": interp.urgency.value,
            "reference_range": interp.reference_range,
            "interpretation": interp.interpretation,
            "possible_causes": interp.possible_causes,
            "recommended_actions": interp.recommended_actions,
            "related_tests": interp.related_tests,
        })
    
    critical = [
        {"test_name": i.test_name, "value": i.value, "status": i.status.value}
        for i in result["critical_findings"]
    ]
    
    abnormal = [
        {"test_name": i.test_name, "value": i.value, "status": i.status.value}
        for i in result["abnormal_findings"]
    ]
    
    return {
        "interpretations": interpretations,
        "critical_findings": critical,
        "abnormal_findings": abnormal,
        "overall_urgency": result["overall_urgency"].value,
        "summary": result["summary"],
        "disclaimer": "Lab interpretation is for clinical decision support. Always correlate with clinical presentation.",
    }


@router.post("/labs/interpret/single")
async def interpret_single_lab(
    test_name: str,
    value: float,
    sex: str = "unknown",
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Interpret a single lab test result.
    """
    interpreter = get_lab_interpreter()
    
    interp = interpreter.interpret(
        test_name=test_name,
        value=value,
        sex=sex,
    )
    
    return {
        "test_name": interp.test_name,
        "value": interp.value,
        "unit": interp.unit,
        "status": interp.status.value,
        "urgency": interp.urgency.value,
        "reference_range": interp.reference_range,
        "interpretation": interp.interpretation,
        "possible_causes": interp.possible_causes,
        "recommended_actions": interp.recommended_actions,
        "related_tests": interp.related_tests,
    }


@router.post("/labs/anion-gap")
async def calculate_anion_gap(
    request: AnionGapRequest,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Calculate anion gap from electrolytes.
    
    Formula: Anion Gap = Na - (Cl + HCO3)
    Normal range: 8-12 mEq/L
    """
    interpreter = get_lab_interpreter()
    
    return interpreter.calculate_anion_gap(
        sodium=request.sodium,
        chloride=request.chloride,
        bicarbonate=request.bicarbonate,
    )


@router.post("/labs/corrected-calcium")
async def calculate_corrected_calcium(
    request: CorrectedCalciumRequest,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Calculate albumin-corrected calcium.
    
    Formula: Corrected Ca = Total Ca + 0.8 * (4.0 - Albumin)
    """
    interpreter = get_lab_interpreter()
    
    return interpreter.calculate_corrected_calcium(
        total_calcium=request.total_calcium,
        albumin=request.albumin,
    )


@router.get("/labs/tests")
async def list_available_tests(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    List all lab tests available in the interpreter database.
    """
    interpreter = get_lab_interpreter()
    tests = interpreter.get_available_tests()
    
    return {
        "count": len(tests),
        "tests": sorted(tests),
    }


# ============== ICD-10 Coding Endpoints ==============

@router.get("/icd10/lookup/{code}")
async def lookup_icd10_code(
    code: str,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Look up an ICD-10 code and return its details.
    """
    service = get_icd10_service()
    result = service.lookup_code(code)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ICD-10 code '{code}' not found",
        )
    
    return {
        "code": result.code,
        "description": result.description,
        "category": result.category,
        "chapter": result.chapter,
        "is_billable": result.is_billable,
    }


@router.post("/icd10/search")
async def search_icd10_codes(
    request: ICD10SearchRequest,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Search for ICD-10 codes by description or code.
    """
    service = get_icd10_service()
    results = service.search_codes(request.query, request.limit)
    
    return {
        "query": request.query,
        "count": len(results),
        "results": [
            {
                "code": r.code,
                "description": r.description,
                "category": r.category,
            }
            for r in results
        ],
    }


@router.post("/icd10/suggest")
async def suggest_icd10_codes(
    request: ICD10SuggestRequest,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Suggest ICD-10 codes based on symptoms and diagnoses.
    
    Uses AI-assisted mapping to recommend appropriate codes.
    """
    service = get_icd10_service()
    suggestions = service.suggest_codes(
        symptoms=request.symptoms,
        diagnoses=request.diagnoses,
    )
    
    return {
        "suggestions": [
            {
                "code": s.code,
                "description": s.description,
                "confidence": round(s.confidence, 2),
                "rationale": s.rationale,
                "alternatives": [{"code": c, "description": d} for c, d in s.alternatives],
            }
            for s in suggestions
        ],
        "disclaimer": "Code suggestions are AI-assisted. Verify accuracy before use in billing.",
    }


@router.get("/icd10/validate/{code}")
async def validate_icd10_code(
    code: str,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Validate an ICD-10 code format and check if it exists.
    """
    service = get_icd10_service()
    return service.validate_code(code)


@router.post("/icd10/encode-encounter")
async def encode_clinical_encounter(
    request: EncounterCodingRequest,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Generate ICD-10 codes for a complete clinical encounter.
    
    Provides primary and secondary diagnosis codes based on
    chief complaint and diagnoses.
    """
    service = get_icd10_service()
    
    return service.encode_encounter(
        chief_complaint=request.chief_complaint,
        diagnoses=request.diagnoses,
        procedures=request.procedures,
    )


@router.get("/icd10/related/{code}")
async def get_related_codes(
    code: str,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get ICD-10 codes related to a given code.
    """
    service = get_icd10_service()
    results = service.get_related_codes(code)
    
    return {
        "base_code": code,
        "related_codes": [
            {
                "code": r.code,
                "description": r.description,
            }
            for r in results
        ],
    }


@router.get("/icd10/categories")
async def list_icd10_categories(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    List all available ICD-10 categories.
    """
    service = get_icd10_service()
    categories = service.get_available_categories()
    
    return {
        "count": len(categories),
        "categories": categories,
    }
