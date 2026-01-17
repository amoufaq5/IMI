"""
UMI Dosage Calculator Service
Provides weight/age-based dosing, renal/hepatic adjustments, and pediatric dosing
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import math


class AgeGroup(str, Enum):
    NEONATE = "neonate"  # 0-28 days
    INFANT = "infant"  # 1-12 months
    CHILD = "child"  # 1-12 years
    ADOLESCENT = "adolescent"  # 12-18 years
    ADULT = "adult"  # 18-65 years
    ELDERLY = "elderly"  # >65 years


class RenalFunction(str, Enum):
    NORMAL = "normal"  # CrCl >= 90
    MILD = "mild"  # CrCl 60-89
    MODERATE = "moderate"  # CrCl 30-59
    SEVERE = "severe"  # CrCl 15-29
    ESRD = "esrd"  # CrCl < 15 or dialysis


class HepaticFunction(str, Enum):
    NORMAL = "normal"
    MILD = "mild"  # Child-Pugh A
    MODERATE = "moderate"  # Child-Pugh B
    SEVERE = "severe"  # Child-Pugh C


@dataclass
class PatientParameters:
    """Patient parameters for dose calculation."""
    age_years: float
    weight_kg: float
    height_cm: Optional[float] = None
    sex: str = "unknown"  # male, female, unknown
    serum_creatinine: Optional[float] = None  # mg/dL
    is_dialysis: bool = False
    hepatic_function: HepaticFunction = HepaticFunction.NORMAL
    is_pregnant: bool = False
    is_lactating: bool = False
    allergies: List[str] = field(default_factory=list)


@dataclass
class DoseRecommendation:
    """Calculated dose recommendation."""
    drug_name: str
    recommended_dose: float
    dose_unit: str
    frequency: str
    route: str
    max_daily_dose: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    adjustments_applied: List[str] = field(default_factory=list)
    calculation_details: Dict[str, Any] = field(default_factory=dict)
    confidence: str = "standard"  # standard, requires_review, contraindicated


# Common drug dosing database
DRUG_DOSING_DB: Dict[str, Dict[str, Any]] = {
    "paracetamol": {
        "adult_dose": 500,
        "adult_max_single": 1000,
        "adult_max_daily": 4000,
        "pediatric_dose_per_kg": 15,
        "pediatric_max_single_per_kg": 15,
        "pediatric_max_daily_per_kg": 60,
        "unit": "mg",
        "frequency": "every 4-6 hours",
        "route": "oral",
        "renal_adjustment": False,
        "hepatic_adjustment": True,
        "hepatic_max_daily_moderate": 2000,
        "pregnancy_category": "B",
        "lactation_safe": True,
    },
    "ibuprofen": {
        "adult_dose": 400,
        "adult_max_single": 800,
        "adult_max_daily": 3200,
        "pediatric_dose_per_kg": 10,
        "pediatric_max_single_per_kg": 10,
        "pediatric_max_daily_per_kg": 40,
        "unit": "mg",
        "frequency": "every 6-8 hours",
        "route": "oral",
        "renal_adjustment": True,
        "renal_avoid_severe": True,
        "hepatic_adjustment": True,
        "pregnancy_category": "D",
        "pregnancy_avoid_third_trimester": True,
        "lactation_safe": True,
    },
    "amoxicillin": {
        "adult_dose": 500,
        "adult_max_single": 1000,
        "adult_max_daily": 3000,
        "pediatric_dose_per_kg": 25,
        "pediatric_max_single_per_kg": 50,
        "pediatric_max_daily_per_kg": 100,
        "unit": "mg",
        "frequency": "every 8 hours",
        "route": "oral",
        "renal_adjustment": True,
        "renal_dose_moderate": 0.5,  # 50% dose
        "renal_dose_severe": 0.5,
        "renal_frequency_severe": "every 12 hours",
        "hepatic_adjustment": False,
        "pregnancy_category": "B",
        "lactation_safe": True,
    },
    "metformin": {
        "adult_dose": 500,
        "adult_max_single": 1000,
        "adult_max_daily": 2550,
        "unit": "mg",
        "frequency": "twice daily with meals",
        "route": "oral",
        "renal_adjustment": True,
        "renal_max_moderate": 1000,
        "renal_avoid_severe": True,
        "hepatic_adjustment": True,
        "hepatic_avoid_severe": True,
        "pregnancy_category": "B",
        "lactation_safe": True,
        "min_age": 10,
    },
    "lisinopril": {
        "adult_dose": 10,
        "adult_max_single": 40,
        "adult_max_daily": 80,
        "unit": "mg",
        "frequency": "once daily",
        "route": "oral",
        "renal_adjustment": True,
        "renal_starting_dose_moderate": 5,
        "renal_starting_dose_severe": 2.5,
        "hepatic_adjustment": False,
        "pregnancy_category": "D",
        "pregnancy_contraindicated": True,
        "lactation_safe": False,
        "min_age": 6,
    },
    "omeprazole": {
        "adult_dose": 20,
        "adult_max_single": 40,
        "adult_max_daily": 40,
        "pediatric_dose_per_kg": 1,
        "pediatric_max_daily": 20,
        "unit": "mg",
        "frequency": "once daily before breakfast",
        "route": "oral",
        "renal_adjustment": False,
        "hepatic_adjustment": True,
        "hepatic_max_moderate": 20,
        "pregnancy_category": "C",
        "lactation_safe": False,
        "min_age": 1,
    },
    "atorvastatin": {
        "adult_dose": 20,
        "adult_max_single": 80,
        "adult_max_daily": 80,
        "unit": "mg",
        "frequency": "once daily",
        "route": "oral",
        "renal_adjustment": False,
        "hepatic_adjustment": True,
        "hepatic_contraindicated_severe": True,
        "pregnancy_category": "X",
        "pregnancy_contraindicated": True,
        "lactation_safe": False,
        "min_age": 10,
    },
    "amlodipine": {
        "adult_dose": 5,
        "adult_max_single": 10,
        "adult_max_daily": 10,
        "unit": "mg",
        "frequency": "once daily",
        "route": "oral",
        "renal_adjustment": False,
        "hepatic_adjustment": True,
        "hepatic_starting_dose": 2.5,
        "pregnancy_category": "C",
        "lactation_safe": True,
        "min_age": 6,
        "elderly_starting_dose": 2.5,
    },
    "metoprolol": {
        "adult_dose": 50,
        "adult_max_single": 200,
        "adult_max_daily": 400,
        "unit": "mg",
        "frequency": "twice daily",
        "route": "oral",
        "renal_adjustment": False,
        "hepatic_adjustment": True,
        "hepatic_reduce_severe": True,
        "pregnancy_category": "C",
        "lactation_safe": True,
        "min_age": 1,
    },
    "ciprofloxacin": {
        "adult_dose": 500,
        "adult_max_single": 750,
        "adult_max_daily": 1500,
        "pediatric_dose_per_kg": 15,
        "pediatric_max_daily_per_kg": 30,
        "unit": "mg",
        "frequency": "every 12 hours",
        "route": "oral",
        "renal_adjustment": True,
        "renal_dose_moderate": 0.75,
        "renal_dose_severe": 0.5,
        "hepatic_adjustment": False,
        "pregnancy_category": "C",
        "lactation_safe": False,
        "min_age": 1,
        "pediatric_caution": "Use only when no alternatives available",
    },
    "azithromycin": {
        "adult_dose": 500,
        "adult_max_single": 500,
        "adult_max_daily": 500,
        "pediatric_dose_per_kg": 10,
        "pediatric_max_single": 500,
        "unit": "mg",
        "frequency": "once daily for 3 days (or 500mg day 1, then 250mg days 2-5)",
        "route": "oral",
        "renal_adjustment": False,
        "hepatic_adjustment": True,
        "hepatic_caution_severe": True,
        "pregnancy_category": "B",
        "lactation_safe": True,
        "min_age": 0.5,
    },
    "prednisone": {
        "adult_dose": 20,
        "adult_max_single": 60,
        "adult_max_daily": 60,
        "pediatric_dose_per_kg": 1,
        "pediatric_max_daily_per_kg": 2,
        "unit": "mg",
        "frequency": "once daily in the morning",
        "route": "oral",
        "renal_adjustment": False,
        "hepatic_adjustment": False,
        "pregnancy_category": "C",
        "lactation_safe": True,
    },
}


class DosageCalculator:
    """
    Calculates medication dosages based on patient parameters.
    """
    
    def __init__(self):
        self.drug_db = DRUG_DOSING_DB
    
    def get_age_group(self, age_years: float) -> AgeGroup:
        """Determine age group from age in years."""
        if age_years < 0.077:  # ~28 days
            return AgeGroup.NEONATE
        elif age_years < 1:
            return AgeGroup.INFANT
        elif age_years < 12:
            return AgeGroup.CHILD
        elif age_years < 18:
            return AgeGroup.ADOLESCENT
        elif age_years < 65:
            return AgeGroup.ADULT
        else:
            return AgeGroup.ELDERLY
    
    def calculate_bsa(self, weight_kg: float, height_cm: float) -> float:
        """Calculate Body Surface Area using Mosteller formula."""
        return math.sqrt((height_cm * weight_kg) / 3600)
    
    def calculate_ibw(self, height_cm: float, sex: str) -> float:
        """Calculate Ideal Body Weight (Devine formula)."""
        height_inches = height_cm / 2.54
        if sex.lower() == "male":
            return 50 + 2.3 * (height_inches - 60)
        else:
            return 45.5 + 2.3 * (height_inches - 60)
    
    def calculate_creatinine_clearance(
        self,
        age_years: float,
        weight_kg: float,
        serum_creatinine: float,
        sex: str
    ) -> float:
        """Calculate CrCl using Cockcroft-Gault equation."""
        crcl = ((140 - age_years) * weight_kg) / (72 * serum_creatinine)
        if sex.lower() == "female":
            crcl *= 0.85
        return round(crcl, 1)
    
    def get_renal_function(self, crcl: float, is_dialysis: bool) -> RenalFunction:
        """Classify renal function based on CrCl."""
        if is_dialysis or crcl < 15:
            return RenalFunction.ESRD
        elif crcl < 30:
            return RenalFunction.SEVERE
        elif crcl < 60:
            return RenalFunction.MODERATE
        elif crcl < 90:
            return RenalFunction.MILD
        else:
            return RenalFunction.NORMAL
    
    def calculate_dose(
        self,
        drug_name: str,
        patient: PatientParameters,
        indication: Optional[str] = None
    ) -> DoseRecommendation:
        """
        Calculate recommended dose for a drug based on patient parameters.
        """
        drug_name_lower = drug_name.lower().strip()
        
        if drug_name_lower not in self.drug_db:
            return DoseRecommendation(
                drug_name=drug_name,
                recommended_dose=0,
                dose_unit="",
                frequency="",
                route="",
                warnings=["Drug not found in database. Please consult pharmacist."],
                confidence="requires_review"
            )
        
        drug = self.drug_db[drug_name_lower]
        warnings = []
        adjustments = []
        calculation_details = {}
        
        age_group = self.get_age_group(patient.age_years)
        calculation_details["age_group"] = age_group.value
        
        # Check minimum age
        if "min_age" in drug and patient.age_years < drug["min_age"]:
            return DoseRecommendation(
                drug_name=drug_name,
                recommended_dose=0,
                dose_unit=drug["unit"],
                frequency="",
                route=drug["route"],
                warnings=[f"Not recommended for patients under {drug['min_age']} years"],
                confidence="contraindicated"
            )
        
        # Calculate base dose
        if age_group in [AgeGroup.NEONATE, AgeGroup.INFANT, AgeGroup.CHILD]:
            # Pediatric dosing
            if "pediatric_dose_per_kg" in drug:
                base_dose = drug["pediatric_dose_per_kg"] * patient.weight_kg
                calculation_details["method"] = "weight-based"
                calculation_details["dose_per_kg"] = drug["pediatric_dose_per_kg"]
                
                # Apply max single dose
                if "pediatric_max_single_per_kg" in drug:
                    max_single = drug["pediatric_max_single_per_kg"] * patient.weight_kg
                    if "pediatric_max_single" in drug:
                        max_single = min(max_single, drug["pediatric_max_single"])
                    if base_dose > max_single:
                        base_dose = max_single
                        adjustments.append("Capped at maximum single dose")
                
                max_daily = None
                if "pediatric_max_daily_per_kg" in drug:
                    max_daily = drug["pediatric_max_daily_per_kg"] * patient.weight_kg
                if "pediatric_max_daily" in drug:
                    if max_daily:
                        max_daily = min(max_daily, drug["pediatric_max_daily"])
                    else:
                        max_daily = drug["pediatric_max_daily"]
                
                if "pediatric_caution" in drug:
                    warnings.append(drug["pediatric_caution"])
            else:
                # No pediatric dosing available
                warnings.append("Pediatric dosing not established. Consult specialist.")
                base_dose = 0
                max_daily = None
        else:
            # Adult dosing
            base_dose = drug["adult_dose"]
            max_daily = drug.get("adult_max_daily")
            calculation_details["method"] = "standard adult dose"
            
            # Elderly adjustment
            if age_group == AgeGroup.ELDERLY:
                if "elderly_starting_dose" in drug:
                    base_dose = drug["elderly_starting_dose"]
                    adjustments.append("Reduced starting dose for elderly")
                else:
                    warnings.append("Consider lower starting dose in elderly")
        
        # Calculate CrCl if serum creatinine provided
        renal_function = RenalFunction.NORMAL
        if patient.serum_creatinine and patient.serum_creatinine > 0:
            crcl = self.calculate_creatinine_clearance(
                patient.age_years,
                patient.weight_kg,
                patient.serum_creatinine,
                patient.sex
            )
            renal_function = self.get_renal_function(crcl, patient.is_dialysis)
            calculation_details["crcl"] = crcl
            calculation_details["renal_function"] = renal_function.value
        elif patient.is_dialysis:
            renal_function = RenalFunction.ESRD
            calculation_details["renal_function"] = "ESRD (dialysis)"
        
        # Apply renal adjustments
        frequency = drug["frequency"]
        if drug.get("renal_adjustment") and renal_function != RenalFunction.NORMAL:
            if drug.get("renal_avoid_severe") and renal_function in [RenalFunction.SEVERE, RenalFunction.ESRD]:
                return DoseRecommendation(
                    drug_name=drug_name,
                    recommended_dose=0,
                    dose_unit=drug["unit"],
                    frequency="",
                    route=drug["route"],
                    warnings=["Contraindicated in severe renal impairment"],
                    confidence="contraindicated"
                )
            
            if renal_function == RenalFunction.MODERATE:
                if "renal_dose_moderate" in drug:
                    base_dose *= drug["renal_dose_moderate"]
                    adjustments.append(f"Dose reduced for moderate renal impairment (CrCl 30-59)")
                if "renal_max_moderate" in drug:
                    max_daily = min(max_daily or float('inf'), drug["renal_max_moderate"])
                if "renal_starting_dose_moderate" in drug:
                    base_dose = drug["renal_starting_dose_moderate"]
                    adjustments.append("Starting dose adjusted for renal function")
            
            elif renal_function in [RenalFunction.SEVERE, RenalFunction.ESRD]:
                if "renal_dose_severe" in drug:
                    base_dose *= drug["renal_dose_severe"]
                    adjustments.append(f"Dose reduced for severe renal impairment")
                if "renal_frequency_severe" in drug:
                    frequency = drug["renal_frequency_severe"]
                    adjustments.append("Frequency adjusted for renal function")
                if "renal_starting_dose_severe" in drug:
                    base_dose = drug["renal_starting_dose_severe"]
                warnings.append("Monitor closely in severe renal impairment")
        
        # Apply hepatic adjustments
        if drug.get("hepatic_adjustment") and patient.hepatic_function != HepaticFunction.NORMAL:
            if drug.get("hepatic_contraindicated_severe") and patient.hepatic_function == HepaticFunction.SEVERE:
                return DoseRecommendation(
                    drug_name=drug_name,
                    recommended_dose=0,
                    dose_unit=drug["unit"],
                    frequency="",
                    route=drug["route"],
                    warnings=["Contraindicated in severe hepatic impairment"],
                    confidence="contraindicated"
                )
            
            if drug.get("hepatic_avoid_severe") and patient.hepatic_function == HepaticFunction.SEVERE:
                warnings.append("Avoid in severe hepatic impairment if possible")
            
            if patient.hepatic_function == HepaticFunction.MODERATE:
                if "hepatic_max_daily_moderate" in drug:
                    max_daily = min(max_daily or float('inf'), drug["hepatic_max_daily_moderate"])
                    adjustments.append("Maximum daily dose reduced for hepatic impairment")
                if "hepatic_max_moderate" in drug:
                    max_daily = min(max_daily or float('inf'), drug["hepatic_max_moderate"])
            
            if "hepatic_starting_dose" in drug:
                base_dose = drug["hepatic_starting_dose"]
                adjustments.append("Starting dose reduced for hepatic impairment")
            
            if drug.get("hepatic_caution_severe") and patient.hepatic_function == HepaticFunction.SEVERE:
                warnings.append("Use with caution in severe hepatic impairment")
            
            if drug.get("hepatic_reduce_severe") and patient.hepatic_function == HepaticFunction.SEVERE:
                base_dose *= 0.5
                adjustments.append("Dose reduced 50% for severe hepatic impairment")
        
        # Pregnancy checks
        if patient.is_pregnant:
            if drug.get("pregnancy_contraindicated"):
                return DoseRecommendation(
                    drug_name=drug_name,
                    recommended_dose=0,
                    dose_unit=drug["unit"],
                    frequency="",
                    route=drug["route"],
                    warnings=[f"Contraindicated in pregnancy (Category {drug.get('pregnancy_category', 'X')})"],
                    confidence="contraindicated"
                )
            
            if drug.get("pregnancy_avoid_third_trimester"):
                warnings.append("Avoid in third trimester of pregnancy")
            
            category = drug.get("pregnancy_category", "Unknown")
            if category in ["C", "D"]:
                warnings.append(f"Pregnancy Category {category}: Use only if benefit outweighs risk")
        
        # Lactation checks
        if patient.is_lactating:
            if not drug.get("lactation_safe", True):
                warnings.append("May not be safe during breastfeeding. Consider alternatives.")
        
        # Round dose appropriately
        if base_dose > 0:
            if drug["unit"] == "mg":
                if base_dose >= 100:
                    base_dose = round(base_dose / 50) * 50  # Round to nearest 50
                elif base_dose >= 10:
                    base_dose = round(base_dose / 5) * 5  # Round to nearest 5
                else:
                    base_dose = round(base_dose, 1)
        
        confidence = "standard"
        if warnings:
            confidence = "requires_review"
        
        return DoseRecommendation(
            drug_name=drug_name,
            recommended_dose=base_dose,
            dose_unit=drug["unit"],
            frequency=frequency,
            route=drug["route"],
            max_daily_dose=max_daily,
            warnings=warnings,
            adjustments_applied=adjustments,
            calculation_details=calculation_details,
            confidence=confidence
        )
    
    def get_available_drugs(self) -> List[str]:
        """Return list of drugs in the database."""
        return list(self.drug_db.keys())
    
    def check_dose_safety(
        self,
        drug_name: str,
        proposed_dose: float,
        patient: PatientParameters
    ) -> Dict[str, Any]:
        """
        Check if a proposed dose is within safe limits.
        """
        recommendation = self.calculate_dose(drug_name, patient)
        
        if recommendation.confidence == "contraindicated":
            return {
                "safe": False,
                "reason": "Drug is contraindicated for this patient",
                "warnings": recommendation.warnings
            }
        
        drug = self.drug_db.get(drug_name.lower())
        if not drug:
            return {
                "safe": None,
                "reason": "Drug not in database",
                "warnings": ["Unable to verify dose safety"]
            }
        
        max_single = drug.get("adult_max_single", float('inf'))
        age_group = self.get_age_group(patient.age_years)
        
        if age_group in [AgeGroup.CHILD, AgeGroup.INFANT, AgeGroup.NEONATE]:
            if "pediatric_max_single_per_kg" in drug:
                max_single = drug["pediatric_max_single_per_kg"] * patient.weight_kg
            if "pediatric_max_single" in drug:
                max_single = min(max_single, drug["pediatric_max_single"])
        
        if proposed_dose > max_single:
            return {
                "safe": False,
                "reason": f"Dose exceeds maximum single dose ({max_single} {drug['unit']})",
                "recommended_dose": recommendation.recommended_dose,
                "max_single_dose": max_single,
                "warnings": recommendation.warnings
            }
        
        return {
            "safe": True,
            "reason": "Dose is within safe limits",
            "recommended_dose": recommendation.recommended_dose,
            "max_single_dose": max_single,
            "warnings": recommendation.warnings
        }


# Singleton instance
_calculator = None


def get_dosage_calculator() -> DosageCalculator:
    """Get singleton instance of DosageCalculator."""
    global _calculator
    if _calculator is None:
        _calculator = DosageCalculator()
    return _calculator
