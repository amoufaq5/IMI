"""Domain-specific services for different user types"""
from .patient import PatientService
from .student import StudentService
from .doctor import DoctorService
from .researcher import ResearcherService
from .pharma import PharmaService
from .hospital import HospitalService
from .general import GeneralService

__all__ = [
    "PatientService",
    "StudentService",
    "DoctorService",
    "ResearcherService",
    "PharmaService",
    "HospitalService",
    "GeneralService",
]
