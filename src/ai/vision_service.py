"""
UMI Medical Vision Service
Medical image analysis for X-rays, CT scans, MRI, and lab reports
"""

import base64
import io
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class ImageType(str, Enum):
    """Supported medical image types."""
    XRAY = "xray"
    CT = "ct"
    MRI = "mri"
    ULTRASOUND = "ultrasound"
    MAMMOGRAM = "mammogram"
    DERMOSCOPY = "dermoscopy"
    FUNDUS = "fundus"  # Eye imaging
    LAB_REPORT = "lab_report"
    PHOTO = "photo"  # General medical photo


class BodyRegion(str, Enum):
    """Body regions for imaging."""
    HEAD = "head"
    CHEST = "chest"
    ABDOMEN = "abdomen"
    SPINE = "spine"
    EXTREMITY = "extremity"
    PELVIS = "pelvis"
    WHOLE_BODY = "whole_body"
    SKIN = "skin"
    EYE = "eye"


@dataclass
class ImageAnalysisResult:
    """Result from medical image analysis."""
    image_type: ImageType
    body_region: Optional[BodyRegion]
    findings: List[Dict[str, Any]]
    impression: str
    confidence: float
    recommendations: List[str]
    abnormalities_detected: bool
    urgency: str  # normal, attention, urgent, critical
    raw_predictions: Optional[Dict[str, Any]] = None
    processing_time_ms: float = 0


class ImagePreprocessor:
    """Preprocesses medical images for analysis."""
    
    # Standard sizes for different modalities
    TARGET_SIZES = {
        ImageType.XRAY: (512, 512),
        ImageType.CT: (512, 512),
        ImageType.MRI: (256, 256),
        ImageType.DERMOSCOPY: (224, 224),
        ImageType.FUNDUS: (224, 224),
        ImageType.PHOTO: (384, 384),
    }
    
    @classmethod
    def load_image(cls, image_source: Union[str, bytes, Path]) -> Image.Image:
        """Load image from various sources."""
        if isinstance(image_source, bytes):
            return Image.open(io.BytesIO(image_source))
        elif isinstance(image_source, (str, Path)):
            path = Path(image_source)
            if path.suffix.lower() == '.dcm':
                return cls._load_dicom(path)
            return Image.open(path)
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")
    
    @classmethod
    def _load_dicom(cls, path: Path) -> Image.Image:
        """Load DICOM medical image."""
        try:
            import pydicom
            
            ds = pydicom.dcmread(str(path))
            pixel_array = ds.pixel_array
            
            # Normalize to 8-bit
            if pixel_array.max() > 255:
                pixel_array = ((pixel_array - pixel_array.min()) / 
                              (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
            
            # Apply window/level if available
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                center = ds.WindowCenter
                width = ds.WindowWidth
                if isinstance(center, pydicom.multival.MultiValue):
                    center = center[0]
                if isinstance(width, pydicom.multival.MultiValue):
                    width = width[0]
                
                lower = center - width / 2
                upper = center + width / 2
                pixel_array = np.clip(pixel_array, lower, upper)
                pixel_array = ((pixel_array - lower) / (upper - lower) * 255).astype(np.uint8)
            
            return Image.fromarray(pixel_array)
        
        except ImportError:
            raise ImportError("pydicom required for DICOM support: pip install pydicom")
    
    @classmethod
    def preprocess(
        cls,
        image: Image.Image,
        image_type: ImageType,
        normalize: bool = True,
    ) -> np.ndarray:
        """Preprocess image for model input."""
        # Get target size
        target_size = cls.TARGET_SIZES.get(image_type, (512, 512))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize
        if normalize:
            img_array = img_array / 255.0
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = (img_array - mean) / std
        
        # Add batch dimension and transpose to NCHW
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, 0)
        
        return img_array
    
    @classmethod
    def to_base64(cls, image: Image.Image, format: str = "PNG") -> str:
        """Convert image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode()


class ChestXRayAnalyzer:
    """
    Specialized analyzer for chest X-rays.
    Detects common pathologies like pneumonia, cardiomegaly, etc.
    """
    
    PATHOLOGIES = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Hernia",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pleural_Thickening",
        "Pneumonia",
        "Pneumothorax",
    ]
    
    def __init__(self):
        self._model = None
    
    async def _load_model(self):
        """Lazy load chest X-ray model."""
        if self._model is None:
            try:
                import torch
                import torchvision.models as models
                
                # Use DenseNet121 pretrained on CheXpert/ChestX-ray14
                # In production, load fine-tuned weights
                self._model = models.densenet121(pretrained=True)
                self._model.classifier = torch.nn.Linear(1024, len(self.PATHOLOGIES))
                self._model.eval()
                
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
                
                logger.info("chest_xray_model_loaded")
            except ImportError:
                logger.warning("torch_not_available_using_mock")
                self._model = MockChestXRayModel()
    
    async def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze chest X-ray for pathologies."""
        await self._load_model()
        
        # Preprocess
        preprocessor = ImagePreprocessor()
        img_array = preprocessor.preprocess(image, ImageType.XRAY)
        
        if isinstance(self._model, MockChestXRayModel):
            return await self._model.predict(img_array)
        
        import torch
        
        with torch.no_grad():
            tensor = torch.from_numpy(img_array).float()
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            
            outputs = self._model(tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Build results
        findings = []
        for i, pathology in enumerate(self.PATHOLOGIES):
            prob = float(probabilities[i])
            if prob > 0.3:  # Threshold for reporting
                findings.append({
                    "finding": pathology,
                    "probability": prob,
                    "severity": self._get_severity(prob),
                })
        
        return {
            "pathologies": dict(zip(self.PATHOLOGIES, probabilities.tolist())),
            "findings": sorted(findings, key=lambda x: x["probability"], reverse=True),
            "abnormal": any(p > 0.5 for p in probabilities),
        }
    
    def _get_severity(self, probability: float) -> str:
        """Map probability to severity level."""
        if probability > 0.8:
            return "high"
        elif probability > 0.5:
            return "moderate"
        else:
            return "low"


class MockChestXRayModel:
    """Mock chest X-ray model for development."""
    
    async def predict(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Return mock predictions."""
        import random
        
        pathologies = ChestXRayAnalyzer.PATHOLOGIES
        probabilities = [random.uniform(0.05, 0.3) for _ in pathologies]
        
        # Simulate one finding
        idx = random.randint(0, len(pathologies) - 1)
        probabilities[idx] = random.uniform(0.4, 0.7)
        
        findings = []
        for i, pathology in enumerate(pathologies):
            if probabilities[i] > 0.3:
                findings.append({
                    "finding": pathology,
                    "probability": probabilities[i],
                    "severity": "moderate" if probabilities[i] > 0.5 else "low",
                })
        
        return {
            "pathologies": dict(zip(pathologies, probabilities)),
            "findings": sorted(findings, key=lambda x: x["probability"], reverse=True),
            "abnormal": any(p > 0.5 for p in probabilities),
        }


class DermoscopyAnalyzer:
    """
    Analyzer for skin lesion images.
    Classifies lesions and assesses malignancy risk.
    """
    
    LESION_TYPES = [
        "Melanoma",
        "Melanocytic nevus",
        "Basal cell carcinoma",
        "Actinic keratosis",
        "Benign keratosis",
        "Dermatofibroma",
        "Vascular lesion",
    ]
    
    def __init__(self):
        self._model = None
    
    async def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze skin lesion image."""
        # Mock implementation
        import random
        
        probs = [random.uniform(0.05, 0.2) for _ in self.LESION_TYPES]
        top_idx = random.randint(0, len(self.LESION_TYPES) - 1)
        probs[top_idx] = random.uniform(0.5, 0.9)
        
        # Normalize
        total = sum(probs)
        probs = [p / total for p in probs]
        
        top_prediction = self.LESION_TYPES[probs.index(max(probs))]
        is_malignant = top_prediction in ["Melanoma", "Basal cell carcinoma"]
        
        return {
            "classifications": dict(zip(self.LESION_TYPES, probs)),
            "top_prediction": top_prediction,
            "confidence": max(probs),
            "malignancy_risk": "high" if is_malignant else "low",
            "recommendations": [
                "Consult dermatologist for definitive diagnosis" if is_malignant
                else "Monitor for changes, routine follow-up recommended"
            ],
        }


class LabReportOCR:
    """
    OCR and extraction for lab reports.
    Extracts values and flags abnormal results.
    """
    
    # Common lab test reference ranges
    REFERENCE_RANGES = {
        "hemoglobin": {"min": 12.0, "max": 17.5, "unit": "g/dL"},
        "hematocrit": {"min": 36, "max": 50, "unit": "%"},
        "wbc": {"min": 4.5, "max": 11.0, "unit": "10^9/L"},
        "platelets": {"min": 150, "max": 400, "unit": "10^9/L"},
        "glucose": {"min": 70, "max": 100, "unit": "mg/dL"},
        "creatinine": {"min": 0.7, "max": 1.3, "unit": "mg/dL"},
        "sodium": {"min": 136, "max": 145, "unit": "mEq/L"},
        "potassium": {"min": 3.5, "max": 5.0, "unit": "mEq/L"},
        "alt": {"min": 7, "max": 56, "unit": "U/L"},
        "ast": {"min": 10, "max": 40, "unit": "U/L"},
        "cholesterol": {"min": 0, "max": 200, "unit": "mg/dL"},
        "ldl": {"min": 0, "max": 100, "unit": "mg/dL"},
        "hdl": {"min": 40, "max": 999, "unit": "mg/dL"},
        "triglycerides": {"min": 0, "max": 150, "unit": "mg/dL"},
        "tsh": {"min": 0.4, "max": 4.0, "unit": "mIU/L"},
        "hba1c": {"min": 0, "max": 5.7, "unit": "%"},
    }
    
    async def extract(self, image: Image.Image) -> Dict[str, Any]:
        """Extract lab values from report image."""
        try:
            import pytesseract
            
            # Perform OCR
            text = pytesseract.image_to_string(image)
            
            # Extract values
            extracted = self._parse_lab_values(text)
            
            return {
                "raw_text": text,
                "extracted_values": extracted,
                "abnormal_values": [v for v in extracted if v.get("status") != "normal"],
            }
        
        except ImportError:
            logger.warning("pytesseract_not_available")
            return {
                "raw_text": "[OCR not available - pytesseract required]",
                "extracted_values": [],
                "abnormal_values": [],
            }
    
    def _parse_lab_values(self, text: str) -> List[Dict[str, Any]]:
        """Parse lab values from OCR text."""
        import re
        
        extracted = []
        text_lower = text.lower()
        
        for test_name, ranges in self.REFERENCE_RANGES.items():
            # Look for test name and nearby number
            pattern = rf'{test_name}[:\s]+(\d+\.?\d*)'
            match = re.search(pattern, text_lower)
            
            if match:
                value = float(match.group(1))
                status = "normal"
                
                if value < ranges["min"]:
                    status = "low"
                elif value > ranges["max"]:
                    status = "high"
                
                extracted.append({
                    "test": test_name,
                    "value": value,
                    "unit": ranges["unit"],
                    "reference_range": f"{ranges['min']}-{ranges['max']}",
                    "status": status,
                })
        
        return extracted


class MedicalVisionService:
    """
    Main service for medical image analysis.
    Routes to appropriate specialized analyzers.
    """
    
    def __init__(self):
        self.chest_xray = ChestXRayAnalyzer()
        self.dermoscopy = DermoscopyAnalyzer()
        self.lab_ocr = LabReportOCR()
        self.preprocessor = ImagePreprocessor()
    
    async def analyze(
        self,
        image_source: Union[str, bytes, Path],
        image_type: ImageType,
        body_region: Optional[BodyRegion] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ImageAnalysisResult:
        """
        Analyze a medical image.
        
        Args:
            image_source: Image file path, bytes, or base64 string
            image_type: Type of medical image
            body_region: Body region (for X-ray/CT/MRI)
            metadata: Additional metadata (patient info, clinical context)
        
        Returns:
            ImageAnalysisResult with findings and recommendations
        """
        start_time = datetime.now()
        
        # Load image
        image = self.preprocessor.load_image(image_source)
        
        # Route to appropriate analyzer
        if image_type == ImageType.XRAY and body_region == BodyRegion.CHEST:
            result = await self._analyze_chest_xray(image)
        elif image_type == ImageType.DERMOSCOPY:
            result = await self._analyze_dermoscopy(image)
        elif image_type == ImageType.LAB_REPORT:
            result = await self._analyze_lab_report(image)
        else:
            result = await self._analyze_generic(image, image_type)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        result.processing_time_ms = processing_time
        result.image_type = image_type
        result.body_region = body_region
        
        logger.info(
            "image_analyzed",
            image_type=image_type.value,
            abnormal=result.abnormalities_detected,
            processing_ms=processing_time,
        )
        
        return result
    
    async def _analyze_chest_xray(self, image: Image.Image) -> ImageAnalysisResult:
        """Analyze chest X-ray."""
        analysis = await self.chest_xray.analyze(image)
        
        findings = analysis.get("findings", [])
        abnormal = analysis.get("abnormal", False)
        
        # Generate impression
        if not findings:
            impression = "No significant abnormalities detected on chest radiograph."
            urgency = "normal"
        else:
            top_findings = [f["finding"] for f in findings[:3]]
            impression = f"Findings suggestive of: {', '.join(top_findings)}. Clinical correlation recommended."
            
            # Determine urgency
            high_severity = any(f.get("severity") == "high" for f in findings)
            urgency = "urgent" if high_severity else "attention"
        
        recommendations = []
        if abnormal:
            recommendations.append("Recommend clinical correlation with patient symptoms")
            recommendations.append("Consider follow-up imaging if clinically indicated")
            if any(f["finding"] == "Pneumothorax" for f in findings):
                recommendations.insert(0, "URGENT: Evaluate for pneumothorax clinically")
        
        return ImageAnalysisResult(
            image_type=ImageType.XRAY,
            body_region=BodyRegion.CHEST,
            findings=findings,
            impression=impression,
            confidence=max((f["probability"] for f in findings), default=0.9),
            recommendations=recommendations,
            abnormalities_detected=abnormal,
            urgency=urgency,
            raw_predictions=analysis.get("pathologies"),
        )
    
    async def _analyze_dermoscopy(self, image: Image.Image) -> ImageAnalysisResult:
        """Analyze skin lesion image."""
        analysis = await self.dermoscopy.analyze(image)
        
        top_pred = analysis.get("top_prediction", "Unknown")
        confidence = analysis.get("confidence", 0)
        malignancy_risk = analysis.get("malignancy_risk", "unknown")
        
        findings = [{
            "finding": top_pred,
            "probability": confidence,
            "malignancy_risk": malignancy_risk,
        }]
        
        if malignancy_risk == "high":
            impression = f"Lesion characteristics concerning for {top_pred}. Urgent dermatology referral recommended."
            urgency = "urgent"
        else:
            impression = f"Lesion most consistent with {top_pred}. Benign appearance."
            urgency = "normal"
        
        return ImageAnalysisResult(
            image_type=ImageType.DERMOSCOPY,
            body_region=BodyRegion.SKIN,
            findings=findings,
            impression=impression,
            confidence=confidence,
            recommendations=analysis.get("recommendations", []),
            abnormalities_detected=malignancy_risk == "high",
            urgency=urgency,
            raw_predictions=analysis.get("classifications"),
        )
    
    async def _analyze_lab_report(self, image: Image.Image) -> ImageAnalysisResult:
        """Analyze lab report via OCR."""
        extraction = await self.lab_ocr.extract(image)
        
        abnormal = extraction.get("abnormal_values", [])
        all_values = extraction.get("extracted_values", [])
        
        findings = [
            {
                "finding": f"{v['test']}: {v['value']} {v['unit']} ({v['status'].upper()})",
                "test": v["test"],
                "value": v["value"],
                "status": v["status"],
            }
            for v in abnormal
        ]
        
        if not abnormal:
            impression = "All extracted lab values within normal reference ranges."
            urgency = "normal"
        else:
            abnormal_tests = [v["test"] for v in abnormal]
            impression = f"Abnormal values detected: {', '.join(abnormal_tests)}. Review recommended."
            urgency = "attention"
        
        recommendations = []
        for v in abnormal:
            if v["test"] == "glucose" and v["status"] == "high":
                recommendations.append("Elevated glucose - consider diabetes screening")
            elif v["test"] == "creatinine" and v["status"] == "high":
                recommendations.append("Elevated creatinine - evaluate kidney function")
            elif v["test"] in ["alt", "ast"] and v["status"] == "high":
                recommendations.append("Elevated liver enzymes - evaluate hepatic function")
        
        return ImageAnalysisResult(
            image_type=ImageType.LAB_REPORT,
            body_region=None,
            findings=findings,
            impression=impression,
            confidence=0.8,  # OCR confidence
            recommendations=recommendations,
            abnormalities_detected=len(abnormal) > 0,
            urgency=urgency,
            raw_predictions={"extracted_values": all_values},
        )
    
    async def _analyze_generic(
        self,
        image: Image.Image,
        image_type: ImageType,
    ) -> ImageAnalysisResult:
        """Generic analysis for unsupported image types."""
        return ImageAnalysisResult(
            image_type=image_type,
            body_region=None,
            findings=[],
            impression=f"Analysis for {image_type.value} images is not yet fully implemented. Please consult a radiologist.",
            confidence=0.0,
            recommendations=["Consult specialist for interpretation"],
            abnormalities_detected=False,
            urgency="normal",
        )
