"""
UMI AI Services Tests
Tests for LLM, RAG, and Medical NLP services
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestLLMService:
    """Tests for LLM Service."""
    
    @pytest.mark.asyncio
    async def test_expert_routing_diagnosis(self):
        """Test that diagnosis queries route to diagnosis expert."""
        from src.ai.llm_service import ExpertRouter, ExpertType
        
        router = ExpertRouter()
        expert, confidence = router.route("I have chest pain and shortness of breath")
        
        assert expert == ExpertType.DIAGNOSIS
        assert confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_expert_routing_drug(self):
        """Test that drug queries route to drug expert."""
        from src.ai.llm_service import ExpertRouter, ExpertType
        
        router = ExpertRouter()
        expert, confidence = router.route("What is the dosage for ibuprofen?")
        
        assert expert == ExpertType.DRUG
        assert confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_expert_routing_pharma(self):
        """Test that QA/QC queries route to pharma expert."""
        from src.ai.llm_service import ExpertRouter, ExpertType
        
        router = ExpertRouter()
        expert, confidence = router.route("Generate a cleaning validation protocol for tablet press")
        
        assert expert == ExpertType.QA_QC
        assert confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_llm_service_generate_mock(self):
        """Test LLM generation with mock client."""
        from src.ai.llm_service import LLMService
        
        service = LLMService()
        
        # Should work with mock client when no API key
        response = await service.generate(
            query="What are symptoms of flu?",
            context={"mode": "general"},
        )
        
        assert response is not None
        assert hasattr(response, 'content')
        assert len(response.content) > 0
    
    @pytest.mark.asyncio
    async def test_chain_of_thought_reasoning(self):
        """Test Chain-of-Thought reasoning format."""
        from src.ai.llm_service import ReasoningEngine, ReasoningMethod
        
        engine = ReasoningEngine()
        
        prompt = engine.apply_reasoning(
            query="Diagnose: 45yo male with chest pain radiating to left arm",
            method=ReasoningMethod.CHAIN_OF_THOUGHT,
        )
        
        assert "step by step" in prompt.lower() or "reasoning" in prompt.lower()


class TestRAGService:
    """Tests for RAG Service."""
    
    @pytest.mark.asyncio
    async def test_rag_service_initialization(self):
        """Test RAG service can be initialized."""
        from src.ai.rag_service import RAGService
        
        service = RAGService()
        assert service is not None
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self):
        """Test embedding generation."""
        from src.ai.rag_service import RAGService
        
        service = RAGService()
        
        # Should work with mock embeddings
        embedding = await service.embed_text("Test medical query")
        
        assert embedding is not None
        assert len(embedding) > 0
    
    @pytest.mark.asyncio
    async def test_retrieve_documents(self):
        """Test document retrieval."""
        from src.ai.rag_service import RAGService
        
        service = RAGService()
        
        result = await service.retrieve(
            query="diabetes treatment guidelines",
            collections=["medical_literature"],
            top_k=5,
        )
        
        assert result is not None
        assert hasattr(result, 'documents')
    
    @pytest.mark.asyncio
    async def test_index_document(self):
        """Test document indexing."""
        from src.ai.rag_service import RAGService
        
        service = RAGService()
        
        success = await service.index_document(
            collection="test_collection",
            doc_id="test_doc_1",
            title="Test Document",
            content="This is test content about diabetes.",
            metadata={"source": "test"},
        )
        
        # Should succeed or fail gracefully
        assert isinstance(success, bool)


class TestMedicalNLP:
    """Tests for Medical NLP Service."""
    
    @pytest.mark.asyncio
    async def test_entity_extraction(self):
        """Test medical entity extraction."""
        from src.ai.medical_nlp import MedicalNLPService
        
        service = MedicalNLPService()
        
        text = "Patient has severe headache and fever for 3 days. Taking paracetamol."
        entities = await service.extract_entities(text)
        
        assert entities is not None
        assert "symptoms" in entities or "medications" in entities
    
    @pytest.mark.asyncio
    async def test_symptom_extraction(self):
        """Test symptom extraction."""
        from src.ai.medical_nlp import MedicalNLPService
        
        service = MedicalNLPService()
        
        text = "I have a headache, nausea, and dizziness"
        result = await service.extract_entities(text)
        
        symptoms = result.get("symptoms", [])
        assert len(symptoms) > 0
    
    @pytest.mark.asyncio
    async def test_negation_detection(self):
        """Test negation detection."""
        from src.ai.medical_nlp import MedicalNLPService
        
        service = MedicalNLPService()
        
        text = "No fever, no cough, but has headache"
        result = await service.extract_entities(text)
        
        # Should detect negated symptoms
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_severity_assessment(self):
        """Test severity assessment."""
        from src.ai.medical_nlp import MedicalNLPService
        
        service = MedicalNLPService()
        
        text = "Severe chest pain with mild nausea"
        result = await service.analyze_severity(text)
        
        assert result is not None
        assert "severity" in result or "level" in result
    
    @pytest.mark.asyncio
    async def test_temporal_extraction(self):
        """Test temporal information extraction."""
        from src.ai.medical_nlp import MedicalNLPService
        
        service = MedicalNLPService()
        
        text = "Headache started 3 days ago, getting worse since yesterday"
        result = await service.extract_temporal(text)
        
        assert result is not None


class TestVisionService:
    """Tests for Vision Service."""
    
    @pytest.mark.asyncio
    async def test_vision_service_initialization(self):
        """Test vision service can be initialized."""
        from src.ai.vision_service import MedicalVisionService
        
        service = MedicalVisionService()
        assert service is not None
    
    @pytest.mark.asyncio
    async def test_image_preprocessor(self):
        """Test image preprocessing."""
        from src.ai.vision_service import ImagePreprocessor, ImageType
        from PIL import Image
        import io
        
        # Create test image
        img = Image.new('RGB', (512, 512), color='gray')
        
        preprocessor = ImagePreprocessor()
        processed = preprocessor.preprocess(img, ImageType.XRAY)
        
        assert processed is not None
        assert processed.shape[0] == 1  # Batch dimension
        assert processed.shape[1] == 3  # Channels
    
    @pytest.mark.asyncio
    async def test_chest_xray_analyzer(self):
        """Test chest X-ray analyzer."""
        from src.ai.vision_service import ChestXRayAnalyzer
        from PIL import Image
        
        analyzer = ChestXRayAnalyzer()
        
        # Create test image
        img = Image.new('RGB', (512, 512), color='gray')
        
        result = await analyzer.analyze(img)
        
        assert result is not None
        assert "pathologies" in result or "findings" in result
    
    @pytest.mark.asyncio
    async def test_dermoscopy_analyzer(self):
        """Test dermoscopy analyzer."""
        from src.ai.vision_service import DermoscopyAnalyzer
        from PIL import Image
        
        analyzer = DermoscopyAnalyzer()
        
        img = Image.new('RGB', (224, 224), color='brown')
        
        result = await analyzer.analyze(img)
        
        assert result is not None
        assert "classifications" in result or "top_prediction" in result
