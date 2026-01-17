"""
UMI Medical Imaging API Tests
Tests for medical image analysis
"""

import io
import pytest
from httpx import AsyncClient
from PIL import Image


def create_test_image(width: int = 512, height: int = 512) -> bytes:
    """Create a simple test image."""
    img = Image.new('RGB', (width, height), color='gray')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_analyze_chest_xray(client: AsyncClient, auth_headers: dict):
    """Test chest X-ray analysis."""
    image_data = create_test_image()
    
    response = await client.post(
        "/api/v1/imaging/analyze",
        files={"file": ("test_xray.png", image_data, "image/png")},
        data={
            "image_type": "xray",
            "body_region": "chest",
        },
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "image_type" in data
    assert "findings" in data
    assert "impression" in data
    assert "urgency" in data
    assert data["image_type"] == "xray"


@pytest.mark.asyncio
async def test_analyze_dermoscopy(client: AsyncClient, auth_headers: dict):
    """Test skin lesion analysis."""
    image_data = create_test_image(224, 224)
    
    response = await client.post(
        "/api/v1/imaging/analyze",
        files={"file": ("skin_lesion.png", image_data, "image/png")},
        data={"image_type": "dermoscopy"},
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["image_type"] == "dermoscopy"
    assert "findings" in data
    assert "recommendations" in data


@pytest.mark.asyncio
async def test_analyze_lab_report(client: AsyncClient, auth_headers: dict):
    """Test lab report OCR analysis."""
    image_data = create_test_image()
    
    response = await client.post(
        "/api/v1/imaging/analyze",
        files={"file": ("lab_report.png", image_data, "image/png")},
        data={"image_type": "lab_report"},
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["image_type"] == "lab_report"


@pytest.mark.asyncio
async def test_invalid_image_type(client: AsyncClient, auth_headers: dict):
    """Test with invalid image type."""
    image_data = create_test_image()
    
    response = await client.post(
        "/api/v1/imaging/analyze",
        files={"file": ("test.png", image_data, "image/png")},
        data={"image_type": "invalid_type"},
        headers=auth_headers,
    )
    
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_invalid_body_region(client: AsyncClient, auth_headers: dict):
    """Test with invalid body region."""
    image_data = create_test_image()
    
    response = await client.post(
        "/api/v1/imaging/analyze",
        files={"file": ("test.png", image_data, "image/png")},
        data={
            "image_type": "xray",
            "body_region": "invalid_region",
        },
        headers=auth_headers,
    )
    
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_unsupported_file_type(client: AsyncClient, auth_headers: dict):
    """Test with unsupported file type."""
    response = await client.post(
        "/api/v1/imaging/analyze",
        files={"file": ("test.txt", b"not an image", "text/plain")},
        data={"image_type": "xray"},
        headers=auth_headers,
    )
    
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_supported_types(client: AsyncClient, auth_headers: dict):
    """Test getting supported image types."""
    response = await client.get(
        "/api/v1/imaging/supported-types",
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "image_types" in data
    assert "body_regions" in data
    assert len(data["image_types"]) > 0
    assert len(data["body_regions"]) > 0


@pytest.mark.asyncio
async def test_imaging_unauthenticated(client: AsyncClient):
    """Test that imaging requires authentication."""
    image_data = create_test_image()
    
    response = await client.post(
        "/api/v1/imaging/analyze",
        files={"file": ("test.png", image_data, "image/png")},
        data={"image_type": "xray"},
    )
    
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_analysis_response_structure(client: AsyncClient, auth_headers: dict):
    """Test that analysis response has correct structure."""
    image_data = create_test_image()
    
    response = await client.post(
        "/api/v1/imaging/analyze",
        files={"file": ("test.png", image_data, "image/png")},
        data={
            "image_type": "xray",
            "body_region": "chest",
        },
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check all required fields
    required_fields = [
        "image_type",
        "findings",
        "impression",
        "confidence",
        "recommendations",
        "abnormalities_detected",
        "urgency",
        "processing_time_ms",
    ]
    
    for field in required_fields:
        assert field in data, f"Missing field: {field}"
    
    # Check types
    assert isinstance(data["findings"], list)
    assert isinstance(data["recommendations"], list)
    assert isinstance(data["confidence"], (int, float))
    assert isinstance(data["abnormalities_detected"], bool)
    assert data["urgency"] in ["normal", "attention", "urgent", "critical"]
