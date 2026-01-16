"""
UMI Consultation Tests
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_start_consultation(client: AsyncClient, auth_headers):
    """Test starting a new consultation."""
    response = await client.post(
        "/api/v1/consultations",
        headers=auth_headers,
        json={
            "initial_message": "I have a headache",
            "language": "en",
        },
    )
    
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["status"] == "in_progress"
    assert len(data["messages"]) > 0


@pytest.mark.asyncio
async def test_list_consultations(client: AsyncClient, auth_headers):
    """Test listing user consultations."""
    # First create a consultation
    await client.post(
        "/api/v1/consultations",
        headers=auth_headers,
        json={"language": "en"},
    )
    
    # List consultations
    response = await client.get(
        "/api/v1/consultations",
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_symptom_check(client: AsyncClient, auth_headers):
    """Test quick symptom check."""
    response = await client.post(
        "/api/v1/consultations/symptom-check",
        headers=auth_headers,
        json={
            "symptoms": ["headache", "fatigue"],
            "age": 30,
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "possible_conditions" in data
    assert "urgency_level" in data
    assert "recommendation" in data
