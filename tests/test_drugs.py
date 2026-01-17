"""
UMI Drug API Tests
Tests for drug information and interaction checking
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_search_drugs(client: AsyncClient, auth_headers: dict):
    """Test searching for drugs."""
    response = await client.get(
        "/api/v1/drugs/search?q=paracetamol",
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_search_drugs_empty_query(client: AsyncClient, auth_headers: dict):
    """Test searching with empty query."""
    response = await client.get(
        "/api/v1/drugs/search?q=",
        headers=auth_headers,
    )
    
    assert response.status_code in [200, 422]


@pytest.mark.asyncio
async def test_get_drug_details(client: AsyncClient, auth_headers: dict):
    """Test getting drug details."""
    # Use a mock drug ID
    response = await client.get(
        "/api/v1/drugs/paracetamol",
        headers=auth_headers,
    )
    
    assert response.status_code in [200, 404]
    
    if response.status_code == 200:
        data = response.json()
        assert "name" in data or "generic_name" in data


@pytest.mark.asyncio
async def test_check_drug_interactions(client: AsyncClient, auth_headers: dict):
    """Test checking drug interactions."""
    response = await client.post(
        "/api/v1/drugs/interactions",
        json={"drugs": ["ibuprofen", "warfarin"]},
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "interactions" in data or isinstance(data, list)


@pytest.mark.asyncio
async def test_check_interactions_single_drug(client: AsyncClient, auth_headers: dict):
    """Test interaction check with single drug (should return empty)."""
    response = await client.post(
        "/api/v1/drugs/interactions",
        json={"drugs": ["paracetamol"]},
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    # Single drug should have no interactions
    if isinstance(data, list):
        assert len(data) == 0
    elif "interactions" in data:
        assert len(data["interactions"]) == 0


@pytest.mark.asyncio
async def test_get_drug_alternatives(client: AsyncClient, auth_headers: dict):
    """Test getting drug alternatives."""
    response = await client.get(
        "/api/v1/drugs/ibuprofen/alternatives",
        headers=auth_headers,
    )
    
    assert response.status_code in [200, 404]
    
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.asyncio
async def test_otc_recommendations(client: AsyncClient, auth_headers: dict):
    """Test OTC drug recommendations."""
    response = await client.post(
        "/api/v1/drugs/otc-recommendations",
        json={
            "symptoms": ["headache", "fever"],
            "age": 35,
        },
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_otc_recommendations_with_allergies(client: AsyncClient, auth_headers: dict):
    """Test OTC recommendations with allergy filtering."""
    response = await client.post(
        "/api/v1/drugs/otc-recommendations",
        json={
            "symptoms": ["pain"],
            "age": 40,
            "allergies": ["aspirin"],
        },
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Should not recommend aspirin
    for rec in data:
        if "generic_name" in rec:
            assert "aspirin" not in rec["generic_name"].lower()


@pytest.mark.asyncio
async def test_otc_recommendations_pediatric(client: AsyncClient, auth_headers: dict):
    """Test OTC recommendations for children."""
    response = await client.post(
        "/api/v1/drugs/otc-recommendations",
        json={
            "symptoms": ["fever"],
            "age": 5,
        },
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    # Should include pediatric warnings
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_drug_search_unauthenticated(client: AsyncClient):
    """Test that drug search requires authentication."""
    response = await client.get("/api/v1/drugs/search?q=aspirin")
    
    assert response.status_code == 401
