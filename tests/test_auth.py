"""
UMI Authentication Tests
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_user(client: AsyncClient):
    """Test user registration."""
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "newuser@example.com",
            "password": "SecurePassword123",
            "confirm_password": "SecurePassword123",
        },
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "newuser@example.com"
    assert "id" in data


@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient, test_user):
    """Test registration with existing email."""
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": test_user.email,
            "password": "SecurePassword123",
            "confirm_password": "SecurePassword123",
        },
    )
    
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_register_weak_password(client: AsyncClient):
    """Test registration with weak password."""
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "weak@example.com",
            "password": "weak",
            "confirm_password": "weak",
        },
    )
    
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_login_success(client: AsyncClient, test_user):
    """Test successful login."""
    response = await client.post(
        "/api/v1/auth/login",
        json={
            "email": test_user.email,
            "password": "TestPassword123",
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient, test_user):
    """Test login with wrong password."""
    response = await client.post(
        "/api/v1/auth/login",
        json={
            "email": test_user.email,
            "password": "WrongPassword123",
        },
    )
    
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_login_nonexistent_user(client: AsyncClient):
    """Test login with non-existent user."""
    response = await client.post(
        "/api/v1/auth/login",
        json={
            "email": "nonexistent@example.com",
            "password": "SomePassword123",
        },
    )
    
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_refresh_token(client: AsyncClient, test_user):
    """Test token refresh."""
    # First login
    login_response = await client.post(
        "/api/v1/auth/login",
        json={
            "email": test_user.email,
            "password": "TestPassword123",
        },
    )
    
    refresh_token = login_response.json()["refresh_token"]
    
    # Refresh token
    response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data


@pytest.mark.asyncio
async def test_protected_endpoint_without_auth(client: AsyncClient):
    """Test accessing protected endpoint without authentication."""
    response = await client.get("/api/v1/users/me")
    
    assert response.status_code == 403  # No auth header


@pytest.mark.asyncio
async def test_protected_endpoint_with_auth(client: AsyncClient, auth_headers):
    """Test accessing protected endpoint with authentication."""
    response = await client.get(
        "/api/v1/users/me",
        headers=auth_headers,
    )
    
    assert response.status_code == 200
