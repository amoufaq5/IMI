"""
UMI Test Configuration
Pytest fixtures and test utilities
"""

import asyncio
from typing import AsyncGenerator, Generator
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from src.core.database import Base, get_db
from src.core.security import create_access_token, get_password_hash
from src.main import app
from src.models.user import User, UserRole, UserProfile


# Test database URL (use SQLite for tests)
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

# Create test engine
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
)

# Test session factory
TestSessionLocal = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database session for each test."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with TestSessionLocal() as session:
        yield session
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with database override."""
    
    async def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create a test user."""
    user = User(
        id=uuid4(),
        email="test@example.com",
        password_hash=get_password_hash("TestPassword123"),
        role=UserRole.GENERAL_USER,
        is_active=True,
        is_verified=True,
    )
    db_session.add(user)
    
    profile = UserProfile(
        user_id=user.id,
        first_name="Test",
        last_name="User",
    )
    db_session.add(profile)
    
    await db_session.commit()
    await db_session.refresh(user)
    
    return user


@pytest_asyncio.fixture
async def test_admin(db_session: AsyncSession) -> User:
    """Create a test admin user."""
    user = User(
        id=uuid4(),
        email="admin@example.com",
        password_hash=get_password_hash("AdminPassword123"),
        role=UserRole.SYSTEM_ADMIN,
        is_active=True,
        is_verified=True,
        is_superuser=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    
    return user


@pytest.fixture
def auth_headers(test_user: User) -> dict:
    """Create authentication headers for test user."""
    token = create_access_token(subject=str(test_user.id))
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers(test_admin: User) -> dict:
    """Create authentication headers for admin user."""
    token = create_access_token(subject=str(test_admin.id))
    return {"Authorization": f"Bearer {token}"}
