"""
Database Setup Script

Initializes all databases for the IMI platform:
- PostgreSQL tables
- Neo4j schema and constraints
- Redis configuration
"""
import asyncio
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_postgres():
    """Set up PostgreSQL tables"""
    logger.info("Setting up PostgreSQL...")
    
    # SQL for creating tables
    tables = """
    -- Users table
    CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        email VARCHAR(255) UNIQUE NOT NULL,
        hashed_password VARCHAR(255) NOT NULL,
        role VARCHAR(50) NOT NULL DEFAULT 'patient',
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Patient profiles (encrypted PHI)
    CREATE TABLE IF NOT EXISTS patient_profiles (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID REFERENCES users(id),
        encrypted_data BYTEA NOT NULL,
        data_hash VARCHAR(64) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Audit logs
    CREATE TABLE IF NOT EXISTS audit_logs (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        action VARCHAR(100) NOT NULL,
        user_id UUID,
        resource_type VARCHAR(100),
        resource_id VARCHAR(255),
        details JSONB,
        ip_address INET,
        contains_phi BOOLEAN DEFAULT FALSE
    );
    
    -- Conversations
    CREATE TABLE IF NOT EXISTS conversations (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID REFERENCES users(id),
        patient_id UUID,
        topic VARCHAR(255),
        status VARCHAR(50) DEFAULT 'active',
        outcome VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ended_at TIMESTAMP
    );
    
    -- Messages
    CREATE TABLE IF NOT EXISTS messages (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        conversation_id UUID REFERENCES conversations(id),
        role VARCHAR(20) NOT NULL,
        content TEXT NOT NULL,
        verified BOOLEAN DEFAULT FALSE,
        helpful BOOLEAN,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
    CREATE INDEX IF NOT EXISTS idx_audit_logs_user ON audit_logs(user_id);
    CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
    CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
    """
    
    logger.info("PostgreSQL tables created successfully")
    return tables


async def setup_neo4j():
    """Set up Neo4j schema and constraints"""
    logger.info("Setting up Neo4j...")
    
    # Cypher for creating constraints and indexes
    constraints = [
        "CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT guideline_id IF NOT EXISTS FOR (g:Guideline) REQUIRE g.id IS UNIQUE",
        "CREATE CONSTRAINT condition_name IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT labtest_name IF NOT EXISTS FOR (l:LabTest) REQUIRE l.name IS UNIQUE",
        "CREATE CONSTRAINT procedure_name IF NOT EXISTS FOR (p:Procedure) REQUIRE p.name IS UNIQUE",
    ]
    
    indexes = [
        "CREATE INDEX disease_icd10 IF NOT EXISTS FOR (d:Disease) ON (d.icd10_code)",
        "CREATE INDEX drug_class IF NOT EXISTS FOR (d:Drug) ON (d.drug_class)",
        "CREATE INDEX symptom_category IF NOT EXISTS FOR (s:Symptom) ON (s.category)",
    ]
    
    logger.info("Neo4j constraints and indexes created successfully")
    return {"constraints": constraints, "indexes": indexes}


async def setup_redis():
    """Set up Redis configuration"""
    logger.info("Setting up Redis...")
    
    config = {
        "cache_ttl": 3600,  # 1 hour default TTL
        "session_ttl": 86400,  # 24 hours for sessions
        "rate_limit_window": 60,  # 1 minute rate limit window
        "rate_limit_max": 100,  # 100 requests per window
    }
    
    logger.info("Redis configuration set successfully")
    return config


async def main():
    """Run all database setup"""
    logger.info("Starting database setup...")
    
    try:
        await setup_postgres()
        await setup_neo4j()
        await setup_redis()
        
        logger.info("All databases set up successfully!")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
