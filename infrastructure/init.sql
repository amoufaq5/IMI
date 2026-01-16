-- UMI Database Initialization Script
-- This script runs when the PostgreSQL container is first created

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Create schemas
CREATE SCHEMA IF NOT EXISTS umi;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA umi TO umi_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA umi TO umi_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA umi TO umi_user;

-- Set default schema
ALTER USER umi_user SET search_path TO umi, public;

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'UMI database initialized successfully';
END $$;
