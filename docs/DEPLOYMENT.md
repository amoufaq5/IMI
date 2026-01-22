# IMI Deployment Guide

## Prerequisites

- Python 3.10+
- PostgreSQL 14+
- Neo4j 5.0+
- Redis 7.0+
- NVIDIA GPU (for LLM inference)

## Local Development

### 1. Clone and Setup

```bash
git clone <repository>
cd imi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

Required environment variables:
- `DATABASE_URL` - PostgreSQL connection string
- `NEO4J_URI` - Neo4j connection URI
- `REDIS_URL` - Redis connection URL
- `SECRET_KEY` - JWT signing key
- `ENCRYPTION_KEY` - PHI encryption key

### 3. Initialize Databases

```bash
# Setup PostgreSQL tables
python scripts/setup_database.py

# Seed knowledge graph
python scripts/seed_knowledge_graph.py
```

### 4. Run Server

```bash
# Development mode with auto-reload
python scripts/run_server.py --reload

# Production mode
python scripts/run_server.py --workers 4
```

## Docker Deployment

### Build Image

```bash
docker build -t imi:latest .
```

### Run with Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    image: imi:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/imi
      - NEO4J_URI=bolt://neo4j:7687
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - neo4j
      - redis

  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: imi
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  neo4j:
    image: neo4j:5
    environment:
      NEO4J_AUTH: neo4j/password
    volumes:
      - neo4j_data:/data

  redis:
    image: redis:7
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  neo4j_data:
  redis_data:
```

```bash
docker-compose up -d
```

## Cloud Deployment

### AWS

1. **ECS/Fargate** for API containers
2. **RDS PostgreSQL** for relational data
3. **Neptune** or self-managed Neo4j for graph
4. **ElastiCache Redis** for caching
5. **S3** for model storage
6. **CloudWatch** for logging

### GCP

1. **Cloud Run** for API
2. **Cloud SQL** for PostgreSQL
3. **Memorystore** for Redis
4. **GCS** for model storage

## GPU Deployment (RunPod)

See `docs/RUNPOD_DEPLOYMENT_GUIDE.md` for GPU-specific deployment.

## Security Checklist

- [ ] Enable HTTPS/TLS
- [ ] Configure firewall rules
- [ ] Set strong passwords
- [ ] Enable audit logging
- [ ] Configure backup schedule
- [ ] Set up monitoring alerts
- [ ] Review RBAC permissions
- [ ] Test encryption at rest

## Monitoring

### Health Checks

```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/detailed
```

### Metrics

The API exposes Prometheus metrics at `/metrics` (when enabled).

### Logs

Logs are structured JSON for easy parsing:

```json
{
  "timestamp": "2024-01-22T14:00:00Z",
  "level": "info",
  "message": "Request processed",
  "user_id": "uuid",
  "duration_ms": 150
}
```
