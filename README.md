Decision Graph Simulator (DGS)

Quick start (local development)

Requirements: Docker Desktop, Git, Python 3.9 (for local runs)

1. Copy environment example:

   cp .env.example .env

2. Start infra (Mongo + Ollama):

   docker-compose up -d mongo ollama

3. (Optional) Pull model inside Ollama container (example `phi3`):

   docker exec -it dgs_ollama ollama run phi3

4. Build and start backend:

   docker-compose build backend
   docker-compose up -d backend

5. Verify health:

   curl http://localhost:8000/health

Useful endpoints:

- GET `/health` — infrastructure health
- POST `/test/generate` — LLM generate test (json {"prompt":"..."})
- POST `/test/scrape` — insert simulated KnowledgeChunk (json {"prompt":"..."})

What to push:

- All source code, Dockerfiles, `docker-compose.yml`, `.env.example`, and `README.md`.
- Exclude runtime volumes and `.env` (added to `.gitignore`).

Next steps:

- Add frontend or hand off to frontend developer with OpenAPI docs at `/docs`.
- Continue backend work: ReasoningEngine, Deep RAG ingestion, tests, and telemetry.
