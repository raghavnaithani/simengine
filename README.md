# Decision Graph Simulator (DGS)

## Overview
Decision Graph Simulator (DGS) is a local-first decision-intelligence toolkit designed to transform ambiguous, real-world prompts into branching simulated futures. Unlike traditional systems, DGS emphasizes interpretation over prediction, creating explicit graphs of decisions, outcomes, and failure modes. This ensures a focus on trade-offs, risks, and uncertainty.

### Key Features
- **Local-first**: All computations run locally within Docker containers, ensuring privacy and independence from external SaaS dependencies.
- **Interpretation over Data**: Separates immutable facts (Data Layer) from stochastic reasoning (Reasoning Layer).
- **Safety-first**: Enforces mandatory risk enumeration and provenance to reduce hallucinations and unsupported claims.
- **Incremental Simulation**: Efficient branching and exploration by avoiding re-processing full histories.

---

## Getting Started

### Prerequisites
- **Docker Desktop**
- **Git**
- **Python 3.9** (for local runs)

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/raghavnaithani/simengine.git
   cd simengine
   ```

2. Copy the environment example file:
   ```bash
   cp .env.example .env
   ```

3. Start the infrastructure (MongoDB + Ollama):
   ```bash
   docker-compose up -d mongo ollama
   ```

4. (Optional) Pull the desired model inside the Ollama container (e.g., `phi3`):
   ```bash
   docker exec -it dgs_ollama ollama pull phi3
   ```

5. Build and start the backend:
   ```bash
   docker-compose build backend
   docker-compose up -d backend
   ```

6. Verify the backend health:
   ```bash
   curl http://localhost:8000/health
   ```

7. Start the frontend:
   ```bash
   docker-compose up -d frontend
   ```

---

## System Architecture

### Components
- **Frontend**: React-based UI for visualizing decision graphs and interacting with nodes.
- **Backend**: FastAPI orchestrator managing engines and APIs.
- **MongoDB**: Primary persistence layer for graphs and context metadata.
- **Ollama**: Local LLM service for reasoning and inference.

### Major Layers
1. **Presentation**: React front-end rendering decision graphs, focus panels, and node editors.
2. **Orchestration**: FastAPI routes and background tasks for session management and job handling.
3. **Intelligence**: Micro-engines for context building, reasoning, and simulation.
4. **Persistence**: MongoDB for storing graphs and KnowledgeChunks with embeddings and TTL metadata.

---

## Usage

### Key Endpoints
- **GET `/health`**: Check infrastructure health.
- **POST `/simulate/start`**: Start a new simulation session.
- **POST `/simulate/branch`**: Create a new branch in the decision graph.
- **GET `/graph/{session_id}`**: Retrieve the full decision graph for a session.
- **GET `/node/{node_id}`**: Retrieve details of a specific node.

### Example Workflow
1. Start a simulation session with a prompt.
2. Explore branching decisions by interacting with the graph.
3. Analyze risks, outcomes, and trade-offs for each decision.

---

## Development

### Recommended Layout
- **backend/**: Contains the FastAPI app, engines, models, and utilities.
- **frontend/**: React-based UI components.
- **docker-compose.yml**: Defines the containerized infrastructure.
- **.env.example**: Example environment configuration.

### Testing
- **Unit Tests**: Validate Pydantic models and engine behaviors.
- **Integration Tests**: End-to-end tests using mocked ReasoningEngine outputs.

Run tests using:
```bash
pytest
```

---

## Roadmap

### Immediate Goals
- Implement Deep RAG ingestion and retrieval.
- Enforce citation and risk validation.
- Optimize background tasks for scraping and embedding.

### Future Enhancements
- Add hybrid sparse fallback for low-confidence retrievals.
- Integrate optional local vector DB for large corpora.
- Enable scheduled pruning and offline embedding refresh.

---

## Contributing
We welcome contributions! Please follow the [contribution guidelines](CONTRIBUTING.md) and ensure all tests pass before submitting a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
Special thanks to all contributors and the open-source community for their support in building this project.
