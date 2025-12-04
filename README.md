# Knowledge Assistant for Support Team

An LLM-powered RAG (Retrieval-Augmented Generation) system that helps support teams respond to customer tickets using relevant Tucows documentation.

## Overview

This application processes customer support tickets and returns structured, MCP-compliant JSON responses with:
- An AI-generated answer based on relevant documentation
- References to source documents used
- Recommended actions for the support team

### Example

**Input:**
```json
{
  "ticket_text": "My domain was suspended and I didn't get any notice. How can I reactivate it?"
}
```

**Output:**
```json
{
  "answer": "Unfortunately, your domain has expired and is currently in the Redemption Period state. You should contact your Domain Provider at https://tucowsdomains.com/provider-search/ to initiate the redemption process and potentially reactivate your domain.",
  "references": ["about_tucows", "renewals_and_redemptions"],
  "action_required": "contact_domain_provider"
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Knowledge Assistant                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐      │
│   │ FastAPI  │───▶│ RAG Pipeline │───▶│ LLM (Ollama) │      │
│   │   API    │    │   (FAISS)    │    │              │      │
│   └──────────┘    └──────────────┘    └──────────────┘      │
│                          │                                   │
│                          ▼                                   │
│                   ┌──────────────┐                          │
│                   │ Vector Store │                          │
│                   │   (FAISS)    │                          │
│                   └──────────────┘                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Component | Technology |
|-----------|------------|
| API | FastAPI |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS |
| LLM | Ollama (`llama3.2:3b`) |

### Data Flow

1. **Request** → User submits a support ticket via `POST /resolve-ticket`
2. **Embed** → Query is converted to a vector embedding
3. **Retrieve** → FAISS finds the most similar document chunks
4. **Generate** → Ollama LLM generates a response using retrieved context
5. **Response** → MCP-compliant JSON returned to user

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/mpat247/Knowledge-Assistant-for-Support-Team-Interview-Exercise.git
cd Knowledge-Assistant-for-Support-Team-Interview-Exercise
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up Ollama**
```bash
ollama serve          # Start Ollama (if not running)
ollama pull llama3.2:3b  # Download the model
```

**5. Run the application**
```bash
python -m src.main
```

The API will be available at `http://localhost:8000`

### Using Docker

```bash
docker-compose up --build
```

> **Note:** Ollama must be running on your host machine. The container connects via `host.docker.internal`.

---

## API Reference

### POST /resolve-ticket

Process a support ticket and get an AI-generated response.

**Request:**
```bash
curl -X POST http://localhost:8000/resolve-ticket \
  -H "Content-Type: application/json" \
  -d '{"ticket_text": "How do I transfer my domain to another registrar?"}'
```

**Response:**
```json
{
  "answer": "To transfer your domain...",
  "references": ["Document: domain_transfers"],
  "action_required": "initiate_transfer"
}
```

### GET /

Health check endpoint.

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "status": "healthy",
  "ollama": "connected"
}
```

### Interactive Docs

- Swagger UI: http://localhost:8000/docs


## Testing

```bash
pytest tests/ -v
```

---

## Project Structure

```
├── src/
│   ├── main.py          # FastAPI app entry point
│   ├── config.py        # Configuration settings
│   ├── api/
│   │   ├── route.py     # API endpoints
│   │   └── schemas.py   # Pydantic models
│   ├── llm/
│   │   └── llm.py       # Ollama client & prompts
│   └── rag/
│       └── rag.py       # RAG pipeline (embed, index, retrieve)
├── data/
│   ├── documents/       # Source .txt files
│   └── vector_store/    # FAISS index (auto-generated)
├── tests/               # Unit tests
├── docker-compose.yml
├── DockerFile
└── requirements.txt
```