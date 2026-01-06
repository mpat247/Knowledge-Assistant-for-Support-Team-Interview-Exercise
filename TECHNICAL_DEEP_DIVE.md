# Technical Deep Dive: Knowledge Assistant RAG System

## Complete End-to-End Study Guide

This document provides a comprehensive technical breakdown of every component, design decision, and code path in the Knowledge Assistant system. Use this to understand exactly how the system works at every level.

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Complete Request Lifecycle](#complete-request-lifecycle)
3. [Module-by-Module Breakdown](#module-by-module-breakdown)
4. [RAG Pipeline Deep Dive](#rag-pipeline-deep-dive)
5. [LLM Integration Deep Dive](#llm-integration-deep-dive)
6. [Data Structures and Formats](#data-structures-and-formats)
7. [Configuration and Environment](#configuration-and-environment)
8. [Testing Strategy](#testing-strategy)
9. [Deployment Architecture](#deployment-architecture)
10. [Design Decisions and Tradeoffs](#design-decisions-and-tradeoffs)

---

## System Architecture Overview

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        Knowledge Assistant                       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               │ HTTP Request
                               ▼
                    ┌──────────────────┐
                    │   FastAPI App    │
                    │   (src/main.py)  │
                    └────────┬─────────┘
                             │
                             │ Route to endpoint
                             ▼
                    ┌──────────────────┐
                    │   API Router     │
                    │ (src/api/route.py)│
                    └────────┬─────────┘
                             │
                  ┌──────────┴──────────┐
                  │                     │
                  ▼                     ▼
         ┌────────────────┐    ┌────────────────┐
         │  RAG Pipeline  │    │  LLM Client    │
         │ (src/rag/rag.py)│    │(src/llm/llm.py)│
         └────────┬───────┘    └────────┬───────┘
                  │                     │
                  │                     │
         ┌────────▼───────┐    ┌────────▼───────┐
         │ Vector Store   │    │ Ollama Server  │
         │ (FAISS Index)  │    │ (External)     │
         └────────────────┘    └────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API Framework** | FastAPI | Modern Python web framework with automatic OpenAPI docs |
| **Embedding Model** | SentenceTransformers (`all-MiniLM-L6-v2`) | Convert text to 384-dim vectors |
| **Vector Store** | FAISS (IndexFlatL2) | Fast similarity search on embeddings |
| **LLM** | Ollama (llama3.2:3b) | Open-source LLM for response generation |
| **Schema Validation** | Pydantic | Request/response data models |
| **Testing** | pytest | Unit and integration tests |
| **Containerization** | Docker | Service packaging and deployment |

---

## Complete Request Lifecycle

### Step-by-Step Trace: `/resolve-ticket` Request

#### **Phase 1: Service Startup**

**File: `src/main.py`**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load index on startup."""
    retriever.ensure_index_loaded()
    yield
```

**What happens:**
1. FastAPI starts and enters the `lifespan` context manager
2. Calls `retriever.ensure_index_loaded()` which:
   - Checks if `data/vector_store/index.faiss` exists
   - If exists: loads index + chunks + sources from disk
   - If missing: builds entire index from scratch (expensive!)
3. App is now ready to handle requests with vector index in memory

---

#### **Phase 2: HTTP Request Arrives**

**Request:**
```bash
curl -X POST http://localhost:8000/resolve-ticket \
  -H "Content-Type: application/json" \
  -d '{"ticket_text": "How do I transfer my domain?"}'
```

**File: `src/api/route.py` - Line 8-30**

```python
@router.post("/resolve-ticket", response_model=TicketResponse)
async def resolve_ticket(request: TicketRequest) -> TicketResponse:
```

**Step 2.1: Pydantic Validation**
- FastAPI deserializes JSON into `TicketRequest` model
- Validates `ticket_text` is present and `min_length=1`
- If validation fails → automatic HTTP 422 response

**Step 2.2: Context Retrieval**
```python
context, references = retriever.get_context(request.ticket_text)
```
This triggers the RAG pipeline (detailed below in Phase 3).

---

#### **Phase 3: RAG Retrieval**

**File: `src/rag/rag.py` - `get_context()` method**

**Step 3.1: Embed the Query**
```python
def get_context(self, query: str, top_k: int = None) -> Tuple[str, List[str]]:
    results = self.retrieve(query, top_k)
```

Inside `retrieve()`:
```python
query_embedding = np.array([self.get_embedding(query)])
```

**What happens:**
- Query "How do I transfer my domain?" → SentenceTransformer model
- Model outputs 384-dimensional float vector
- Vector normalized to unit length (automatically by model)

**Step 3.2: FAISS Search**
```python
distances, indices = self.index.search(query_embedding, top_k)
```

**What happens:**
- FAISS uses `IndexFlatL2` (brute-force L2 distance)
- Computes distance between query vector and ALL chunk vectors
- Returns top-k closest matches (default k=3)
- Returns: `indices` (positions in chunk array), `distances` (L2 distances)

**Step 3.3: Build Context String**
```python
for chunk, source, _ in results:
    context_parts.append(f"[Source: {source}]\n{chunk}")
    ref = f"Document: {source}"
    if ref not in references:
        references.append(ref)

return "\n\n---\n\n".join(context_parts), references
```

**Example output:**
```
[Source: domain_transfers]
To transfer your domain to another registrar, you must first unlock
your domain and obtain an authorization code...

---

[Source: tucows_domain_promise]
Tucows Domains provides support for all domain transfers...
```

---

#### **Phase 4: LLM Generation**

**File: `src/api/route.py` - Line 20-21**
```python
response = await ollama_client.generate(context, request.ticket_text)
```

**File: `src/llm/llm.py` - `generate()` method**

**Step 4.1: Build Prompt**
```python
prompt = build_prompt(context, query)

payload = {
    "model": self.model,
    "prompt": f"{prompt['system']}\n\n{prompt['user']}",
    "stream": False,
    "format": "json"
}
```

**Prompt Structure:**
```
SYSTEM PROMPT:
You are a customer support assistant for Tucows.
Analyze tickets and respond based on the provided documentation.

Always respond with JSON in this format:
{
    "answer": "Your response to the customer",
    "references": ["Source documents used"],
    "action_required": "Recommended action or null"
}
...

USER PROMPT:
## Context:
[Source: domain_transfers]
To transfer your domain...

---

[Source: tucows_domain_promise]
Tucows Domains provides...

## Ticket:
How do I transfer my domain?

Respond with valid JSON only.
```

**Step 4.2: Call Ollama API**
```python
async with httpx.AsyncClient(timeout=60.0) as client:
    response = await client.post(f"{self.base_url}/api/generate", json=payload)
    response.raise_for_status()
```

**Ollama API:**
- Endpoint: `http://localhost:11434/api/generate`
- Method: POST
- Payload includes `"format": "json"` which constrains model output
- Timeout: 60 seconds (generation can be slow)

**Step 4.3: Parse Response**
```python
result = response.json()
return self._parse_response(result.get("response", ""))
```

**Raw Ollama response structure:**
```json
{
  "model": "llama3.2:3b",
  "created_at": "2026-01-05T...",
  "response": "{\"answer\": \"To transfer...\", \"references\": [...], ...}",
  "done": true
}
```

**Parser behavior:**
```python
def _parse_response(self, response_text: str) -> TicketResponse:
    try:
        data = json.loads(response_text)
        return TicketResponse(
            answer=data.get("answer", "Unable to generate response."),
            references=data.get("references", []),
            action_required=data.get("action_required")
        )
    except json.JSONDecodeError:
        # Fallback: treat entire response as answer
        return TicketResponse(
            answer=response_text if response_text else "Unable to generate response.",
            references=[],
            action_required=None
        )
```

---

#### **Phase 5: Response Assembly**

**File: `src/api/route.py` - Line 23-26**
```python
if not response.references:
    response.references = references
    
return response
```

**Logic:**
- If LLM forgot to include references in its JSON, backfill from retrieval
- Return final `TicketResponse` object

**FastAPI behavior:**
- Pydantic serializes `TicketResponse` to JSON
- Adds proper `Content-Type: application/json` header
- Returns HTTP 200 with response body

**Final Response:**
```json
{
  "answer": "To transfer your domain, you must first unlock it and obtain an authorization code from your current registrar. Then, initiate the transfer process with your new registrar using this code.",
  "references": ["Document: domain_transfers", "Document: tucows_domain_promise"],
  "action_required": "initiate_transfer"
}
```

---

## Module-by-Module Breakdown

### `src/main.py` - Application Entry Point

**Purpose:** FastAPI app initialization and lifecycle management

**Key Components:**

1. **Lifespan Context Manager:**
   ```python
   @asynccontextmanager
   async def lifespan(app: FastAPI):
       retriever.ensure_index_loaded()  # Pre-load vector index
       yield
   ```
   - Runs on startup before accepting requests
   - Ensures vector index is loaded into memory
   - `yield` separates startup from shutdown logic

2. **FastAPI App:**
   ```python
   app = FastAPI(
       title="Knowledge Assistant API",
       description="RAG-powered support ticket resolution for Tucows",
       version="1.0.0",
       lifespan=lifespan
   )
   ```
   - Metadata used for auto-generated OpenAPI docs
   - Available at `/docs` (Swagger UI) and `/redoc`

3. **Router Registration:**
   ```python
   app.include_router(router)
   ```
   - Mounts all endpoints from `src/api/route.py`

4. **Development Server:**
   ```python
   if __name__ == "__main__":
       uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
   ```
   - Only runs when executed directly (not imported)
   - `reload=True` watches file changes during development

---

### `src/config.py` - Configuration Management

**Purpose:** Centralized configuration with environment variable support

**Configuration Categories:**

1. **File Paths:**
   ```python
   BASE_DIR = Path(__file__).resolve().parent.parent
   DATA_DIR = BASE_DIR / "data" / "documents"
   VECTOR_STORE_PATH = BASE_DIR / "data" / "vector_store"
   ```
   - Computed relative to source file location
   - Works across different deployment environments

2. **Ollama Configuration:**
   ```python
   OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
   OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
   ```
   - Defaults to local Ollama instance
   - Docker override: `http://host.docker.internal:11434`

3. **Embedding Configuration:**
   ```python
   EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
   ```
   - SentenceTransformers model name
   - Must match model used to build vector store

4. **RAG Parameters:**
   ```python
   TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
   CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
   CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
   ```
   - TOP_K: number of chunks to retrieve
   - CHUNK_SIZE: characters per chunk
   - CHUNK_OVERLAP: overlap between adjacent chunks

---

### `src/api/route.py` - API Endpoints

**Purpose:** Define HTTP endpoints and request handling logic

**Endpoints:**

#### 1. `POST /resolve-ticket`

**Function signature:**
```python
async def resolve_ticket(request: TicketRequest) -> TicketResponse
```

**Request validation:**
- Pydantic automatically validates JSON structure
- Returns 422 if `ticket_text` missing or empty

**Processing steps:**
1. Retrieve context: `context, references = retriever.get_context(request.ticket_text)`
2. Check context exists (else HTTP 500)
3. Generate response: `response = await ollama_client.generate(context, request.ticket_text)`
4. Backfill references if LLM omitted them
5. Return `TicketResponse` (auto-serialized to JSON)

**Error handling:**
- `HTTPException` → passes through (status code preserved)
- Other exceptions → HTTP 500 with error message

#### 2. `GET /`

**Function signature:**
```python
async def health_check()
```

**Purpose:** Service health check

**Logic:**
```python
ollama_healthy = await ollama_client.health_check()
return {
    "status": "healthy" if ollama_healthy else "degraded",
    "ollama": "connected" if ollama_healthy else "disconnected"
}
```

**Returns:**
- `"status": "healthy"` if Ollama is reachable
- `"status": "degraded"` if Ollama is down (app still runs, but can't generate)

---

### `src/api/schemas.py` - Data Models

**Purpose:** Pydantic models for request/response validation

#### `TicketRequest`

```python
class TicketRequest(BaseModel):
    ticket_text: str = Field(
        ...,  # Required field
        description="Support ticket text",
        min_length=1,  # Cannot be empty
        examples=["My domain was suspended. How can I reactivate it?"]
    )
```

**Validation rules:**
- Must be string
- Cannot be empty string
- Automatic validation by FastAPI

#### `TicketResponse`

```python
class TicketResponse(BaseModel):
    answer: str = Field(..., description="Response to the ticket")
    references: List[str] = Field(default_factory=list, description="Source documents used")
    action_required: Optional[str] = Field(default=None, description="Recommended action")
```

**Field behaviors:**
- `answer`: required field
- `references`: defaults to empty list if not provided
- `action_required`: optional, defaults to null

**Serialization:**
```python
response.model_dump()  # → dict
response.model_dump_json()  # → JSON string
```

---

## RAG Pipeline Deep Dive

### Architecture

```
Documents (*.txt)
      ↓
   Chunking (500 chars, 50 overlap)
      ↓
   Embedding (SentenceTransformer)
      ↓
   FAISS Index (IndexFlatL2)
      ↓
   Persist to disk (*.npy + *.faiss)
      ↓
   Load into memory on startup
      ↓
   Query → Retrieve top-k chunks
```

### `RAGPipeline` Class Breakdown

**File: `src/rag/rag.py`**

#### **Initialization**

```python
def __init__(self):
    self.model = SentenceTransformer(EMBEDDING_MODEL)
    self.index = None
    self.chunks = []
    self.chunk_sources = []
```

**State variables:**
- `model`: SentenceTransformer instance (loaded once, reused)
- `index`: FAISS index object (in-memory)
- `chunks`: list of text chunks (parallel to index)
- `chunk_sources`: list of document names (parallel to chunks)

---

#### **Document Loading**

```python
def load_documents(self) -> List[Tuple[str, str]]:
    documents = []
    for file_path in DATA_DIR.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            documents.append((file_path.stem, f.read()))
    return documents
```

**Logic:**
- Scans `data/documents/` for `.txt` files
- `file_path.stem` extracts filename without extension
- Returns list of `(filename, content)` tuples

**Example:**
```python
[
    ("about_tucows", "Tucows Domains is the world's largest..."),
    ("domain_transfers", "To transfer a domain..."),
    ...
]
```

---

#### **Text Chunking**

```python
def chunk_text(self, text: str, source: str) -> List[Tuple[str, str]]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append((chunk.strip(), source))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks
```

**Algorithm:**
- Sliding window with overlap
- `CHUNK_SIZE = 500` characters
- `CHUNK_OVERLAP = 50` characters
- Each iteration moves `450` characters forward

**Example:**
```
Text: "ABCDEFGHIJ..." (1000 chars)

Chunk 1: chars [0:500]   → "ABCDE..."
Chunk 2: chars [450:950] → "...DEFGH..."  (overlaps with chunk 1)
Chunk 3: chars [900:1000] → "...IJ"
```

**Why overlap?**
- Prevents cutting context across chunk boundaries
- Ensures semantic concepts aren't split

**Tradeoff:**
- More chunks = slower search, but better recall
- Less overlap = faster, but might miss context

---

#### **Embedding Generation**

```python
def get_embedding(self, text: str) -> np.ndarray:
    return self.model.encode([text])[0].astype('float32')
```

**SentenceTransformer behavior:**
- Model: `all-MiniLM-L6-v2`
- Input: string
- Output: 384-dimensional float32 vector
- Vectors are L2-normalized (unit length)

**Properties:**
- Semantically similar text → similar vectors (high cosine similarity)
- Model is lightweight (~23MB)
- Fast inference (~10ms per encoding on CPU)

---

#### **Index Building**

```python
def build_index(self) -> None:
    documents = self.load_documents()
    
    all_chunks = []
    for source, content in documents:
        all_chunks.extend(self.chunk_text(content, source))
    
    self.chunks = [chunk for chunk, _ in all_chunks]
    self.chunk_sources = [source for _, source in all_chunks]
    
    embeddings = self.model.encode(self.chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
    self.index = faiss.IndexFlatL2(dimension)
    self.index.add(embeddings)
    
    self._save_index()
```

**Step-by-step:**

1. **Load all documents** → list of (source, content) tuples
2. **Chunk all content** → list of (chunk_text, source) tuples
3. **Separate chunks and sources** → parallel lists
4. **Encode all chunks** → numpy array of shape (N, 384)
5. **Create FAISS index:**
   - `IndexFlatL2`: brute-force L2 distance search
   - `dimension=384`: embedding size
6. **Add vectors to index** → index.add(embeddings)
7. **Persist to disk** → save index, chunks, sources

**FAISS `IndexFlatL2`:**
- Simplest FAISS index type
- Computes exact L2 distance to every vector
- No approximation, perfect recall
- Complexity: O(N × D) per query (N=num chunks, D=dimension)
- Good for <100k vectors

**Alternative indexes:**
- `IndexIVFFlat`: approximate search with clustering
- `IndexHNSW`: graph-based approximate search
- Use when N > 100k for speed

---

#### **Persistence**

```python
def _save_index(self) -> None:
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    faiss.write_index(self.index, str(VECTOR_STORE_PATH / "index.faiss"))
    np.save(VECTOR_STORE_PATH / "chunks.npy", np.array(self.chunks, dtype=object))
    np.save(VECTOR_STORE_PATH / "sources.npy", np.array(self.chunk_sources, dtype=object))
```

**Files created:**
- `index.faiss`: FAISS index (binary format)
- `chunks.npy`: numpy array of text chunks
- `sources.npy`: numpy array of source document names

**Why persist?**
- Building index is expensive (seconds to minutes for large corpora)
- Load from disk is fast (milliseconds)
- Enables stateless restarts

```python
def load_index(self) -> bool:
    index_path = VECTOR_STORE_PATH / "index.faiss"
    chunks_path = VECTOR_STORE_PATH / "chunks.npy"
    sources_path = VECTOR_STORE_PATH / "sources.npy"
    
    if not all(p.exists() for p in [index_path, chunks_path, sources_path]):
        return False
        
    self.index = faiss.read_index(str(index_path))
    self.chunks = np.load(chunks_path, allow_pickle=True).tolist()
    self.chunk_sources = np.load(sources_path, allow_pickle=True).tolist()
    return True
```

---

#### **Retrieval**

```python
def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, str, float]]:
    self.ensure_index_loaded()
    
    if top_k is None:
        top_k = TOP_K_RESULTS
        
    query_embedding = np.array([self.get_embedding(query)])
    distances, indices = self.index.search(query_embedding, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(self.chunks):
            results.append((
                self.chunks[idx],
                self.chunk_sources[idx],
                float(distances[0][i])
            ))
    return results
```

**Step-by-step:**

1. **Ensure index loaded** (lazy loading)
2. **Embed query** → 384-dim vector
3. **FAISS search** → returns (distances, indices)
   - `distances`: L2 distances to top-k vectors
   - `indices`: positions of top-k vectors in index
4. **Reconstruct results** → (chunk_text, source, distance) tuples

**FAISS search output:**
```python
distances = [[1.234, 1.567, 2.891]]  # L2 distances (lower = more similar)
indices = [[42, 156, 89]]  # Index positions
```

**Final results:**
```python
[
    ("To transfer your domain...", "domain_transfers", 1.234),
    ("Tucows provides support...", "tucows_domain_promise", 1.567),
    ("Domain management...", "domain_management", 2.891)
]
```

---

#### **Context Formatting**

```python
def get_context(self, query: str, top_k: int = None) -> Tuple[str, List[str]]:
    results = self.retrieve(query, top_k)
    
    context_parts = []
    references = []
    
    for chunk, source, _ in results:
        context_parts.append(f"[Source: {source}]\n{chunk}")
        ref = f"Document: {source}"
        if ref not in references:
            references.append(ref)
    
    return "\n\n---\n\n".join(context_parts), references
```

**Output format:**
```
[Source: domain_transfers]
To transfer your domain to another registrar, you must first...

---

[Source: tucows_domain_promise]
Tucows Domains provides comprehensive support...

---

[Source: domain_management]
Managing your domain settings can be done through...
```

**References list:**
```python
["Document: domain_transfers", "Document: tucows_domain_promise", "Document: domain_management"]
```

---

## LLM Integration Deep Dive

### Ollama Architecture

**Ollama:**
- Local LLM server (alternative to cloud APIs)
- Runs models like Llama, Mistral, etc.
- HTTP API on port 11434

**Model: `llama3.2:3b`**
- 3 billion parameters
- ~2GB disk space
- Fast inference on CPU/GPU
- Good instruction-following

---

### Prompt Engineering

**File: `src/llm/llm.py`**

#### **System Prompt**

```python
SYSTEM_PROMPT = """You are a customer support assistant for Tucows.
Analyze tickets and respond based on the provided documentation.

Always respond with JSON in this format:
{
    "answer": "Your response to the customer",
    "references": ["Source documents used"],
    "action_required": "Recommended action or null"
}

Guidelines:
- Be professional and concise
- Only use information from the provided context
- If context is insufficient, say so
- action_required options: "contact_domain_provider", "escalate_to_compliance", 
  "escalate_to_abuse_team", "update_whois_info", "initiate_transfer", or null
"""
```

**Design decisions:**
1. **Role definition:** "customer support assistant for Tucows"
2. **Output constraint:** "Always respond with JSON"
3. **Schema specification:** Explicit field names and types
4. **Behavioral guidelines:**
   - Professional tone
   - Use only provided context (reduce hallucination)
   - Admit when context is insufficient
5. **Enumerated actions:** Predefined action types for consistency

---

#### **User Prompt Template**

```python
USER_PROMPT_TEMPLATE = """
## Context:
{context}

## Ticket:
{query}

Respond with valid JSON only.
"""
```

**Structure:**
- Clear separation between context and query
- Reinforces JSON-only output requirement

---

#### **Prompt Assembly**

```python
def build_prompt(context: str, query: str) -> dict:
    return {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT_TEMPLATE.format(context=context, query=query)
    }
```

**Combined prompt sent to model:**
```
You are a customer support assistant for Tucows.
Analyze tickets and respond based on the provided documentation.
...

## Context:
[Source: domain_transfers]
To transfer your domain...

---

[Source: tucows_domain_promise]
Tucows provides support...

## Ticket:
How do I transfer my domain?

Respond with valid JSON only.
```

---

### Ollama API Integration

#### **Request Format**

```python
payload = {
    "model": self.model,  # "llama3.2:3b"
    "prompt": f"{prompt['system']}\n\n{prompt['user']}",
    "stream": False,
    "format": "json"
}
```

**Key parameters:**
- `model`: Which model to use
- `prompt`: Full combined prompt
- `stream=False`: Return complete response (not streaming)
- `format="json"`: Constrain output to valid JSON

**`format="json"` behavior:**
- Ollama uses grammar-based sampling
- Ensures output is valid JSON
- May still produce incorrect schema (wrong field names)

---

#### **HTTP Request**

```python
async with httpx.AsyncClient(timeout=60.0) as client:
    response = await client.post(f"{self.base_url}/api/generate", json=payload)
    response.raise_for_status()
```

**Details:**
- Endpoint: `POST /api/generate`
- Timeout: 60 seconds (generation can be slow)
- `raise_for_status()`: Raises exception on 4xx/5xx
- Async: Doesn't block FastAPI event loop

---

#### **Response Parsing**

```python
result = response.json()
return self._parse_response(result.get("response", ""))
```

**Ollama response structure:**
```json
{
  "model": "llama3.2:3b",
  "created_at": "2026-01-05T10:30:00.123Z",
  "response": "{\"answer\": \"To transfer...\", \"references\": [...], ...}",
  "done": true,
  "total_duration": 5234000000,
  "load_duration": 234000000,
  "prompt_eval_count": 245,
  "eval_count": 87
}
```

**Fields:**
- `response`: The generated text (JSON string in our case)
- `done`: Whether generation is complete
- Timing fields: useful for monitoring

---

#### **JSON Parsing Logic**

```python
def _parse_response(self, response_text: str) -> TicketResponse:
    try:
        data = json.loads(response_text)
        return TicketResponse(
            answer=data.get("answer", "Unable to generate response."),
            references=data.get("references", []),
            action_required=data.get("action_required")
        )
    except json.JSONDecodeError:
        # Fallback: treat entire response as answer
        return TicketResponse(
            answer=response_text if response_text else "Unable to generate response.",
            references=[],
            action_required=None
        )
```

**Parsing strategy:**
1. **Try to parse as JSON**
2. **Extract fields with fallbacks:**
   - `answer`: Required, default to error message
   - `references`: Default to empty list
   - `action_required`: Default to null
3. **On JSON error:** Use raw text as answer

**Why fallback?**
- Model may produce invalid JSON despite `format="json"`
- Better to return something than crash
- Logs/monitoring should alert on fallback usage

---

### Health Check

```python
async def health_check(self) -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
    except Exception:
        return False
```

**Endpoint: `GET /api/tags`**
- Lists available models
- Doesn't generate anything (fast)
- Used to check if Ollama is running

**Returns:**
- `True` if Ollama reachable
- `False` on any error (connection refused, timeout, etc.)

---

## Data Structures and Formats

### Document Format

**Location:** `data/documents/*.txt`

**Requirements:**
- Plain text files
- UTF-8 encoding
- No specific structure required

**Example: `domain_transfers.txt`**
```
Domain Transfers

To transfer your domain to another registrar, you must:
1. Unlock your domain at your current registrar
2. Obtain an authorization code (EPP code)
3. Initiate the transfer at your new registrar
...
```

---

### Vector Store Format

**Location:** `data/vector_store/`

**Files:**

1. **`index.faiss`** (Binary)
   - FAISS index data structure
   - Contains embedded vectors and index metadata
   - Not human-readable

2. **`chunks.npy`** (NumPy array)
   ```python
   np.load("chunks.npy", allow_pickle=True)
   # Returns: array(['chunk1 text...', 'chunk2 text...', ...], dtype=object)
   ```

3. **`sources.npy`** (NumPy array)
   ```python
   np.load("sources.npy", allow_pickle=True)
   # Returns: array(['domain_transfers', 'about_tucows', ...], dtype=object)
   ```

**Relationship:**
- All three arrays have same length N
- Position i in index → chunks[i] text → sources[i] document name

---

### Request/Response JSON

**Request schema:**
```json
{
  "ticket_text": "string (min_length=1)"
}
```

**Response schema:**
```json
{
  "answer": "string (required)",
  "references": ["string", ...] (default=[]),
  "action_required": "string | null" (default=null)
}
```

**MCP Compliance:**
- Structured output format
- Clear role/task separation in prompt
- Explicit output schema definition
- Verifiable references

---

## Configuration and Environment

### Environment Variables

**Production deployment:**
```bash
export OLLAMA_BASE_URL=http://ollama-server:11434
export OLLAMA_MODEL=llama3.2:3b
export EMBEDDING_MODEL=all-MiniLM-L6-v2
export TOP_K_RESULTS=5
export CHUNK_SIZE=600
export CHUNK_OVERLAP=100
```

**Docker environment:**
```yaml
environment:
  - OLLAMA_BASE_URL=http://host.docker.internal:11434
  - OLLAMA_MODEL=llama3.2:3b
  - EMBEDDING_MODEL=all-MiniLM-L6-v2
  - TOP_K_RESULTS=3
  - CHUNK_SIZE=500
  - CHUNK_OVERLAP=50
```

---

### Configuration Tuning Guide

**`TOP_K_RESULTS`:**
- Higher → more context, but longer prompts and slower
- Lower → faster, but may miss relevant info
- Recommended: 3-5

**`CHUNK_SIZE`:**
- Larger → better context preservation, fewer chunks
- Smaller → more granular retrieval, more chunks
- Recommended: 300-1000 characters

**`CHUNK_OVERLAP`:**
- Higher → better boundary handling, more duplication
- Lower → less redundancy, risk of split concepts
- Recommended: 10-20% of CHUNK_SIZE

**`EMBEDDING_MODEL`:**
- `all-MiniLM-L6-v2`: Fast, small, good baseline
- `all-mpnet-base-v2`: Better quality, slower
- `instructor-large`: Task-specific embeddings

---

## Testing Strategy

### Test Structure

**Test files:**
- `tests/test_api.py`: API endpoint tests
- `tests/test_rag.py`: RAG pipeline tests
- `tests/test_llm.py`: LLM client and prompt tests

### API Tests (`test_api.py`)

**Test categories:**

1. **Endpoint Validation:**
   ```python
   def test_root(self):
       response = client.get("/")
       assert response.status_code == 200
       assert "status" in response.json()
   ```
   - Tests health check endpoint
   - Verifies basic connectivity

2. **Request Validation:**
   ```python
   def test_resolve_ticket_empty(self):
       response = client.post("/resolve-ticket", json={"ticket_text": ""})
       assert response.status_code == 422
   ```
   - Tests Pydantic validation
   - Ensures empty strings are rejected

3. **Happy Path:**
   ```python
   def test_resolve_ticket_valid(self):
       response = client.post(
           "/resolve-ticket",
           json={"ticket_text": "How do I transfer my domain?"}
       )
       assert response.status_code == 200
       data = response.json()
       assert "answer" in data
       assert "references" in data
   ```
   - Tests full pipeline
   - Requires Ollama running

### RAG Tests (`test_rag.py`)

**Test categories:**

1. **Document Loading:**
   ```python
   def test_load_documents(self, rag):
       docs = rag.load_documents()
       assert len(docs) > 0
   ```
   - Verifies documents exist in `data/documents/`

2. **Chunking:**
   ```python
   def test_chunk_text(self, rag):
       chunks = rag.chunk_text("A" * 1000, "test")
       assert len(chunks) > 1
   ```
   - Tests chunking algorithm
   - Verifies overlap logic

3. **Embedding Consistency:**
   ```python
   def test_embedding_consistency(self, rag):
       text = "How do I transfer my domain?"
       emb1 = rag.get_embedding(text)
       emb2 = rag.get_embedding(text)
       np.testing.assert_array_almost_equal(emb1, emb2)
   ```
   - Ensures deterministic embeddings

4. **Retrieval:**
   ```python
   def test_retrieve_top_k(self, rag_loaded):
       results = rag_loaded.retrieve("domain", top_k=2)
       assert len(results) <= 2
   ```
   - Tests FAISS search
   - Verifies top-k limit

5. **Relevance:**
   ```python
   def test_relevance(self, rag_loaded):
       context, _ = rag_loaded.get_context("WHOIS information")
       assert "whois" in context.lower() or "contact" in context.lower()
   ```
   - Sanity check on retrieval quality

### LLM Tests (`test_llm.py`)

**Test categories:**

1. **Prompt Structure:**
   ```python
   def test_system_prompt_has_json_format(self):
       assert "JSON" in SYSTEM_PROMPT or "json" in SYSTEM_PROMPT
       assert "answer" in SYSTEM_PROMPT
       assert "references" in SYSTEM_PROMPT
   ```
   - Verifies prompt includes schema

2. **JSON Parsing:**
   ```python
   def test_parse_valid_json(self, client):
       text = json.dumps({
           "answer": "Test answer",
           "references": ["doc1"],
           "action_required": "contact_domain_provider"
       })
       result = client._parse_response(text)
       assert result.answer == "Test answer"
   ```
   - Tests happy path parsing

3. **Error Handling:**
   ```python
   def test_parse_invalid_json(self, client):
       result = client._parse_response("not json")
       assert result.answer == "not json"
       assert result.references == []
   ```
   - Tests fallback behavior

---

### Running Tests

**Command:**
```bash
pytest tests/ -v
```

**Requirements:**
- `pytest` and `pytest-asyncio` installed
- Dependencies installed (`faiss-cpu`, `sentence-transformers`, etc.)
- Vector store built (or tests will build it)
- Ollama running for full API tests

**Test execution flow:**
1. Pytest discovers test files
2. Fixtures set up test instances
3. Tests run in isolation
4. Teardown (if any)

---

## Deployment Architecture

### Local Development

**Setup:**
```bash
# Terminal 1: Start Ollama
ollama serve
ollama pull llama3.2:3b

# Terminal 2: Start application
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.main
```

**Architecture:**
```
┌─────────────────┐
│   Browser       │
│  localhost:8000 │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│  FastAPI App    │
│  (Python)       │
└────────┬────────┘
         │
         ├──────────► FAISS (in-memory)
         │
         └──────────► Ollama (localhost:11434)
```

---

### Docker Deployment

**Architecture:**
```
┌─────────────────┐
│   Browser       │
│  localhost:8000 │
└────────┬────────┘
         │ HTTP
         ▼
┌──────────────────────────────────┐
│  Docker Container                │
│  ┌────────────────────────────┐  │
│  │   FastAPI App              │  │
│  │   (Python)                 │  │
│  └──────┬─────────────────────┘  │
│         │                         │
│         ├──► FAISS (in-memory)   │
│         │                         │
└─────────┼─────────────────────────┘
          │ host.docker.internal
          ▼
┌─────────────────┐
│  Ollama Server  │
│  (Host Machine) │
└─────────────────┘
```

**docker-compose.yml:**
```yaml
services:
  knowledge-assistant:
    build:
      context: .
      dockerfile: DockerFile
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
    volumes:
      - ./data:/app/data
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

**Key details:**
- `host.docker.internal`: Docker special DNS name for host
- Volume mount: Shares `data/` between host and container
- Port mapping: Container 8000 → Host 8000

**DockerFile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install build tools for dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/data/vector_store

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build process:**
1. Base image: Python 3.11
2. Install system dependencies
3. Install Python packages
4. Copy application code
5. Create vector store directory
6. Run uvicorn server

---

### Production Considerations

**Missing components for production:**

1. **Reverse Proxy:**
   - Nginx or Traefik for HTTPS termination
   - Rate limiting
   - Request logging

2. **Observability:**
   - Structured logging (JSON logs)
   - Metrics (Prometheus)
   - Tracing (OpenTelemetry)
   - Error tracking (Sentry)

3. **Scaling:**
   - Multiple uvicorn workers
   - Load balancer
   - Separate Ollama instances
   - Redis cache for frequent queries

4. **CI/CD:**
   - Automated testing
   - Vector store pre-build
   - Docker image registry
   - Rolling deployments

5. **Security:**
   - API authentication
   - Input sanitization
   - Rate limiting per user
   - CORS configuration

---

## Design Decisions and Tradeoffs

### Embedding Model Choice

**Decision: `all-MiniLM-L6-v2`**

**Pros:**
- Small (23MB)
- Fast inference
- Good quality for semantic search
- Widely used baseline

**Cons:**
- Not domain-specific
- 384 dimensions (some models use 768+)
- Not optimized for Tucows domain terminology

**Alternatives:**
- `all-mpnet-base-v2`: Better quality, 768 dims, slower
- Fine-tuned model: Custom embeddings for domain jargon
- OpenAI embeddings: Cloud API, higher cost

---

### Vector Store Choice

**Decision: FAISS `IndexFlatL2`**

**Pros:**
- Exact search (100% recall)
- Simple, no hyperparameters
- Fast for <100k vectors
- Well-documented

**Cons:**
- O(N) search complexity
- Doesn't scale to millions of vectors
- No filtering capabilities

**Alternatives:**
- `IndexIVFFlat`: Approximate search, faster for large datasets
- `IndexHNSW`: Graph-based, best for large-scale
- Qdrant/Weaviate: Full vector databases with metadata filtering

---

### Chunking Strategy

**Decision: Fixed-size character chunking with overlap**

**Pros:**
- Simple to implement
- Predictable chunk sizes
- Works for any text

**Cons:**
- Ignores sentence boundaries
- May split semantic units
- Overlap creates redundancy

**Alternatives:**
- Sentence-based chunking: Split on periods
- Semantic chunking: Use embeddings to find natural breaks
- Paragraph-based: Split on double newlines
- Recursive splitting: Try paragraphs, then sentences, then chars

---

### LLM Integration

**Decision: Ollama local hosting**

**Pros:**
- No cloud API costs
- Data privacy (everything local)
- No rate limits
- Fast (no network latency)

**Cons:**
- Requires local compute
- Model quality vs. GPT-4/Claude
- Must manage model updates
- Single point of failure

**Alternatives:**
- OpenAI API: Better models, but cost + privacy concerns
- Anthropic Claude: Strong reasoning, cloud-based
- Azure OpenAI: Enterprise-grade cloud
- Hybrid: Ollama for dev, cloud for prod

---

### Prompt Strategy

**Decision: System + user prompt with JSON constraint**

**Pros:**
- Clear output format
- `format="json"` reduces parsing errors
- Explicit guidelines reduce hallucination

**Cons:**
- Model may not follow schema exactly
- No function calling / tool use
- JSON constraint may limit creativity

**Alternatives:**
- Function calling: OpenAI-style tool use
- XML output: More flexible structure
- Free-form + regex: Parse natural language
- JSON schema validation: Retry on invalid schema

---

### Context Assembly

**Decision: Concatenate top-k chunks with separators**

**Pros:**
- Simple to implement
- Preserves source attribution
- Easy to understand

**Cons:**
- May exceed token limits with high k
- Duplicate info if overlap is high
- No reranking step

**Alternatives:**
- Reranking: Use cross-encoder to reorder results
- MMR (Maximal Marginal Relevance): Diversify results
- Hierarchical retrieval: Retrieve docs, then chunks
- Query decomposition: Multiple retrieval rounds

---

### Async vs. Sync

**Decision: Async Ollama client, sync FAISS**

**Pros:**
- Async LLM call doesn't block event loop
- Multiple requests can overlap
- FastAPI natively async

**Cons:**
- Mixed async/sync code
- FAISS search is blocking (but fast)

**Alternatives:**
- Full async: Run FAISS in thread pool
- Full sync: Simpler, but blocks on LLM
- Background tasks: Celery for heavy operations

---

### Error Handling

**Decision: Try/except with fallback responses**

**Pros:**
- Service stays up on errors
- Returns something rather than 500
- Explicit error messages

**Cons:**
- May hide issues
- Fallback responses may confuse users
- No automatic retry

**Alternatives:**
- Fail fast: Return 500 on any error
- Retry logic: Automatic retries for transient failures
- Circuit breaker: Stop calling Ollama if it's down
- Graceful degradation: Return cached responses

---

### Testing Approach

**Decision: Unit tests per module**

**Pros:**
- Fast feedback
- Clear failure localization
- Can run offline (mostly)

**Cons:**
- Requires mocks for full coverage
- End-to-end behavior not fully tested
- Ollama dependency makes some tests flaky

**Alternatives:**
- Integration tests: Test full pipeline
- Contract tests: Verify API contracts
- Load tests: Test under concurrent load
- Golden dataset: Regression testing with known outputs

---

## Performance Characteristics

### Latency Breakdown

**Typical request:**
```
Total: 3000-8000ms
├─ Request parsing: 1-5ms (Pydantic)
├─ Query embedding: 10-50ms (SentenceTransformer)
├─ FAISS search: 1-10ms (depends on corpus size)
├─ LLM generation: 2500-7500ms (Ollama)
└─ Response serialization: 1-5ms (Pydantic)
```

**Bottleneck: LLM generation**
- Llama 3.2 3B generates ~15-30 tokens/sec on CPU
- Typical response: 100-200 tokens
- GPU would reduce to ~500-2000ms

---

### Memory Usage

**Components:**
```
Total: ~1.5-3GB
├─ SentenceTransformer model: ~100MB
├─ FAISS index: ~N × 384 × 4 bytes (N=chunk count)
│  └─ Example: 1000 chunks = ~1.5MB
├─ Chunks in memory: ~N × avg_chunk_size
│  └─ Example: 1000 chunks × 500 chars = ~0.5MB
├─ Ollama (separate process): 2-4GB
└─ Python runtime: ~50-100MB
```

---

### Scalability Limits

**Current architecture:**
- Single process, single worker
- Synchronous FAISS search
- No request queuing

**Limits:**
- ~10-20 concurrent requests (limited by Ollama)
- Corpus size: <100k chunks (FAISS IndexFlatL2)
- Memory: Scales with chunk count

**Scaling strategies:**
1. Horizontal: Multiple FastAPI instances + load balancer
2. Ollama: Dedicated Ollama instances per worker
3. Caching: Redis for frequent queries
4. Index: Switch to IndexIVF for larger corpora

---

## Common Issues and Solutions

### Issue: Vector store not found

**Symptom:**
```
Building index...
```
Every restart rebuilds index (slow).

**Cause:** `data/vector_store/` doesn't exist or is empty.

**Solution:**
```bash
# Pre-build index
python -c "from src.rag.rag import retriever; retriever.build_index()"
```

---

### Issue: Ollama connection refused

**Symptom:**
```
httpx.ConnectError: [Errno 61] Connection refused
```

**Cause:** Ollama not running.

**Solution:**
```bash
ollama serve  # Start Ollama server
```

---

### Issue: Model not found

**Symptom:**
```
{"error": "model 'llama3.2:3b' not found"}
```

**Cause:** Model not downloaded.

**Solution:**
```bash
ollama pull llama3.2:3b
```

---

### Issue: Empty/irrelevant responses

**Symptom:** LLM returns generic answers not based on context.

**Causes:**
- Context not reaching LLM
- Model ignoring context
- Retrieval returning wrong chunks

**Debug:**
1. Print `context` variable before LLM call
2. Check if relevant chunks appear
3. Verify embedding model matches index
4. Try different query phrasing

**Solutions:**
- Increase `TOP_K_RESULTS`
- Rebuild index with better chunking
- Improve prompt to emphasize context usage

---

### Issue: JSON parsing errors

**Symptom:**
```
Fallback: using raw text as answer
```

**Cause:** LLM returns invalid JSON despite `format="json"`.

**Solutions:**
1. Improve prompt to emphasize JSON
2. Add retry logic with stricter instructions
3. Use JSON schema validation
4. Switch to model with better instruction-following

---

### Issue: Slow response times

**Symptom:** Requests take 10+ seconds.

**Causes:**
- CPU-bound LLM inference
- Large context (many chunks)
- Cold model loading

**Solutions:**
1. Use GPU for Ollama
2. Reduce `TOP_K_RESULTS`
3. Cache frequent queries
4. Keep Ollama warm with health checks

---

## Extension Ideas

### 1. Conversational History

**Current:** Stateless (no memory of previous turns)

**Implementation:**
- Add session management
- Store conversation history
- Include history in prompt context
- Limit history to last N turns

---

### 2. Hybrid Search

**Current:** Dense retrieval only (embeddings)

**Enhancement:** Combine with keyword search (BM25)

**Benefits:**
- Better for exact terms (domain names, codes)
- Improved recall

---

### 3. Reranking

**Current:** Return top-k from FAISS directly

**Enhancement:** Use cross-encoder to rerank

**Benefits:**
- Better relevance
- Can consider query-document interaction

---

### 4. Streaming Responses

**Current:** Wait for full LLM response

**Enhancement:** Stream tokens as they generate

**Benefits:**
- Lower perceived latency
- Better UX for long answers

---

### 5. Evaluation Framework

**Current:** Manual testing only

**Enhancement:**
- Build test set of (query, expected_docs, expected_answer)
- Compute metrics: retrieval recall, answer faithfulness, JSON compliance
- Automated regression testing

---

### 6. Observability

**Current:** Minimal logging

**Enhancement:**
- Structured logging with request IDs
- Trace entire request path
- Monitor: latency, error rate, token usage
- Alert on anomalies

---

### 7. Fine-tuning

**Current:** Zero-shot prompting

**Enhancement:**
- Fine-tune embedding model on domain data
- Fine-tune LLM on support ticket examples
- Better understanding of Tucows terminology

---

## Key Takeaways for Interview

1. **Architecture:** RAG = Retrieval (FAISS) + Augmentation (context injection) + Generation (LLM)

2. **Request flow:** API → RAG (embed query, search FAISS) → LLM (build prompt, call Ollama, parse JSON) → Response

3. **Critical decisions:**
   - Embedding model: Semantic similarity
   - FAISS index: Fast exact search
   - Chunking: Balance context vs. granularity
   - Ollama: Local, private, cost-effective

4. **Tradeoffs:**
   - Simple chunking vs. complex semantic splitting
   - Exact search vs. approximate (speed vs. recall)
   - Local LLM vs. cloud API (privacy vs. quality)

5. **Production gaps:**
   - No auth, rate limiting, caching
   - No reranking or query understanding
   - No evaluation metrics
   - Single point of failure (Ollama)

6. **Improvements:**
   - Add reranking with cross-encoder
   - Implement caching layer
   - Build evaluation dataset
   - Add structured logging and monitoring

---

## Study Strategy

1. **Trace code path:** Start from `POST /resolve-ticket`, follow every function call
2. **Run locally:** See actual requests/responses, observe latency
3. **Modify parameters:** Change TOP_K, CHUNK_SIZE, see effects
4. **Break things:** Stop Ollama, delete index, send invalid requests
5. **Read tests:** Understand expected behavior and edge cases
6. **Compare alternatives:** Think about what you'd change for production

---

## Quick Reference Commands

```bash
# Start Ollama
ollama serve
ollama pull llama3.2:3b

# Run app (development)
python -m src.main

# Run app (Docker)
docker-compose up --build

# Run tests
pytest tests/ -v

# Build index manually
python -c "from src.rag.rag import retriever; retriever.build_index()"

# Test endpoint
curl -X POST http://localhost:8000/resolve-ticket \
  -H "Content-Type: application/json" \
  -d '{"ticket_text": "How do I transfer my domain?"}'

# Health check
curl http://localhost:8000/
```

---

**This document covers every component of your RAG system. Study each section, reference the actual code files, and you'll be prepared to discuss any aspect of the implementation.**
