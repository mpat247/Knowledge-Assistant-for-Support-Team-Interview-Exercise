# Interview Questions - Knowledge Assistant Project

## 1. Configuration & Environment

### Q1: Why use environment variables instead of hardcoded values?
**Key Points:**
- Security: API keys, URLs shouldn't be in code
- Flexibility: Different configs for dev/staging/production
- Twelve-Factor App methodology
- Easy deployment without code changes

### Q2: Why remove default fallbacks from config.py?
**Key Points:**
- Fail-fast principle: catch missing config at startup
- Prevent production bugs from using dev defaults
- Explicit > Implicit (Zen of Python)
- Forces proper environment setup

### Q3: What happens if OLLAMA_BASE_URL is missing?
**Key Points:**
- `os.getenv()` returns None
- Code tries to use None as URL
- App crashes immediately at startup
- Better than cryptic errors during requests

---

## 2. API Schemas (Pydantic Models)

### Q1: Why use Pydantic instead of plain dictionaries?
**Key Points:**
- Automatic validation: type checking, required fields
- Better errors: tells user exactly what's wrong
- IDE autocomplete: `request.ticket_text` not `request['ticket_text']`
- Auto-generated docs in `/docs`

### Q2: What's the difference between Field default and default_factory?
**Key Points:**
- `default=[]` → Same list shared across ALL instances (mutable default bug)
- `default_factory=list` → New list created for EACH instance
- Python gotcha: mutable defaults are dangerous
- Example: `def func(items=[]): ...` shares same list

### Q3: Why make references and action_required optional?
**Key Points:**
- LLM might not always provide them
- Graceful degradation: answer is most important
- We backfill references from retrieval anyway
- Flexibility in response structure

### Q4: Why min_length=1 for ticket_text?
**Key Points:**
- Empty tickets are useless
- Saves processing time
- FastAPI auto-returns 422 error
- Client gets clear validation error

---

## 3. RAG Pipeline

### Q1: Why chunk documents instead of embedding entire files?
**Key Points:**
- Long documents exceed embedding model limits (512 tokens)
- Retrieve only relevant sections, not entire document
- Better precision: specific paragraphs vs whole file
- Smaller vectors = faster search

### Q2: Explain the chunking parameters (500 chars, 50 overlap)
**Key Points:**
- 500 chars ≈ 1-2 paragraphs (context window)
- 50 char overlap prevents splitting mid-sentence
- Tradeoff: smaller = precise but fragmented, larger = context but noisy
- 500 is sweet spot for domain docs

### Q3: What is FAISS IndexFlatL2 and why use it?
**Key Points:**
- FAISS = Facebook AI Similarity Search (vector database)
- IndexFlatL2 = brute force L2 distance search
- "Flat" = no compression (exact search, not approximate)
- Simple, accurate, fast for ~150 vectors

### Q4: How does vector similarity search work?
**Key Points:**
1. Embed query → 384-dim vector
2. Compare to all chunk embeddings (L2 distance)
3. Sort by distance (closer = more similar)
4. Return top-3 chunks
- Math: sqrt((v1-v2)²) for each dimension

### Q5: Why save/load the index instead of rebuilding each time?
**Key Points:**
- Embedding 150 chunks takes ~5-10 seconds
- Index saved to disk (FAISS, NumPy arrays)
- Load in ~1 second at startup
- Documents don't change often in production

### Q6: Why lazy loading (ensure_index_loaded)?
**Key Points:**
- Don't load index on import (might not be needed)
- Load only when first request arrives
- But we call it in lifespan manager (startup)
- Prevents double-loading

### Q7: What's the difference between get_embedding and retrieve?
**Key Points:**
- `get_embedding`: text → 384-dim vector (uses sentence-transformer)
- `retrieve`: query → search FAISS → return (distances, indices)
- `get_context`: high-level wrapper → formatted string with sources

---

## 4. LLM Client

### Q1: Why use Ollama instead of OpenAI?
**Key Points:**
- Local deployment: no API costs, data privacy
- Runs on CPU (no GPU needed)
- Model choice: llama3.2:3b (small, fast)
- Tradeoff: slower, less capable, but free & private

### Q2: Explain the LLM parameters (temperature, top_p, num_predict)
**Key Points:**
- **Temperature** (0.7): randomness (0=deterministic, 1=creative)
- **Top_p** (0.9): nucleus sampling (consider top 90% probability tokens)
- **Num_predict** (500): max tokens to generate
- Lower temp = consistent answers, higher = diverse

### Q3: Why use format='json' in the request?
**Key Points:**
- Forces LLM to output valid JSON
- Ollama constrains generation to JSON tokens
- Still can fail (LLM ignores instruction)
- We have fallback parsing in `_parse_response`

### Q4: What's the fallback logic in _parse_response?
**Key Points:**
1. Try `json.loads(text)`
2. If fails, look for ```json code block
3. If fails, return plain text as "answer"
4. Always returns TicketResponse (never crashes)
- Defensive programming: don't trust LLM

### Q5: Why async/await for LLM calls?
**Key Points:**
- LLM takes 2-5 seconds to respond
- Blocking = can't handle other requests
- Async = server handles multiple requests concurrently
- FastAPI manages event loop automatically

### Q6: How does the health check work?
**Key Points:**
- Call Ollama `/api/tags` endpoint
- 200 response = healthy, else unhealthy
- Used in `/` health check endpoint
- Could add to monitoring system

---

## 5. API Routes

### Q1: Walk through the /resolve-ticket request flow
**Key Points:**
1. FastAPI validates JSON → TicketRequest
2. Call `retriever.get_context()` → get top-3 chunks
3. Check context exists (raise 500 if not)
4. Call `ollama_client.generate()` → await LLM response
5. Backfill references if LLM forgot them
6. Return TicketResponse → FastAPI serializes to JSON

### Q2: Why backfill references instead of trusting the LLM?
**Key Points:**
- LLM might forget to include references
- We KNOW which docs were used (from retrieval)
- Guarantees citations for traceability
- Could merge instead of replace (better approach)

### Q3: What if the LLM takes longer than 60 seconds?
**Key Points:**
- httpx timeout set to 60 seconds
- Raises `httpx.TimeoutException`
- Caught by except block → 500 error
- Should rarely happen (normal: 2-5 sec)
- Production could have progressive timeouts

### Q4: Why separate HTTPException vs general Exception handling?
**Key Points:**
- `except HTTPException: raise` → don't catch our own errors
- `except Exception:` → wrap unexpected errors
- Prevents double-wrapping
- Consistent error responses

### Q5: How would you add authentication?
**Key Points:**
- Option 1: API key in header (`X-API-Key`)
- Option 2: JWT tokens (standard)
- Option 3: OAuth 2.0 (enterprise)
- FastAPI Depends() for reusable auth
- Middleware for global auth

---

## 6. Application Entry Point

### Q1: What is the lifespan context manager?
**Key Points:**
- Runs code at startup (before yield) and shutdown (after yield)
- We load FAISS index before accepting requests
- Replaces deprecated `@app.on_event("startup")`
- Guarantees cleanup with context manager pattern

### Q2: Walk through app startup sequence
**Key Points:**
1. Python imports (creates retriever, ollama_client)
2. FastAPI app created
3. Router registered
4. Uvicorn starts
5. **Lifespan manager runs → load index**
6. Server ready, accepts requests

### Q3: What's the difference between reload=True in dev vs production?
**Key Points:**
- Dev: watches files, auto-restarts on changes
- Production: NEVER use reload (memory leaks, race conditions)
- Production uses multiple workers (Gunicorn)
- Zero-downtime deployments (rolling restart)

### Q4: Why host="0.0.0.0" instead of "127.0.0.1"?
**Key Points:**
- `127.0.0.1` = localhost only (container can't be reached)
- `0.0.0.0` = all interfaces (needed for Docker)
- Docker maps host:8000 → container:8000
- Production access controlled by firewall/proxy

### Q5: How would you deploy to production?
**Key Points:**
- Multiple workers (Gunicorn + Uvicorn)
- Reverse proxy (Nginx) for HTTPS, rate limiting
- Container orchestration (Kubernetes, Docker Swarm)
- Monitoring (Prometheus, logs)
- Zero-downtime deployments (blue-green, rolling)

---

## 7. Architecture & Design

### Q1: Why separate RAG pipeline from LLM client?
**Key Points:**
- Separation of concerns (retrieval vs generation)
- Could swap embedding models independently
- Could swap LLM providers (OpenAI, Anthropic)
- Easier testing (mock each component)

### Q2: What are the bottlenecks in this system?
**Key Points:**
- **LLM generation**: 2-5 seconds (slowest)
- FAISS search: <50ms (fast)
- Embedding query: ~100ms
- Could parallelize retrieval if multiple queries
- Could cache common questions

### Q3: How would you handle 1000 concurrent requests?
**Key Points:**
- Horizontal scaling: multiple instances behind load balancer
- Queue system: requests → queue → workers process
- Caching: Redis for common questions
- Rate limiting: prevent abuse
- Async helps but doesn't solve capacity

### Q4: What's missing for production readiness?
**Key Points:**
- Authentication & authorization
- Rate limiting
- Monitoring & alerting (Prometheus)
- Structured logging (JSON logs)
- Error tracking (Sentry)
- Load testing & benchmarks
- API versioning

### Q5: How would you test this system?
**Key Points:**
- Unit tests: individual functions (RAG, LLM)
- Integration tests: full endpoint flow
- Mock Ollama responses for deterministic tests
- Test edge cases (empty context, timeouts)
- Load testing (Locust, k6)

---

## 8. RAG & LLM Concepts

### Q1: What is RAG and why use it?
**Key Points:**
- Retrieval-Augmented Generation
- Retrieves relevant context → feeds to LLM
- Prevents hallucination (LLM uses real docs)
- No fine-tuning needed (works with any LLM)
- More accurate than LLM alone

### Q2: What are embeddings?
**Key Points:**
- Text → vector of numbers (semantic representation)
- Similar meaning → similar vectors
- 384 dimensions (all-MiniLM-L6-v2)
- Enables similarity search

### Q3: What are tokens?
**Key Points:**
- Text → chunks for LLM processing
- ~4 characters per token on average
- "Hello world" ≈ 2 tokens
- Models have token limits (context window)

### Q4: Horizontal vs vertical scaling?
**Key Points:**
- **Vertical**: bigger machine (more CPU, RAM)
- **Horizontal**: more machines (load balancer)
- Horizontal preferred (better redundancy)
- This project: horizontal (stateless API)

---

## 9. Security & Best Practices

### Q1: What security vulnerabilities exist?
**Key Points:**
- **Prompt injection**: user could manipulate LLM
- No input sanitization
- No authentication
- Acceptable for demo, not production

### Q2: How would you prevent prompt injection?
**Key Points:**
- Input validation: max length, character whitelist
- Separate system prompt from user input
- LLM guardrails (detect malicious prompts)
- Content filtering on output
- No perfect solution (ongoing research)

### Q3: Why async/await throughout?
**Key Points:**
- Non-blocking I/O (network calls)
- Handle concurrent requests efficiently
- FastAPI is async-first framework
- Alternative: threads (less efficient)

### Q4: Why Pydantic validation is important?
**Key Points:**
- Prevents bad data from reaching business logic
- Type safety (no `KeyError` or `TypeError`)
- Auto-generated error messages
- Documentation (OpenAPI schema)

---

## 10. Docker & Deployment

### Q1: Why use Docker for this project?
**Key Points:**
- Consistent environment (works on any machine)
- Isolates dependencies
- Easy deployment (same image everywhere)
- docker-compose orchestrates app + Ollama

### Q2: What does docker-compose do?
**Key Points:**
- Defines multi-container application
- App service + Ollama service
- Networking between containers
- Volume mounts for persistence

### Q3: Why volume mount data/ directory?
**Key Points:**
- Vector store persists across container restarts
- Documents accessible to container
- Don't rebuild index every time
- Production: use cloud storage (S3)

---

## Bonus: Tradeoffs & Decisions

### Q1: Why llama3.2:3b instead of larger model?
**Key Points:**
- 3B params = runs on CPU (no GPU needed)
- Fast inference (2-5 sec)
- Good enough for domain QA
- Tradeoff: less capable than GPT-4

### Q2: Why 3 chunks (TOP_K_RESULTS=3)?
**Key Points:**
- Tradeoff: more context vs noise
- 3 chunks ≈ 1500 chars (fits in prompt)
- Enough context, not overwhelming
- Could make dynamic based on query

### Q3: Why sentence-transformers vs OpenAI embeddings?
**Key Points:**
- Local = free & private
- all-MiniLM-L6-v2 = small, fast
- 384 dims (vs 1536 for OpenAI)
- Good quality for domain docs

### Q4: Why synchronous index building vs async?
**Key Points:**
- Index built once at startup
- Not worth async complexity
- Blocking is fine (before serving requests)
- Loading is already async (on startup)
