# 🧠 AI Coding Challenge: Knowledge Assistant for Support Team

Welcome to the AI engineering challenge! This is part of the interview process for the AI Engineer role (1–3 years experience). The goal is to design a minimal **LLM-powered RAG system** that helps a support team respond to customer tickets efficiently using relevant documentation.

---

## 📌 Problem Statement

You will build a **Knowledge Assistant** that can analyze customer support queries and return structured, relevant, and helpful responses. The assistant should use a **Retrieval-Augmented Generation (RAG)** pipeline powered by an **LLM** and follow the **Model Context Protocol (MCP)** to produce structured output.

### 🎯 Sample Input (Support Ticket):
```
My domain was suspended and I didn't get any notice. How can I reactivate it?
```
### ✅ Expected Output (MCP-compliant JSON):
```json
{
  "answer": "Your domain may have been suspended due to a violation of policy or missing WHOIS information. Please update your WHOIS details and contact support.",
  "references": ["Policy: Domain Suspension Guidelines, Section 4.2"],
  "action_required": "escalate_to_abuse_team"
}
```

## 🔧 Requirements
### 1.  RAG Pipeline
- Embed sample support docs and policy FAQs (provided or synthetic).
- Use a vector database (e.g., FAISS, Qdrant, etc.) to retrieve context based on the query.

### 2.  LLM Integration
- Use a language model (e.g., OpenAI GPT, LLaMA2 via Ollama, Mistral, etc.)
- Inject context and query into the prompt to generate the final answer.

### 3.  MCP (Model Context Protocol)
- Prompt should have clearly defined role, context, task, and output schema.
- Output must be valid JSON in the following format:
  ```json
  {
    "answer": "...",
    "references": [...],
    "action_required": "..."
  }
  ```
### 4.  API Endpoint
- Expose a single endpoint: POST /resolve-ticket
- Input: { "ticket_text": "..." }
- Output: structured JSON response as shown above

## 📂 Suggested Tech Stack (Use what you're comfortable with)
- Languages: Python or Go
- Embedding Models: Sentence Transformers / OpenAI / HuggingFace
- Vector Store: FAISS, Qdrant, Weaviate, etc.
- LLMs: OpenAI, Ollama, Local LLM, or APIs
- API: FastAPI (Python), Gin/Fiber (Go)
- Docker Compose

## 📝 Scoring Criteria (Total: 100 Points)

| Criteria                     | Description                                                                 | Points |
|------------------------------|-----------------------------------------------------------------------------|--------|
| Correctness & Functionality | Does the assistant generate accurate and relevant responses?                 | 35     |
| RAG Architecture           | Is the retrieval pipeline well-structured, efficient, and properly integrated? | 20     |
| Prompt Design (MCP)        | Is the prompt construction clear, structured, and aligned with MCP principles? | 15     |
| Code Quality & Modularity | Is the code clean, readable, modular, and maintainable and covered with unit tests?                      | 20     |
| Documentation             | Is the `README.md` clear, with setup instructions and design explanation?    | 10     |
|                             | **Total**                                                                   | **100** |

## 🚀 Getting Started
- Fork this repository (do not clone directly)
- Work within your forked copy
- Add your code in /src and include a clear README.md with setup instructions
- Commit your changes regularly
- Once complete, follow the submission instructions below

## 📬 Submission Instructions
- You have 1 week to complete the challenge.
- We expect this to take around 1–2 focused days of work.
- Once complete:
  - Push your forked repo to GitHub
  - Submit the repository link through the portal in the original email.
