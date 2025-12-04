"""
FastAPI app entry point.
"""
from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.api.route import router
from src.rag.rag import retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load index on startup."""
    retriever.ensure_index_loaded()
    yield


app = FastAPI(
    title="Knowledge Assistant API",
    description="RAG-powered support ticket resolution for Tucows",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
