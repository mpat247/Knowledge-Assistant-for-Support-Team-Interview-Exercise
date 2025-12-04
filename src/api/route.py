"""
API routes.
"""
from fastapi import APIRouter, HTTPException

from src.api.schemas import TicketRequest, TicketResponse
from src.rag.rag import retriever
from src.llm.llm import ollama_client

router = APIRouter()


@router.post("/resolve-ticket", response_model=TicketResponse)
async def resolve_ticket(request: TicketRequest) -> TicketResponse:
    """Process a support ticket through the RAG pipeline."""
    try:
        # Get relevant context
        context, references = retriever.get_context(request.ticket_text)
        
        if not context:
            raise HTTPException(status_code=500, detail="No relevant context found")
        
        # Generate response
        response = await ollama_client.generate(context, request.ticket_text)
        
        # Add references if missing
        if not response.references:
            response.references = references
            
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/")
async def health_check():
    """Check if services are running."""
    ollama_healthy = await ollama_client.health_check()
    return {
        "status": "healthy" if ollama_healthy else "degraded",
        "ollama": "connected" if ollama_healthy else "disconnected"
    }
