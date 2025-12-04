"""
Request/response models.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    """Input for /resolve-ticket."""
    ticket_text: str = Field(
        ...,
        description="Support ticket text",
        min_length=1,
        examples=["My domain was suspended. How can I reactivate it?"]
    )


class TicketResponse(BaseModel):
    """MCP-compliant response format."""
    answer: str = Field(..., description="Response to the ticket")
    references: List[str] = Field(default_factory=list, description="Source documents used")
    action_required: Optional[str] = Field(default=None, description="Recommended action")
