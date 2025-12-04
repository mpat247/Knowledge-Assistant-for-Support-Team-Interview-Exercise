"""
LLM client and prompt templates.
"""
import json
import httpx

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from src.api.schemas import TicketResponse


SYSTEM_PROMPT = """You are a customer support assistant for Tucowsr.
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
- action_required options: "contact_domain_provider", "escalate_to_compliance", "escalate_to_abuse_team", "update_whois_info", "initiate_transfer", or null
"""

USER_PROMPT_TEMPLATE = """
## Context:
{context}

## Ticket:
{query}

Respond with valid JSON only.
"""


def build_prompt(context: str, query: str) -> dict:
    """Build system and user prompts."""
    return {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT_TEMPLATE.format(context=context, query=query)
    }


class OllamaClient:
    """Ollama API client."""
    
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = OLLAMA_MODEL
        
    async def generate(self, context: str, query: str) -> TicketResponse:
        """Generate a response from Ollama."""
        prompt = build_prompt(context, query)
        
        payload = {
            "model": self.model,
            "prompt": f"{prompt['system']}\n\n{prompt['user']}",
            "stream": False,
            "format": "json"
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            
        result = response.json()
        return self._parse_response(result.get("response", ""))
    
    def _parse_response(self, response_text: str) -> TicketResponse:
        """Parse JSON response from LLM."""
        try:
            data = json.loads(response_text)
            return TicketResponse(
                answer=data.get("answer", "Unable to generate response."),
                references=data.get("references", []),
                action_required=data.get("action_required")
            )
        except json.JSONDecodeError:
            return TicketResponse(
                answer=response_text if response_text else "Unable to generate response.",
                references=[],
                action_required=None
            )
    
    async def health_check(self) -> bool:
        """Check if Ollama is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False


ollama_client = OllamaClient()
