"""
LLM client tests.
"""
import pytest
import json

from src.llm.llm import build_prompt, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, OllamaClient
from src.api.schemas import TicketResponse


class TestPrompts:
    
    # System prompt should not be empty
    def test_system_prompt_exists(self):
        assert len(SYSTEM_PROMPT) > 0
    
    # Prompt should specify JSON output format
    def test_system_prompt_has_json_format(self):
        assert "JSON" in SYSTEM_PROMPT or "json" in SYSTEM_PROMPT
        assert "answer" in SYSTEM_PROMPT
        assert "references" in SYSTEM_PROMPT
    
    # Template should have context and query placeholders
    def test_user_template_placeholders(self):
        assert "{context}" in USER_PROMPT_TEMPLATE
        assert "{query}" in USER_PROMPT_TEMPLATE
    
    # build_prompt should combine system and user prompts
    def test_build_prompt(self):
        prompt = build_prompt("context here", "query here")
        assert "system" in prompt
        assert "user" in prompt
        assert "context here" in prompt["user"]
        assert "query here" in prompt["user"]


class TestOllamaClient:
    
    @pytest.fixture
    def client(self):
        return OllamaClient()
    
    # Client should have base_url and model set
    def test_init(self, client):
        assert client.base_url is not None
        assert client.model is not None
    
    # Valid JSON should parse into TicketResponse
    def test_parse_valid_json(self, client):
        text = json.dumps({
            "answer": "Test answer",
            "references": ["doc1"],
            "action_required": "contact_domain_provider"
        })
        result = client._parse_response(text)
        assert result.answer == "Test answer"
        assert result.references == ["doc1"]
    
    # Invalid JSON should fallback to raw text as answer
    def test_parse_invalid_json(self, client):
        result = client._parse_response("not json")
        assert result.answer == "not json"
        assert result.references == []
    
    # Missing fields should use defaults
    def test_parse_partial_json(self, client):
        text = json.dumps({"answer": "Just answer"})
        result = client._parse_response(text)
        assert result.answer == "Just answer"
        assert result.references == []
    
    # Health check should return boolean
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        result = await client.health_check()
        assert isinstance(result, bool)


class TestMCPFormat:
    
    # Response should have all MCP required fields
    def test_response_fields(self):
        resp = TicketResponse(answer="Test", references=[], action_required=None)
        data = resp.model_dump()
        assert "answer" in data
        assert "references" in data
        assert "action_required" in data
    
    # Response should serialize to valid JSON
    def test_response_to_json(self):
        resp = TicketResponse(
            answer="Domain can be transferred",
            references=["domain_transfers"],
            action_required="initiate_transfer"
        )
        parsed = json.loads(resp.model_dump_json())
        assert parsed["answer"] == "Domain can be transferred"
