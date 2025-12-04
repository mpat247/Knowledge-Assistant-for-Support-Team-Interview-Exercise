"""
API tests.
"""
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.api.schemas import TicketRequest, TicketResponse


client = TestClient(app)


class TestEndpoints:
    
    # Check root endpoint returns health status
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "status" in response.json()
    
    # Valid ticket should return answer with references
    def test_resolve_ticket_valid(self):
        response = client.post(
            "/resolve-ticket",
            json={"ticket_text": "How do I transfer my domain?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "references" in data
    
    # Empty ticket text should fail validation
    def test_resolve_ticket_empty(self):
        response = client.post("/resolve-ticket", json={"ticket_text": ""})
        assert response.status_code == 422
    
    # Missing ticket_text field should fail validation
    def test_resolve_ticket_missing_field(self):
        response = client.post("/resolve-ticket", json={})
        assert response.status_code == 422


class TestSchemas:
    
    # Valid request should parse correctly
    def test_request_valid(self):
        req = TicketRequest(ticket_text="My domain expired")
        assert req.ticket_text == "My domain expired"
    
    # Empty string should raise validation error
    def test_request_empty_fails(self):
        with pytest.raises(ValueError):
            TicketRequest(ticket_text="")
    
    # Response with all fields should work
    def test_response_valid(self):
        resp = TicketResponse(
            answer="Contact support",
            references=["doc1"],
            action_required="contact_domain_provider"
        )
        assert resp.answer == "Contact support"
    
    # Response should have sensible defaults
    def test_response_defaults(self):
        resp = TicketResponse(answer="Test")
        assert resp.references == []
        assert resp.action_required is None
