"""
RAG pipeline tests.
"""
import pytest
import numpy as np

from src.rag.rag import RAGPipeline


class TestRAGPipeline:
    
    @pytest.fixture
    def rag(self):
        return RAGPipeline()
    
    @pytest.fixture
    def rag_loaded(self):
        r = RAGPipeline()
        r.ensure_index_loaded()
        return r
    
    # Should load all txt files from data/documents
    def test_load_documents(self, rag):
        docs = rag.load_documents()
        assert len(docs) > 0
        for source, content in docs:
            assert isinstance(source, str)
            assert len(content) > 0
    
    # Long text should be split into multiple chunks
    def test_chunk_text(self, rag):
        chunks = rag.chunk_text("A" * 1000, "test")
        assert len(chunks) > 1
    
    # Should return a numpy array embedding
    def test_get_embedding(self, rag):
        emb = rag.get_embedding("test query")
        assert isinstance(emb, np.ndarray)
        assert len(emb) > 0
    
    # Same text should produce identical embeddings
    def test_embedding_consistency(self, rag):
        text = "How do I transfer my domain?"
        emb1 = rag.get_embedding(text)
        emb2 = rag.get_embedding(text)
        np.testing.assert_array_almost_equal(emb1, emb2)
    
    # Index should be loaded with chunks
    def test_index_loaded(self, rag_loaded):
        assert rag_loaded.index is not None
        assert len(rag_loaded.chunks) > 0
    
    # Should return results with text, source, and distance
    def test_retrieve(self, rag_loaded):
        results = rag_loaded.retrieve("domain transfer")
        assert len(results) > 0
        for chunk, source, dist in results:
            assert isinstance(chunk, str)
            assert isinstance(dist, float)
    
    # Should respect the top_k limit
    def test_retrieve_top_k(self, rag_loaded):
        results = rag_loaded.retrieve("domain", top_k=2)
        assert len(results) <= 2
    
    # Should return formatted context and reference list
    def test_get_context(self, rag_loaded):
        context, refs = rag_loaded.get_context("How do I renew my domain?")
        assert len(context) > 0
        assert len(refs) > 0
    
    # Results should be relevant to the query
    def test_relevance(self, rag_loaded):
        context, _ = rag_loaded.get_context("WHOIS information")
        assert "whois" in context.lower() or "contact" in context.lower()
