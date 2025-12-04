"""
RAG pipeline: embeddings, indexing, and retrieval.
"""
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.config import DATA_DIR, VECTOR_STORE_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS


class RAGPipeline:
    """Handles document embeddings and FAISS retrieval."""
    
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.chunks = []
        self.chunk_sources = []
        
    def load_documents(self) -> List[Tuple[str, str]]:
        """Load all .txt files from data directory."""
        documents = []
        for file_path in DATA_DIR.glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append((file_path.stem, f.read()))
        return documents
    
    def chunk_text(self, text: str, source: str) -> List[Tuple[str, str]]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            if chunk.strip():
                chunks.append((chunk.strip(), source))
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks
    
    def build_index(self) -> None:
        """Build FAISS index from documents."""
        documents = self.load_documents()
        
        all_chunks = []
        for source, content in documents:
            all_chunks.extend(self.chunk_text(content, source))
        
        self.chunks = [chunk for chunk, _ in all_chunks]
        self.chunk_sources = [source for _, source in all_chunks]
        
        embeddings = self.model.encode(self.chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        self._save_index()
        
    def _save_index(self) -> None:
        """Save index to disk."""
        VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(VECTOR_STORE_PATH / "index.faiss"))
        np.save(VECTOR_STORE_PATH / "chunks.npy", np.array(self.chunks, dtype=object))
        np.save(VECTOR_STORE_PATH / "sources.npy", np.array(self.chunk_sources, dtype=object))
        
    def load_index(self) -> bool:
        """Load index from disk. Returns False if not found."""
        index_path = VECTOR_STORE_PATH / "index.faiss"
        chunks_path = VECTOR_STORE_PATH / "chunks.npy"
        sources_path = VECTOR_STORE_PATH / "sources.npy"
        
        if not all(p.exists() for p in [index_path, chunks_path, sources_path]):
            return False
            
        self.index = faiss.read_index(str(index_path))
        self.chunks = np.load(chunks_path, allow_pickle=True).tolist()
        self.chunk_sources = np.load(sources_path, allow_pickle=True).tolist()
        return True
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text."""
        return self.model.encode([text])[0].astype('float32')
    
    def ensure_index_loaded(self) -> None:
        """Load index or build it if missing."""
        if self.index is None:
            if not self.load_index():
                print("Building index...")
                self.build_index()
    
    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, str, float]]:
        """Find similar chunks for a query. Returns (text, source, distance)."""
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
    
    def get_context(self, query: str, top_k: int = None) -> Tuple[str, List[str]]:
        """Get formatted context string and reference list for the LLM."""
        results = self.retrieve(query, top_k)
        
        context_parts = []
        references = []
        
        for chunk, source, _ in results:
            context_parts.append(f"[Source: {source}]\n{chunk}")
            ref = f"Document: {source}"
            if ref not in references:
                references.append(ref)
        
        return "\n\n---\n\n".join(context_parts), references


retriever = RAGPipeline()
