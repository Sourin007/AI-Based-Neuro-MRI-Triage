from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
import re

from .config import Settings
from .state import RetrievedContextItem

try:
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma

    LANGCHAIN_RAG_AVAILABLE = True
except ImportError:
    Document = None
    RecursiveCharacterTextSplitter = None
    Chroma = None
    LANGCHAIN_RAG_AVAILABLE = False


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class ChunkRecord:
    source: str
    content: str


class DeterministicHashEmbeddings:
    """Embedding fallback that avoids external model downloads."""

    def __init__(self, dimensions: int = 96):
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) % self.dimensions
            sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class MedicalKnowledgeBase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._fallback_chunks = self._load_fallback_chunks()
        self._vector_store = self._build_vector_store()

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedContextItem]:
        top_k = k or self.settings.retrieval_k
        if self._vector_store is not None:
            results = self._vector_store.similarity_search_with_score(query, k=top_k)
            context: list[RetrievedContextItem] = []
            for document, score in results:
                metadata = document.metadata or {}
                context.append(
                    RetrievedContextItem(
                        source=metadata.get("source", "knowledge_base"),
                        content=document.page_content,
                        relevance_score=float(1 / (1 + score)),
                    )
                )
            if context:
                return context

        return self._fallback_retrieve(query, top_k)

    def _build_vector_store(self):
        if not LANGCHAIN_RAG_AVAILABLE:
            return None

        documents = [
            Document(page_content=chunk.content, metadata={"source": chunk.source})
            for chunk in self._fallback_chunks
        ]
        if not documents:
            return None

        os.makedirs(self.settings.vector_store_dir, exist_ok=True)
        return Chroma.from_documents(
            documents=documents,
            embedding=DeterministicHashEmbeddings(),
            persist_directory=self.settings.vector_store_dir,
            collection_name="brain_tumor_knowledge",
        )

    def _load_fallback_chunks(self) -> list[ChunkRecord]:
        corpus = self._load_corpus_files()
        if not corpus:
            return []

        if LANGCHAIN_RAG_AVAILABLE:
            splitter = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=120)
            records: list[ChunkRecord] = []
            for source, content in corpus:
                for chunk in splitter.split_text(content):
                    records.append(ChunkRecord(source=source, content=chunk))
            return records

        records: list[ChunkRecord] = []
        for source, content in corpus:
            parts = [part.strip() for part in content.split("\n\n") if part.strip()]
            records.extend(ChunkRecord(source=source, content=part) for part in parts)
        return records

    def _load_corpus_files(self) -> list[tuple[str, str]]:
        directory = self.settings.knowledge_base_dir
        if not os.path.isdir(directory):
            return []

        corpus: list[tuple[str, str]] = []
        for filename in sorted(os.listdir(directory)):
            if not filename.endswith((".md", ".txt")):
                continue
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as handle:
                corpus.append((filename, handle.read()))
        return corpus

    def _fallback_retrieve(self, query: str, k: int) -> list[RetrievedContextItem]:
        query_tokens = set(_tokenize(query))
        scored: list[tuple[float, ChunkRecord]] = []
        for chunk in self._fallback_chunks:
            tokens = set(_tokenize(chunk.content))
            overlap = len(query_tokens & tokens)
            if overlap == 0:
                continue
            score = overlap / max(len(query_tokens), 1)
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedContextItem(
                source=chunk.source,
                content=chunk.content,
                relevance_score=float(score),
            )
            for score, chunk in scored[:k]
        ]
