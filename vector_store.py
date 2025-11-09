import os
import time
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
from mistralai import Mistral
from mistralai.utils.retries import RetryConfig

# Local fallback (lazy-loaded)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class LegalVectorStore:
    """
    Vector database for jurisdiction-aware legal document retrieval.
    Uses ChromaDB with Mistral embeddings, and falls back to a local model
    when the API is rate/capacity-limited (429) or otherwise unavailable.
    """

    def __init__(
        self,
        collection_name: str = "legal_documents",
        mistral_api_key: Optional[str] = None,
        persist_path: str = "./chroma_db",
        use_local_fallback: bool = True,              # <- toggle fallback on/off
        local_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        mistral_embed_model: str = "mistral-embed",   # or "mistral-embed-latest"
        batch_retry_attempts: int = 6,
    ):
        self.collection_name = collection_name
        self.mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
        self.mistral_embed_model = mistral_embed_model
        self.use_local_fallback = use_local_fallback
        self.local_model_name = local_model_name
        self.batch_retry_attempts = batch_retry_attempts

        # Mistral client (optional: only needed for embeddings)
        self.mistral_client = Mistral(api_key=self.mistral_api_key) if self.mistral_api_key else None

        # Local embedder is created lazily on first need
        self.local_embedder = None

        # Persistent Chroma so data survives restarts
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Legal documents with jurisdiction metadata"},
            )
            print(f"Created new collection: {collection_name}")

    # ---------- Embeddings ----------

    def _ensure_local_embedder(self):
        if self.local_embedder is None:
            if not SentenceTransformer:
                raise RuntimeError(
                    "Local embedding fallback requested, but sentence-transformers is not installed."
                )
            print(f"Loading local embedding model: {self.local_model_name}")
            self.local_embedder = SentenceTransformer(self.local_model_name)

    def _embed_with_mistral(self, texts: List[str]) -> List[List[float]]:
        """
        Embed via Mistral with explicit retries (429/5xx). Avoids SDK-internal RetryConfig.
        Compatible across 1.x where signature is input=... (older examples use inputs=...).
        """
        if not self.mistral_client:
            raise ValueError("Mistral API key not configured for Mistral embeddings.")

        attempt = 0
        max_attempts = max(1, int(self.batch_retry_attempts))
        backoff = 1.0

        last_err = None
        while attempt < max_attempts:
            try:
                # Try the modern signature first
                try:
                    resp = self.mistral_client.embeddings.create(
                        model=self.mistral_embed_model,
                        input=texts,   # modern param
                    )
                except TypeError:
                    # Fall back to older param name if needed
                    resp = self.mistral_client.embeddings.create(
                        model=self.mistral_embed_model,
                        inputs=texts,  # legacy param
                    )
                return [d.embedding for d in resp.data]
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # crude status detection; works for typical HTTP client errors
                retryable = any(code in msg for code in [" 429", " 500", " 502", " 503", " 504"])
                if not retryable:
                    break
                attempt += 1
                time.sleep(backoff)
                backoff *= 2.0

        raise RuntimeError(f"Mistral embedding failed after {max_attempts} attempt(s): {last_err}")


    def _embed_with_fallback(self, texts: List[str]) -> List[List[float]]:
        """
        Try Mistral first; on failure (429, service tier capacity, network), fall back to local.
        """
        # Normalize inputs
        safe_texts = [t if isinstance(t, str) else str(t) for t in texts]

        # Try Mistral
        if self.mistral_client:
            try:
                return self._embed_with_mistral(safe_texts)
            except Exception as e:
                print(f"[Embeddings] Mistral failed ({e}).", flush=True)
                if not self.use_local_fallback:
                    raise

        # Local fallback
        self._ensure_local_embedder()
        vecs = self.local_embedder.encode(safe_texts, normalize_embeddings=True)
        return vecs.tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Public embedding helper used by add/search.
        """
        return self._embed_with_fallback(texts)

    # ---------- Indexing ----------

    def add_documents(self, documents: List[Dict], batch_size: int = 2):
        """
        Add legal documents to the vector store with jurisdiction metadata.
        Smaller default batch_size to be gentle to API tiers.
        """
        if not documents:
            print("No documents to add")
            return

        print(f"Adding {len(documents)} documents to vector store...")
        total = len(documents)
        added = 0

        for i in range(0, total, batch_size):
            batch = documents[i: i + batch_size]

            ids = [str(doc.get("id", f"doc_{i + j}")) for j, doc in enumerate(batch)]
            texts = [doc.get("text") or doc.get("snippet", "") for doc in batch]  # (fixed bug: 'batch', not 'm')

            metadatas = []
            for doc in batch:
                metadatas.append(
                    {
                        "case_name": doc.get("case_name", "Unknown"),
                        "citation": doc.get("citation", "N/A"),
                        "court": doc.get("court", "unknown"),
                        "jurisdiction": doc.get("jurisdiction", "unknown"),
                        "date_filed": doc.get("date_filed", ""),
                        "document_type": doc.get("document_type", "case_law"),
                        "url": doc.get("url", ""),
                    }
                )

            try:
                embeddings = self.get_embeddings(texts)  # will auto-fallback if Mistral is unavailable
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
                added += len(batch)
                print(f"Added batch {i // batch_size + 1}/{(total - 1) // batch_size + 1}")
            except Exception as e:
                print(f"Error adding batch {i // batch_size + 1}: {e}")

        print(f"Successfully added {added}/{total} documents")

    # ---------- Query ----------

    def search(self, query: str, jurisdiction: Optional[str] = None, n_results: int = 5) -> Dict:
        """
        Search for relevant legal documents with optional jurisdiction filtering.
        If embedding the query fails, fall back to Chroma's text search so the UI still returns results.
        """
        where_filter = None
        if jurisdiction and jurisdiction != "all":
            where_filter = {"jurisdiction": {"$eq": jurisdiction}}

        # Try embedding search first
        try:
            query_embedding = self.get_embeddings([query])[0]
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
            )
        except Exception as e:
            print(f"[Search] Embedding query failed ({e}); falling back to text search.", flush=True)
            try:
                return self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where_filter,
                )
            except Exception as e2:
                print(f"[Search] Text search also failed: {e2}")
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    # ---------- Stats & Admin ----------

    def get_collection_stats(self) -> Dict:
        count = self.collection.count()
        try:
            all_data = self.collection.get()
            jurisdictions = {}
            for md in all_data.get("metadatas", []):
                jur = (md or {}).get("jurisdiction", "unknown")
                jurisdictions[jur] = jurisdictions.get(jur, 0) + 1
            return {"total_documents": count, "by_jurisdiction": jurisdictions}
        except Exception:
            return {"total_documents": count}

    def reset_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Legal documents with jurisdiction metadata"},
            )
            print(f"Reset collection: {self.collection_name}")
        except Exception as e:
            print(f"Error resetting collection: {e}")
