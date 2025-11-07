import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from mistralai import Mistral
import os


class LegalVectorStore:
    """
    Vector database for jurisdiction-aware legal document retrieval.
    Uses ChromaDB with Mistral embeddings.
    """
    
    def __init__(self, collection_name: str = "legal_documents", 
                 mistral_api_key: Optional[str] = None):
        self.collection_name = collection_name
        self.mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
        
        if self.mistral_api_key:
            self.mistral_client = Mistral(api_key=self.mistral_api_key)
        else:
            self.mistral_client = None
        
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Legal documents with jurisdiction metadata"}
            )
            print(f"Created new collection: {collection_name}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Mistral's embedding model.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.mistral_client:
            raise ValueError("Mistral API key not configured")
        
        try:
            response = self.mistral_client.embeddings.create(
                model="mistral-embed",
                inputs=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            raise
    
    def add_documents(self, documents: List[Dict], batch_size: int = 10):
        """
        Add legal documents to the vector store with jurisdiction metadata.
        
        Args:
            documents: List of document dictionaries
            batch_size: Number of documents to process at once
        """
        if not documents:
            print("No documents to add")
            return
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            ids = [str(doc.get("id", f"doc_{i+j}")) for j, doc in enumerate(batch)]
            
            texts = [doc.get("text", doc.get("snippet", "")) for doc in batch]
            
            metadatas = []
            for doc in batch:
                metadata = {
                    "case_name": doc.get("case_name", "Unknown"),
                    "citation": doc.get("citation", "N/A"),
                    "court": doc.get("court", "unknown"),
                    "jurisdiction": doc.get("jurisdiction", "unknown"),
                    "date_filed": doc.get("date_filed", ""),
                    "document_type": doc.get("document_type", "case_law"),
                    "url": doc.get("url", "")
                }
                metadatas.append(metadata)
            
            try:
                if self.mistral_client:
                    embeddings = self.get_embeddings(texts)
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=texts,
                        metadatas=metadatas
                    )
                else:
                    self.collection.add(
                        ids=ids,
                        documents=texts,
                        metadatas=metadatas
                    )
                
                print(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            except Exception as e:
                print(f"Error adding batch: {e}")
        
        print(f"Successfully added {len(documents)} documents")
    
    def search(self, query: str, jurisdiction: Optional[str] = None, 
               n_results: int = 5) -> Dict:
        """
        Search for relevant legal documents with optional jurisdiction filtering.
        
        Args:
            query: Search query
            jurisdiction: Filter by 'federal', 'state', or None for all
            n_results: Number of results to return
            
        Returns:
            Dictionary with documents, metadatas, and distances
        """
        where_filter = None
        if jurisdiction and jurisdiction != "all":
            where_filter = {"jurisdiction": jurisdiction}
        
        try:
            if self.mistral_client:
                query_embedding = self.get_embeddings([query])[0]
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where_filter
                )
            else:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where_filter
                )
            
            return results
        except Exception as e:
            print(f"Error searching: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the document collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        
        try:
            all_data = self.collection.get()
            jurisdictions = {}
            for metadata in all_data.get("metadatas", []):
                jur = metadata.get("jurisdiction", "unknown")
                jurisdictions[jur] = jurisdictions.get(jur, 0) + 1
            
            return {
                "total_documents": count,
                "by_jurisdiction": jurisdictions
            }
        except:
            return {"total_documents": count}
    
    def reset_collection(self):
        """
        Delete and recreate the collection.
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Legal documents with jurisdiction metadata"}
            )
            print(f"Reset collection: {self.collection_name}")
        except Exception as e:
            print(f"Error resetting collection: {e}")
