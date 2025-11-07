from mistralai import Mistral
from typing import List, Dict, Optional
import os


class LegalRAGPipeline:
    """
    RAG pipeline for legal question answering with citation grounding.
    Uses Mistral models for generation.
    """
    
    def __init__(self, vector_store, mistral_api_key: Optional[str] = None):
        self.vector_store = vector_store
        self.mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
        
        if self.mistral_api_key:
            self.mistral_client = Mistral(api_key=self.mistral_api_key)
        else:
            self.mistral_client = None
    
    def retrieve_context(self, query: str, jurisdiction: Optional[str] = None, 
                        n_results: int = 5) -> List[Dict]:
        """
        Retrieve relevant legal documents for the query.
        
        Args:
            query: User question
            jurisdiction: Filter by jurisdiction ('federal', 'state', or None)
            n_results: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        results = self.vector_store.search(
            query=query,
            jurisdiction=jurisdiction,
            n_results=n_results
        )
        
        retrieved_docs = []
        if results and results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                doc = {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results.get("distances") else None
                }
                retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents into context string with citations.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant legal precedents found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc.get("metadata", {})
            case_name = metadata.get("case_name", "Unknown Case")
            citation = metadata.get("citation", "N/A")
            jurisdiction = metadata.get("jurisdiction", "unknown")
            date = metadata.get("date_filed", "")
            text = doc.get("text", "")
            
            context_parts.append(
                f"[{i}] {case_name}, {citation} ({jurisdiction.upper()}, {date})\n"
                f"{text}\n"
            )
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str, 
                       model: str = "mistral-large-latest") -> str:
        """
        Generate citation-grounded answer using Mistral model.
        
        Args:
            query: User question
            context: Retrieved legal context
            model: Mistral model to use
            
        Returns:
            Generated answer with citations
        """
        if not self.mistral_client:
            return "Error: Mistral API key not configured. Please add your API key to use the answer generation feature."
        
        system_prompt = """You are a legal research assistant specializing in case law analysis. 
Your role is to provide accurate, citation-grounded answers to legal questions.

IMPORTANT INSTRUCTIONS:
1. Base your answers ONLY on the provided legal sources
2. Always cite specific cases using their full citations (e.g., "Smith v. Jones, 123 F.3d 456")
3. Reference the source number in brackets [1], [2], etc. when citing
4. Clearly distinguish between federal and state law when relevant
5. If the sources don't contain enough information, acknowledge the limitation
6. Provide clear, professional legal analysis
7. Do not make up cases or citations not present in the sources

Format your response with:
- A direct answer to the question
- Supporting analysis with specific citations
- Relevant case law references
- Any important limitations or caveats"""

        user_message = f"""Legal Question: {query}

Relevant Legal Sources:
{context}

Please provide a comprehensive answer to the legal question based on the sources provided. Include specific citations and analyze how the cases apply to the question."""

        try:
            response = self.mistral_client.chat.complete(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def answer_question(self, query: str, jurisdiction: Optional[str] = None,
                       n_results: int = 5, model: str = "mistral-large-latest") -> Dict:
        """
        Complete RAG pipeline: retrieve, format, and generate answer.
        
        Args:
            query: User question
            jurisdiction: Filter by jurisdiction
            n_results: Number of sources to retrieve
            model: Mistral model to use
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        retrieved_docs = self.retrieve_context(query, jurisdiction, n_results)
        
        context = self.format_context(retrieved_docs)
        
        answer = self.generate_answer(query, context, model)
        
        return {
            "question": query,
            "answer": answer,
            "sources": retrieved_docs,
            "jurisdiction_filter": jurisdiction or "all",
            "num_sources": len(retrieved_docs)
        }
    
    def format_sources_for_display(self, sources: List[Dict]) -> List[Dict]:
        """
        Format sources for UI display.
        
        Args:
            sources: List of source documents
            
        Returns:
            List of formatted source dictionaries
        """
        formatted = []
        for i, doc in enumerate(sources, 1):
            metadata = doc.get("metadata", {})
            formatted.append({
                "number": i,
                "case_name": metadata.get("case_name", "Unknown Case"),
                "citation": metadata.get("citation", "N/A"),
                "jurisdiction": metadata.get("jurisdiction", "unknown").upper(),
                "date": metadata.get("date_filed", ""),
                "court": metadata.get("court", "unknown"),
                "text": doc.get("text", "")[:500] + "..." if len(doc.get("text", "")) > 500 else doc.get("text", ""),
                "relevance": f"{(1 - doc.get('distance', 0)):.2%}" if doc.get('distance') is not None else "N/A",
                "url": metadata.get("url", "")
            })
        return formatted
