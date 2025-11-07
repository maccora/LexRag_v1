# LexRAG: Jurisdiction-Aware Legal Q&A

## Overview

LexRAG is a Retrieval-Augmented Generation (RAG) system designed for legal research. It enables lawyers, students, and analysts to efficiently search and interpret legal documents across multiple jurisdictions. The system combines semantic search capabilities with AI-powered answer generation to provide citation-grounded responses backed by relevant case law and statutes.

The application ingests legal documents from open data sources (primarily CourtListener API), stores them in a vector database with jurisdiction metadata, and uses Mistral AI's models for both embeddings and natural language generation to answer legal questions with proper citations.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Streamlit-based Web Interface**
- The application uses Streamlit for the user interface, providing an interactive web application without requiring complex frontend framework setup
- Single-page application design with sidebar configuration and main content area for Q&A
- Caching strategy using `@st.cache_resource` to maintain vector store and RAG pipeline instances across user sessions
- Lazy initialization pattern: sample data is only loaded when the vector database is empty

### Backend Architecture

**Modular Pipeline Design**
- The system follows a three-tier architecture separating concerns:
  1. **Data Ingestion Layer** (`data_ingestion.py`): Handles fetching and normalizing legal documents
  2. **Storage Layer** (`vector_store.py`): Manages vector embeddings and semantic search
  3. **RAG Pipeline Layer** (`rag_pipeline.py`): Orchestrates retrieval and answer generation

**RAG Flow**
- Query → Vector Search (filtered by jurisdiction) → Context Retrieval → LLM Generation → Citation-Grounded Answer
- The system retrieves top-k relevant documents using semantic similarity, then passes them as context to Mistral's language model for answer synthesis
- Jurisdiction filtering is applied at the vector search level using metadata filters

**Vector Database Strategy**
- ChromaDB serves as the in-memory vector store with persistence capabilities
- Documents are embedded using Mistral's embedding model before storage
- Each document includes metadata: court, jurisdiction, citation, date, case name
- Collection-based organization allows for multiple legal document corpora

### Data Layer

**Document Schema**
- Normalized JSONL format for legal documents containing:
  - Text content (opinion text or statute text)
  - Metadata (jurisdiction type, court identifier, citation, date, case name)
  - Jurisdiction classification (federal vs. state)

**Data Sources**
- Primary: CourtListener API for case law opinions
- Fallback: Sample legal data generated programmatically for demonstration
- Token-based authentication for CourtListener API access
- No persistent database; ChromaDB operates in-memory with optional persistence

### AI/ML Components

**Mistral AI Integration**
- Embedding model for semantic search (converts legal text to vector representations)
- Large language model for answer generation with citation grounding
- Single API key manages both embedding and generation endpoints
- Client initialization pattern ensures API key availability before operations

**Retrieval Strategy**
- Semantic similarity search using cosine distance in vector space
- Configurable top-k retrieval (default: 5 documents)
- Optional jurisdiction filtering to narrow search scope
- Distance scores tracked for relevance assessment

## External Dependencies

### Third-Party APIs

**Mistral AI**
- **Purpose**: Embeddings generation and language model inference
- **Authentication**: API key stored in environment variable `MISTRAL_API_KEY`
- **Models Used**: 
  - Embedding model for document vectorization
  - LLM for question answering and citation generation
- **Integration Point**: `mistralai` Python SDK

**CourtListener API**
- **Purpose**: Fetching real legal opinions and case law
- **Endpoint**: `https://www.courtlistener.com/api/rest/v3`
- **Authentication**: Optional token-based auth via `Authorization` header
- **Rate Limits**: Not explicitly handled in current implementation
- **Data Format**: JSON responses containing opinion metadata and text

### Third-Party Services

**ChromaDB**
- **Purpose**: Vector database for semantic search
- **Deployment**: In-process (no separate server)
- **Storage**: In-memory with optional persistence
- **Configuration**: Anonymous telemetry disabled, reset capability enabled

**Streamlit**
- **Purpose**: Web application framework and hosting
- **Deployment**: Local development server or Streamlit Cloud
- **Caching**: Resource-level caching for expensive initializations

### Python Dependencies

Key libraries (inferred from imports):
- `mistralai`: Mistral AI client SDK
- `chromadb`: Vector database
- `streamlit`: Web UI framework
- `requests`: HTTP client for API calls
- `pandas`: Data manipulation (data ingestion)
- `python-dotenv`: Environment variable management

### Environment Configuration

Required environment variables:
- `MISTRAL_API_KEY`: Authentication for Mistral AI services (required for embeddings and generation)

Optional environment variables:
- CourtListener API token (can be passed programmatically or via environment)