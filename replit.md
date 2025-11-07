# LexRAG: Jurisdiction-Aware Legal Q&A

## Overview

LexRAG is an advanced Retrieval-Augmented Generation (RAG) system designed for legal research. It enables lawyers, students, and analysts to efficiently search and interpret legal documents across multiple jurisdictions. The system combines semantic search capabilities with AI-powered answer generation, multi-step agentic reasoning, and AI-based evaluation to provide citation-grounded responses backed by relevant case law, statutes, and regulations.

The application ingests legal documents from multiple open data sources (CourtListener, GovInfo, eCFR, Regulations.gov), stores them in a vector database with jurisdiction metadata, and uses Mistral AI's models for embeddings, generation, and evaluation.

## Recent Updates (November 2025)

**New Features:**
- 🤖 Multi-step agentic reasoning with jurisdiction verification and citation cross-checking
- 📊 AI-as-judge evaluation framework for answer quality scoring
- 📈 Advanced retrieval metrics dashboard (Recall@K, MRR, NDCG)
- 💬 User feedback collection and analysis system
- 🌐 Multi-source data integration (GovInfo, eCFR, Regulations.gov)
- ⚡ Mistral Small as default model (4x faster, 1/3 cost, 85-90% accuracy)

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Streamlit-based Web Interface**
- Multi-tab application with six main sections:
  1. Ask Question - Basic Q&A with citation-grounded answers
  2. Agentic Research - Multi-step reasoning with verification
  3. Evaluation & Metrics - AI judge scoring and retrieval metrics
  4. Browse Sources - Document corpus exploration
  5. Feedback - User ratings and quality analytics
  6. Manage Data - Multi-source data ingestion and management

- Caching strategy using `@st.cache_resource` to maintain vector store and RAG pipeline instances across user sessions
- Lazy initialization pattern: sample data is only loaded when the vector database is empty
- Interactive controls for jurisdiction filtering, model selection, and agentic mode

### Backend Architecture

**Modular Pipeline Design**
The system follows a modular architecture with clear separation of concerns:

1. **Data Ingestion Layer**:
   - `data_ingestion.py`: CourtListener API integration for case law
   - `data_sources.py`: GovInfo, eCFR, and Regulations.gov integration for federal regulations

2. **Storage Layer**:
   - `vector_store.py`: ChromaDB vector database with Mistral embeddings
   - Jurisdiction-aware metadata filtering
   - Batch processing for efficient document ingestion

3. **RAG Pipeline Layer**:
   - `rag_pipeline.py`: Orchestrates retrieval and answer generation
   - Context formatting with citation preservation
   - Mistral-powered answer synthesis

4. **Advanced Features**:
   - `agent.py`: Multi-step agentic reasoning with jurisdiction verification and citation cross-checking
   - `evaluation.py`: AI-as-judge evaluation framework using Mistral for quality scoring
   - `metrics.py`: Advanced retrieval metrics (Recall@K, Precision@K, MRR, NDCG, AP)
   - `feedback.py`: User feedback collection, analysis, and recommendations

**RAG Flow**
- Query → Vector Search (filtered by jurisdiction) → Context Retrieval → LLM Generation → Citation-Grounded Answer
- The system retrieves top-k relevant documents using semantic similarity, then passes them as context to Mistral's language model for answer synthesis
- Jurisdiction filtering is applied at the vector search level using metadata filters
- Distance scores tracked for relevance assessment and metrics calculation

**Agentic Research Flow**
1. **Jurisdiction Analysis**: Determine appropriate jurisdiction from query
2. **Document Retrieval**: Fetch relevant documents with jurisdiction filtering
3. **Citation Verification**: Validate citation formats and completeness
4. **Consistency Checking**: Ensure jurisdictional alignment across sources
5. **Answer Generation**: Synthesize verified, citation-grounded response

**Evaluation Pipeline**
- AI judge evaluates answers on:
  - Factual accuracy (alignment with sources)
  - Citation validity (correct attribution and formatting)
  - Jurisdictional alignment (proper federal/state distinction)
  - Completeness (fully addresses question)
  - Clarity (well-organized and understandable)
- JSON-structured evaluation output with scores, strengths, weaknesses, and feedback

**Vector Database Strategy**
- ChromaDB serves as the in-memory vector store with persistence capabilities
- Documents are embedded using Mistral's `mistral-embed` model before storage
- Each document includes metadata: court, jurisdiction, citation, date, case name, document type
- Collection-based organization allows for multiple legal document corpora
- Supports filtering by jurisdiction, date range, and document type

### Data Layer

**Document Schema**
- Normalized JSONL format for legal documents containing:
  - `id`: Unique identifier
  - `case_name`: Full case or regulation name
  - `citation`: Legal citation (e.g., "123 F.3d 456")
  - `court`: Court identifier or source
  - `jurisdiction`: "federal" or "state"
  - `date_filed`: Filing or effective date
  - `text`: Full document text or summary
  - `snippet`: Brief excerpt
  - `url`: Link to full document
  - `document_type`: "case_law", "regulation", "regulatory_comment"

**Data Sources**
1. **CourtListener API** (case law opinions)
   - Endpoint: `https://www.courtlistener.com/api/rest/v3`
   - Optional token-based authentication
   - Search by keyword with jurisdiction filtering

2. **GovInfo API** (federal regulations and statutes)
   - Endpoint: `https://api.govinfo.gov`
   - Access to CFR (Code of Federal Regulations)
   - Optional API key for higher rate limits

3. **eCFR API** (Electronic Code of Federal Regulations)
   - Endpoint: `https://www.ecfr.gov/api/search/v1`
   - Current CFR sections and titles
   - No authentication required

4. **Regulations.gov API** (rulemaking and regulatory comments)
   - Endpoint: `https://api.regulations.gov/v4`
   - Optional API key via `X-Api-Key` header
   - Access to proposed rules and public comments

5. **Sample Data**: Programmatically generated for demonstration (8 case law + 4 regulatory documents)

### AI/ML Components

**Mistral AI Integration**
- **Embedding Model** (`mistral-embed`): Converts legal text to 1024-dimensional vectors for semantic search
- **Generation Models**:
  - `mistral-small-latest` (default): 22B parameters, 4x faster, 1/3 cost, 85-90% accuracy
  - `mistral-medium-latest`: ~70B parameters, balanced performance
  - `mistral-large-latest`: 123B parameters, maximum accuracy, reduced hallucinations
- **Evaluation Model**: Uses same generation models for AI-as-judge scoring
- Single API key manages all Mistral endpoints
- JSON mode support for structured evaluation outputs

**Retrieval Strategy**
- Semantic similarity search using cosine distance in vector space
- Configurable top-k retrieval (1-10 documents, default: 5)
- Optional jurisdiction filtering ("federal", "state", or "all")
- Distance scores tracked for relevance assessment and NDCG calculation
- Batch processing for efficient embedding generation

**Metrics & Analytics**
- **Retrieval Metrics**: Recall@K, Precision@K, MRR, NDCG@K, Average Precision
- **Evaluation Metrics**: Factual accuracy, citation validity, jurisdictional alignment, completeness, clarity
- **User Feedback**: Star ratings (1-5), text comments, jurisdiction-based aggregation
- **Quality Analytics**: Hallucination detection, improvement area identification, recommendation generation

## External Dependencies

### Third-Party APIs

**Mistral AI**
- **Purpose**: Embeddings generation, language model inference, and evaluation
- **Authentication**: API key stored in environment variable `MISTRAL_API_KEY`
- **Models Used**: 
  - `mistral-embed` for document vectorization
  - `mistral-small-latest`, `mistral-medium-latest`, `mistral-large-latest` for generation
- **Integration Point**: `mistralai` Python SDK (v1.9.11)
- **Cost**: $1.50/M tokens (blended) for Small, $4.50/M for Large

**CourtListener API**
- **Purpose**: Fetching real legal opinions and case law
- **Endpoint**: `https://www.courtlistener.com/api/rest/v3`
- **Authentication**: Optional token-based auth via `Authorization` header
- **Rate Limits**: Not explicitly handled in current implementation
- **Data Format**: JSON responses containing opinion metadata and text

**GovInfo API**
- **Purpose**: Access to federal regulations and official documents
- **Endpoint**: `https://api.govinfo.gov`
- **Authentication**: Optional API key for higher limits
- **Coverage**: CFR, Federal Register, Congressional documents

**eCFR API**
- **Purpose**: Current Code of Federal Regulations
- **Endpoint**: `https://www.ecfr.gov/api/search/v1`
- **Authentication**: None required
- **Coverage**: All CFR titles and sections

**Regulations.gov API**
- **Purpose**: Rulemaking documents and public comments
- **Endpoint**: `https://api.regulations.gov/v4`
- **Authentication**: Optional API key via `X-Api-Key` header
- **Coverage**: Federal regulations, proposed rules, comments

### Third-Party Services

**ChromaDB**
- **Purpose**: Vector database for semantic search
- **Deployment**: In-process (no separate server)
- **Storage**: In-memory with optional persistence
- **Configuration**: Anonymous telemetry disabled, reset capability enabled
- **Performance**: Handles thousands of documents efficiently

**Streamlit**
- **Purpose**: Web application framework and hosting
- **Deployment**: Local development server (port 5000)
- **Caching**: Resource-level caching for expensive initializations
- **Features**: Multi-tab interface, interactive widgets, real-time updates

### Python Dependencies

Key libraries:
- `mistralai` (1.9.11): Mistral AI client SDK
- `chromadb` (1.3.4): Vector database
- `streamlit`: Web UI framework
- `requests`: HTTP client for API calls
- `pandas`: Data manipulation (data ingestion)
- `python-dotenv`: Environment variable management
- `numpy`: Numerical computations for metrics

### Environment Configuration

Required environment variables:
- `MISTRAL_API_KEY`: Authentication for Mistral AI services (required for embeddings, generation, and evaluation)

Optional environment variables:
- CourtListener API token (can be passed programmatically or via environment)
- GovInfo API key (for higher rate limits)
- Regulations.gov API key (for access to full API)

## Key Features

### 1. Jurisdiction-Aware Retrieval
- Filter documents by federal vs. state jurisdiction
- Automatic jurisdiction detection from queries
- Jurisdiction consistency checking in agentic mode

### 2. Citation-Grounded Answers
- All responses include specific case citations
- Sources numbered and referenced in answers
- Full citation verification in agentic mode

### 3. Multi-Step Agentic Reasoning
- Five-step verification workflow
- Jurisdiction analysis and validation
- Citation accuracy checking
- Consistency verification across sources
- Transparent reasoning steps displayed to user

### 4. AI-Powered Evaluation
- Automated quality scoring using Mistral
- Multi-dimensional assessment (5 metrics)
- Hallucination detection
- Strengths and weaknesses identification
- Actionable feedback generation

### 5. Advanced Metrics Dashboard
- NDCG@K for ranking quality
- Relevance score tracking
- User feedback analytics
- Rating distribution visualization
- Improvement recommendations

### 6. User Feedback Loop
- 5-star rating system
- Optional text comments
- Jurisdiction-based aggregation
- Low-rated question identification
- Export functionality for analysis

### 7. Multi-Source Integration
- CourtListener for case law
- GovInfo for federal documents
- eCFR for current regulations
- Regulations.gov for rulemaking
- Unified document schema across sources

## File Structure

```
LexRAG/
├── app.py                     # Main Streamlit application (6 tabs)
├── data_ingestion.py          # CourtListener API integration
├── data_sources.py            # GovInfo, eCFR, Regulations.gov integration
├── vector_store.py            # ChromaDB vector database with Mistral embeddings
├── rag_pipeline.py            # RAG pipeline with citation grounding
├── agent.py                   # Multi-step agentic reasoning
├── evaluation.py              # AI-as-judge evaluation framework
├── metrics.py                 # Retrieval metrics and analytics
├── feedback.py                # User feedback collection and analysis
├── .env.example               # Environment variables template
├── .streamlit/config.toml     # Streamlit configuration
├── README.md                  # Project documentation
└── replit.md                  # Technical architecture (this file)
```

## Performance Considerations

- **Embedding Speed**: Batch processing reduces API calls
- **Caching**: Streamlit resource caching prevents re-initialization
- **Model Selection**: Small model recommended for MVP (4x faster)
- **Response Time**: Typical query < 5 seconds with Small model
- **Scalability**: ChromaDB handles 10K+ documents efficiently
- **Cost Optimization**: Small model reduces costs by 67% vs Large

## Future Enhancements

Potential improvements for next phase:
- Persistent vector database storage
- Fine-tuned Mistral models on legal corpus
- Multi-document comparative analysis
- Temporal tracking of legal precedent evolution
- Integration with legal citation databases (Casetext, Fastcase)
- Advanced RAG techniques (HyDE, query decomposition)
- Production deployment with authentication
- API endpoint for programmatic access
