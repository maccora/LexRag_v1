# LexRAG: Jurisdiction-Aware Legal Q&A

A Retrieval-Augmented Generation (RAG) system tailored for legal research, powered by Mistral AI. LexRAG helps lawyers, students, and analysts efficiently locate and interpret relevant cases, statutes, and regulations with accurate, jurisdiction-specific, and properly cited responses.

## Features

- **🎯 Jurisdiction-Aware Retrieval**: Filter legal documents by federal vs. state jurisdiction
- **📚 Citation-Grounded Answers**: Responses include specific case citations and legal references
- **⚡ Powered by Mistral AI**: Uses Mistral embeddings and large language models
- **🔍 Case Law Search**: Integration with CourtListener API for real legal data
- **📊 Interactive Interface**: Clean Streamlit UI for legal research
- **🗄️ Vector Database**: ChromaDB for efficient semantic search

## Architecture

### Core Components

1. **Data Ingestion** (`data_ingestion.py`)
   - CourtListener API integration
   - Normalized JSONL schema for legal documents
   - Metadata extraction (court, jurisdiction, citation, date)
   - Sample legal corpus for demonstration

2. **Vector Store** (`vector_store.py`)
   - ChromaDB for document storage and retrieval
   - Mistral embeddings for semantic search
   - Jurisdiction-based filtering
   - Collection statistics and management

3. **RAG Pipeline** (`rag_pipeline.py`)
   - Context retrieval with relevance ranking
   - Citation-grounded answer generation
   - Mistral-powered legal analysis
   - Source formatting and display

4. **Streamlit UI** (`app.py`)
   - Interactive question interface
   - Jurisdiction filter controls
   - Source browsing and exploration
   - Data management tools

## Getting Started

### Prerequisites

- Python 3.11+
- Mistral API key (get one at [console.mistral.ai](https://console.mistral.ai))

### Installation

Dependencies are already configured in this Replit environment:
- `mistralai` - Mistral AI SDK
- `chromadb` - Vector database
- `streamlit` - Web interface
- `requests` - API integration
- `pandas` - Data processing
- `python-dotenv` - Environment management

### Configuration

1. **Set your Mistral API key**:
   - Enter it directly in the sidebar of the app, or
   - Create a `.env` file with:
     ```
     MISTRAL_API_KEY=your_mistral_api_key_here
     ```

2. **Initialize the system**:
   - On first run, sample legal documents are automatically loaded
   - Access the "Manage Data" tab to load more documents

## Usage

### Asking Questions

1. Select jurisdiction filter (all, federal, or state)
2. Choose number of sources to retrieve (1-10)
3. Enter your legal question
4. Review the citation-grounded answer
5. Explore supporting case law sources

### Example Questions

- "What are the requirements for trade secret protection?"
- "How do courts interpret employment contracts?"
- "What are Miranda rights and when do they apply?"
- "What are tenant rights regarding habitability?"
- "How does the Fourth Amendment apply to digital data?"

### Managing Data

**Load Sample Data**: Pre-configured legal cases for demonstration

**Fetch from CourtListener**: Retrieve real legal opinions from the API
- Enter a search query
- Specify maximum number of results
- Documents are automatically added to the vector database

**Reset Database**: Clear all documents and start fresh

## System Architecture

```
User Question
     ↓
Jurisdiction Filter → Vector Search (ChromaDB)
     ↓
Retrieve Top-K Cases (with metadata)
     ↓
Format Context with Citations
     ↓
Mistral LLM Generation
     ↓
Citation-Grounded Answer + Sources
```

## Data Schema

Each legal document contains:
- **id**: Unique identifier
- **case_name**: Full case name (e.g., "Smith v. Jones")
- **citation**: Legal citation (e.g., "123 F.3d 456")
- **court**: Court identifier
- **jurisdiction**: "federal" or "state"
- **date_filed**: Filing date
- **text**: Full case text or summary
- **document_type**: "case_law", "statute", "regulation"
- **url**: Link to full opinion

## Models Used

- **Embeddings**: `mistral-embed` - Semantic representation of legal text
- **Generation**: `mistral-large-latest` (default) - High-quality legal analysis
- **Alternatives**: `mistral-medium-latest`, `mistral-small-latest`

## Future Enhancements

- **Multi-step Agentic Reasoning**: Chain verification and citation cross-checking
- **Additional Data Sources**: GovInfo, eCFR, Regulations.gov integration
- **AI-as-Judge Evaluation**: Automated scoring of answer quality
- **Retrieval Metrics**: Recall@K and Mean Reciprocal Rank tracking
- **Comparative Analysis**: Side-by-side jurisdiction comparison
- **Regulatory Updates**: Real-time tracking of legal changes

## Project Structure

```
LexRAG/
├── app.py                  # Main Streamlit application
├── data_ingestion.py       # CourtListener API integration
├── vector_store.py         # ChromaDB vector database
├── rag_pipeline.py         # Mistral-powered RAG pipeline
├── .env.example            # Environment variables template
├── .streamlit/
│   └── config.toml         # Streamlit configuration
└── README.md               # This file
```

## Technical Considerations

- **Embeddings**: Mistral's embedding model provides legal domain understanding
- **Retrieval**: Semantic search with jurisdiction metadata filtering
- **Generation**: Temperature set to 0.3 for factual consistency
- **Citations**: Explicit instruction to ground answers in provided sources
- **Scalability**: ChromaDB allows efficient scaling to larger corpora

## License & Legal Notice

This is an educational and research tool. Legal information provided by LexRAG should not be considered legal advice. Always consult with qualified legal professionals for specific legal matters.

## Citation

If you use LexRAG in your research, please cite:

```
LexRAG: Jurisdiction-Aware, Citation-Grounded Legal Q&A
A Retrieval-Augmented Generation system for legal research
Powered by Mistral AI and CourtListener
```

---

Built with ⚖️ for legal research professionals
