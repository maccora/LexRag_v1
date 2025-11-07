import streamlit as st
import os
from dotenv import load_dotenv
from data_ingestion import CourtListenerIngestion, create_sample_legal_data
from vector_store import LegalVectorStore
from rag_pipeline import LegalRAGPipeline

load_dotenv()

st.set_page_config(
    page_title="LexRAG - Legal Research Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ LexRAG: Jurisdiction-Aware Legal Q&A")
st.markdown("*Citation-grounded legal research powered by Mistral AI*")


@st.cache_resource
def initialize_system():
    """
    Initialize the RAG system with vector store and sample data.
    """
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    
    vector_store = LegalVectorStore(
        collection_name="legal_documents",
        mistral_api_key=mistral_api_key
    )
    
    stats = vector_store.get_collection_stats()
    if stats["total_documents"] == 0:
        st.info("🔄 Initializing legal corpus with sample data...")
        sample_docs = create_sample_legal_data()
        vector_store.add_documents(sample_docs)
        st.success(f"✅ Loaded {len(sample_docs)} sample legal documents")
    
    rag_pipeline = LegalRAGPipeline(
        vector_store=vector_store,
        mistral_api_key=mistral_api_key
    )
    
    return vector_store, rag_pipeline


with st.sidebar:
    st.header("⚙️ Configuration")
    
    mistral_key = st.text_input(
        "Mistral API Key",
        type="password",
        value=os.getenv("MISTRAL_API_KEY", ""),
        help="Enter your Mistral API key for embeddings and generation"
    )
    
    if mistral_key and mistral_key != os.getenv("MISTRAL_API_KEY"):
        os.environ["MISTRAL_API_KEY"] = mistral_key
        st.cache_resource.clear()
        st.rerun()
    
    st.divider()
    
    st.header("🔍 Search Settings")
    
    jurisdiction_filter = st.selectbox(
        "Jurisdiction",
        options=["all", "federal", "state"],
        index=0,
        help="Filter results by jurisdiction"
    )
    
    num_sources = st.slider(
        "Number of Sources",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of legal sources to retrieve"
    )
    
    model_choice = st.selectbox(
        "Mistral Model",
        options=["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
        index=0,
        help="Choose the Mistral model for answer generation"
    )
    
    st.divider()
    
    st.header("📊 System Status")
    
    if mistral_key:
        try:
            vector_store, rag_pipeline = initialize_system()
            stats = vector_store.get_collection_stats()
            
            st.metric("Total Documents", stats["total_documents"])
            
            if "by_jurisdiction" in stats:
                st.write("**By Jurisdiction:**")
                for jur, count in stats["by_jurisdiction"].items():
                    st.write(f"- {jur.upper()}: {count}")
            
            st.success("✅ System Ready")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            vector_store, rag_pipeline = None, None
    else:
        st.warning("⚠️ Mistral API key required")
        vector_store, rag_pipeline = None, None
    
    st.divider()
    
    with st.expander("ℹ️ About LexRAG"):
        st.markdown("""
        **LexRAG** is a Retrieval-Augmented Generation system for legal research.
        
        **Features:**
        - 🎯 Jurisdiction-aware retrieval
        - 📚 Citation-grounded answers
        - ⚡ Powered by Mistral AI
        - 🔍 Case law search
        
        **How to use:**
        1. Enter your Mistral API key
        2. Select jurisdiction filter
        3. Ask your legal question
        4. Review cited sources
        """)


tab1, tab2, tab3 = st.tabs(["💬 Ask Question", "📚 Browse Sources", "🔧 Manage Data"])

with tab1:
    st.header("Ask a Legal Question")
    
    example_questions = [
        "What are the requirements for trade secret protection?",
        "How do courts interpret employment contracts?",
        "What are Miranda rights and when do they apply?",
        "What are tenant rights regarding habitability?",
        "How does the Fourth Amendment apply to digital data?"
    ]
    
    selected_example = st.selectbox(
        "Try an example question:",
        [""] + example_questions,
        index=0
    )
    
    user_question = st.text_area(
        "Your Question:",
        value=selected_example if selected_example else "",
        height=100,
        placeholder="e.g., What are the legal requirements for employment termination?"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🔍 Search & Answer", type="primary", use_container_width=True)
    
    if ask_button and user_question and rag_pipeline:
        with st.spinner("Retrieving relevant legal sources..."):
            result = rag_pipeline.answer_question(
                query=user_question,
                jurisdiction=jurisdiction_filter if jurisdiction_filter != "all" else None,
                n_results=num_sources,
                model=model_choice
            )
        
        st.success("✅ Analysis Complete")
        
        st.subheader("📝 Answer")
        st.markdown(result["answer"])
        
        st.divider()
        
        st.subheader(f"📚 Supporting Sources ({result['num_sources']})")
        st.caption(f"Jurisdiction filter: {result['jurisdiction_filter'].upper()}")
        
        formatted_sources = rag_pipeline.format_sources_for_display(result["sources"])
        
        for source in formatted_sources:
            with st.expander(
                f"[{source['number']}] {source['case_name']} - {source['citation']} "
                f"({source['jurisdiction']}) - Relevance: {source['relevance']}"
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Court:** {source['court']}")
                with col2:
                    st.write(f"**Date:** {source['date']}")
                with col3:
                    st.write(f"**Jurisdiction:** {source['jurisdiction']}")
                
                st.markdown("**Summary:**")
                st.write(source['text'])
                
                if source['url'] and source['url'] != "":
                    st.markdown(f"[View Full Opinion]({source['url']})")
    
    elif ask_button and not user_question:
        st.warning("⚠️ Please enter a question")
    elif ask_button and not rag_pipeline:
        st.error("❌ Please configure your Mistral API key in the sidebar")

with tab2:
    st.header("Browse Legal Corpus")
    
    if vector_store:
        stats = vector_store.get_collection_stats()
        
        st.metric("Total Documents in Database", stats["total_documents"])
        
        if stats["total_documents"] > 0:
            st.subheader("Document Distribution by Jurisdiction")
            
            if "by_jurisdiction" in stats:
                jur_data = stats["by_jurisdiction"]
                cols = st.columns(len(jur_data))
                for i, (jur, count) in enumerate(jur_data.items()):
                    with cols[i]:
                        st.metric(jur.upper(), count)
            
            st.divider()
            
            st.subheader("Sample Documents")
            try:
                all_docs = vector_store.collection.get(limit=10)
                
                for i in range(min(5, len(all_docs["documents"]))):
                    metadata = all_docs["metadatas"][i] if all_docs["metadatas"] else {}
                    doc_text = all_docs["documents"][i] if all_docs["documents"] else ""
                    
                    with st.expander(
                        f"{metadata.get('case_name', 'Unknown')} - "
                        f"{metadata.get('citation', 'N/A')} ({metadata.get('jurisdiction', 'unknown').upper()})"
                    ):
                        st.write(f"**Court:** {metadata.get('court', 'unknown')}")
                        st.write(f"**Date Filed:** {metadata.get('date_filed', 'N/A')}")
                        st.write(f"**Jurisdiction:** {metadata.get('jurisdiction', 'unknown')}")
                        st.write(f"**Text:** {doc_text[:300]}...")
            except Exception as e:
                st.error(f"Error loading documents: {e}")
        else:
            st.info("📭 No documents in the database yet. Use the Manage Data tab to load data.")
    else:
        st.warning("⚠️ Please configure your Mistral API key")

with tab3:
    st.header("Manage Legal Corpus")
    
    if vector_store:
        st.subheader("Load Sample Data")
        st.write("Load pre-configured sample legal documents for demonstration.")
        
        if st.button("📥 Load Sample Legal Documents"):
            with st.spinner("Loading sample data..."):
                sample_docs = create_sample_legal_data()
                vector_store.reset_collection()
                vector_store.add_documents(sample_docs)
                st.cache_resource.clear()
            st.success(f"✅ Loaded {len(sample_docs)} sample documents")
            st.rerun()
        
        st.divider()
        
        st.subheader("Fetch from CourtListener")
        st.write("Retrieve legal opinions from the CourtListener API.")
        
        col1, col2 = st.columns(2)
        with col1:
            search_query = st.text_input("Search Query", value="contract law")
        with col2:
            max_results = st.number_input("Max Results", min_value=1, max_value=50, value=10)
        
        if st.button("🌐 Fetch from CourtListener API"):
            with st.spinner("Fetching from CourtListener..."):
                ingestion = CourtListenerIngestion()
                docs = ingestion.search_opinions(search_query, max_results=max_results)
                
                if docs:
                    vector_store.add_documents(docs)
                    st.success(f"✅ Added {len(docs)} documents from CourtListener")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.warning("⚠️ No documents found. The API might require authentication for better results.")
        
        st.divider()
        
        st.subheader("⚠️ Reset Database")
        st.write("Delete all documents and start fresh.")
        
        if st.button("🗑️ Reset All Data", type="secondary"):
            if st.checkbox("I understand this will delete all documents"):
                vector_store.reset_collection()
                st.cache_resource.clear()
                st.success("✅ Database reset successfully")
                st.rerun()
    else:
        st.warning("⚠️ Please configure your Mistral API key")

st.divider()
st.caption("LexRAG - Jurisdiction-Aware Legal Research | Powered by Mistral AI")
