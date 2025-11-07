import streamlit as st
import os
from dotenv import load_dotenv
import time
from data_ingestion import CourtListenerIngestion, create_sample_legal_data
from data_sources import GovInfoIngestion, ECFRIngestion, RegulationsGovIngestion, create_sample_regulatory_data
from vector_store import LegalVectorStore
from rag_pipeline import LegalRAGPipeline
from evaluation import AIJudgeEvaluator
from metrics import RetrievalMetrics
from agent import LegalResearchAgent
from feedback import UserFeedbackSystem, FeedbackAnalyzer

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
    """Initialize the RAG system with vector store and sample data."""
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    
    vector_store = LegalVectorStore(
        collection_name="legal_documents",
        mistral_api_key=mistral_api_key
    )
    
    stats = vector_store.get_collection_stats()
    if stats["total_documents"] == 0:
        st.info("🔄 Initializing legal corpus with sample data...")
        sample_docs = create_sample_legal_data()
        sample_regs = create_sample_regulatory_data()
        vector_store.add_documents(sample_docs + sample_regs)
        st.success(f"✅ Loaded {len(sample_docs + sample_regs)} sample documents")
    
    rag_pipeline = LegalRAGPipeline(
        vector_store=vector_store,
        mistral_api_key=mistral_api_key
    )
    
    evaluator = AIJudgeEvaluator(mistral_api_key=mistral_api_key)
    agent = LegalResearchAgent(rag_pipeline, mistral_api_key=mistral_api_key)
    feedback_system = UserFeedbackSystem()
    
    return vector_store, rag_pipeline, evaluator, agent, feedback_system


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
        options=["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"],
        index=0,
        help="Choose the Mistral model for answer generation (Small recommended for MVP: 4x faster, 1/3 cost, 85-90% accuracy)"
    )
    
    use_agentic_mode = st.checkbox(
        "🤖 Agentic Reasoning Mode",
        value=False,
        help="Enable multi-step verification with jurisdiction analysis and citation checking"
    )
    
    st.divider()
    
    st.header("📊 System Status")
    
    if mistral_key:
        try:
            vector_store, rag_pipeline, evaluator, agent, feedback_system = initialize_system()
            stats = vector_store.get_collection_stats()
            
            st.metric("Total Documents", stats["total_documents"])
            
            if "by_jurisdiction" in stats:
                st.write("**By Jurisdiction:**")
                for jur, count in stats["by_jurisdiction"].items():
                    st.write(f"- {jur.upper()}: {count}")
            
            feedback_stats = feedback_system.get_statistics()
            if feedback_stats["total_feedback"] > 0:
                st.metric("Avg User Rating", f"{feedback_stats['avg_rating']:.1f} ⭐")
            
            st.success("✅ System Ready")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            vector_store, rag_pipeline, evaluator, agent, feedback_system = None, None, None, None, None
    else:
        st.warning("⚠️ Mistral API key required")
        vector_store, rag_pipeline, evaluator, agent, feedback_system = None, None, None, None, None
    
    st.divider()
    
    with st.expander("ℹ️ About LexRAG"):
        st.markdown("""
        **LexRAG** is an advanced RAG system for legal research.
        
        **Core Features:**
        - 🎯 Jurisdiction-aware retrieval
        - 📚 Citation-grounded answers
        - ⚡ Powered by Mistral AI
        - 🤖 Multi-step agentic reasoning
        - 📊 AI-powered evaluation
        - 💬 User feedback collection
        
        **Data Sources:**
        - CourtListener (case law)
        - GovInfo (federal documents)
        - eCFR (federal regulations)
        - Regulations.gov (rulemaking)
        """)


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Ask Question", 
    "🤖 Agentic Research", 
    "📊 Evaluation & Metrics",
    "📚 Browse Sources", 
    "💬 Feedback",
    "🔧 Manage Data"
])

with tab1:
    st.header("Ask a Legal Question")
    
    example_questions = [
        "What are the requirements for trade secret protection?",
        "How do courts interpret employment contracts?",
        "What are Miranda rights and when do they apply?",
        "What are tenant rights regarding habitability?",
        "How does the Fourth Amendment apply to digital data?",
        "What are COPPA requirements for children's online privacy?"
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
    
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        ask_button = st.button("🔍 Search & Answer", type="primary", use_container_width=True)
    with col2:
        evaluate_button = st.button("📊 Answer + Evaluate", use_container_width=True)
    
    if (ask_button or evaluate_button) and user_question and rag_pipeline:
        start_time = time.time()
        
        with st.spinner("Retrieving relevant legal sources..."):
            result = rag_pipeline.answer_question(
                query=user_question,
                jurisdiction=jurisdiction_filter if jurisdiction_filter != "all" else None,
                n_results=num_sources,
                model=model_choice
            )
        
        response_time = time.time() - start_time
        
        st.session_state['last_result'] = result
        
        st.success("✅ Analysis Complete")
        
        st.subheader("📝 Answer")
        st.markdown(result["answer"])
        
        st.divider()
        
        st.subheader(f"📚 Supporting Sources ({result['num_sources']})")
        st.caption(f"Jurisdiction filter: {result['jurisdiction_filter'].upper()} | Response time: {response_time:.2f}s")
        
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
        
        if evaluate_button and evaluator:
            st.divider()
            st.subheader("🎯 AI Judge Evaluation")
            
            with st.spinner("Evaluating answer quality..."):
                evaluation = evaluator.evaluate_answer(
                    question=user_question,
                    answer=result["answer"],
                    sources=result["sources"],
                    model=model_choice
                )
            
            if "error" not in evaluation:
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Factual Accuracy", f"{evaluation['factual_accuracy']}/10")
                col2.metric("Citation Validity", f"{evaluation['citation_validity']}/10")
                col3.metric("Jurisdiction Align", f"{evaluation['jurisdictional_alignment']}/10")
                col4.metric("Completeness", f"{evaluation['completeness']}/10")
                col5.metric("Clarity", f"{evaluation['clarity']}/10")
                
                st.metric("**Overall Score**", f"{evaluation['overall_score']:.1f}/10")
                
                if evaluation.get('hallucination_detected'):
                    st.warning("⚠️ Potential hallucination detected")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Strengths:**")
                    for strength in evaluation.get('strengths', []):
                        st.write(f"✅ {strength}")
                with col2:
                    st.write("**Weaknesses:**")
                    for weakness in evaluation.get('weaknesses', []):
                        st.write(f"⚠️ {weakness}")
                
                st.info(f"**Feedback:** {evaluation.get('feedback', 'No feedback provided')}")
            else:
                st.error(evaluation["error"])
        
        st.divider()
        st.subheader("📝 Rate This Answer")
        
        rating_col1, rating_col2 = st.columns([1, 3])
        with rating_col1:
            user_rating = st.select_slider(
                "Your Rating",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: "⭐" * x
            )
        with rating_col2:
            user_comments = st.text_input("Comments (optional)", placeholder="What could be improved?")
        
        if st.button("Submit Feedback"):
            if feedback_system:
                feedback_system.submit_feedback(
                    question=user_question,
                    answer=result["answer"],
                    rating=user_rating,
                    comments=user_comments,
                    sources=result["sources"],
                    jurisdiction=jurisdiction_filter
                )
                st.success("✅ Thank you for your feedback!")
    
    elif (ask_button or evaluate_button) and not user_question:
        st.warning("⚠️ Please enter a question")
    elif (ask_button or evaluate_button) and not rag_pipeline:
        st.error("❌ Please configure your Mistral API key in the sidebar")

with tab2:
    st.header("🤖 Agentic Legal Research")
    st.markdown("Multi-step reasoning with jurisdiction verification and citation cross-checking")
    
    agentic_question = st.text_area(
        "Legal Research Question:",
        height=100,
        placeholder="Enter a complex legal question requiring multi-jurisdictional analysis..."
    )
    
    run_agentic = st.button("🔬 Run Agentic Research", type="primary", key="agentic_research_btn")
    
    if run_agentic:
        if not agentic_question:
            st.warning("⚠️ Please enter a research question")
        elif not agent:
            st.error("❌ Please configure your Mistral API key")
        else:
            with st.spinner("Executing multi-step research workflow..."):
                agentic_result = agent.multi_step_research(agentic_question, model=model_choice)
            
            if "error" not in agentic_result:
                st.success("✅ Agentic Research Complete")
                
                st.subheader("🧠 Research Steps")
                for step in agentic_result["steps"]:
                    with st.expander(f"Step {step['step']}: {step['action']}", expanded=True):
                        st.write(f"**Result:** {step['result']}")
                        if "reasoning" in step:
                            st.info(f"**Reasoning:** {step['reasoning']}")
                        if "warnings" in step and step["warnings"]:
                            for warning in step["warnings"]:
                                st.warning(f"⚠️ {warning}")
                
                st.divider()
                st.subheader("📝 Final Answer")
                st.markdown(agentic_result["answer"])
                
                st.divider()
                st.subheader("🔍 Verification Details")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Citation Verification:**")
                    cit_ver = agentic_result.get("citation_verification", {})
                    st.metric("Verified Citations", cit_ver.get("total_verified", 0))
                    if cit_ver.get("issues"):
                        for issue in cit_ver["issues"]:
                            st.warning(f"⚠️ {issue}")
                
                with col2:
                    st.write("**Consistency Check:**")
                    consistency = agentic_result.get("consistency_check", {})
                    status = "✅ Consistent" if consistency.get("consistent") else "⚠️ Issues Found"
                    st.write(f"**Status:** {status}")
                    st.write(f"**Distribution:** {consistency.get('jurisdiction_distribution', {})}")
            else:
                st.error(agentic_result["error"])

with tab3:
    st.header("📊 Evaluation & Metrics Dashboard")
    
    metrics_tab1, metrics_tab2 = st.tabs(["Retrieval Metrics", "User Feedback Analytics"])
    
    with metrics_tab1:
        st.subheader("Retrieval Performance Metrics")
        
        if st.session_state.get('last_result'):
            result = st.session_state['last_result']
            
            metrics = RetrievalMetrics.calculate_all_metrics(
                retrieved_docs=result.get("sources", []),
                k_values=[1, 3, 5]
            )
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Retrieved", metrics.get("total_retrieved", 0))
            col2.metric("Avg Relevance", f"{metrics.get('avg_relevance_score', 0):.2%}")
            col3.metric("Min Distance", f"{metrics.get('min_distance', 0):.3f}")
            col4.metric("Max Distance", f"{metrics.get('max_distance', 0):.3f}")
            
            st.subheader("NDCG Scores (Normalized Discounted Cumulative Gain)")
            ndcg_cols = st.columns(3)
            for i, k in enumerate([1, 3, 5]):
                ndcg_cols[i].metric(f"NDCG@{k}", f"{metrics.get(f'ndcg@{k}', 0):.3f}")
            
            st.info("💡 NDCG measures ranking quality, considering position and relevance. Higher is better (max 1.0)")
        else:
            st.info("📊 Run a query in the 'Ask Question' tab to see metrics")
    
    with metrics_tab2:
        st.subheader("User Feedback Analytics")
        
        if feedback_system:
            stats = feedback_system.get_statistics()
            
            if stats["total_feedback"] > 0:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Feedback", stats["total_feedback"])
                col2.metric("Avg Rating", f"{stats['avg_rating']:.2f} ⭐")
                col3.metric("Positive (%)", f"{stats['positive_feedback']/stats['total_feedback']*100:.0f}%")
                col4.metric("Negative (%)", f"{stats['negative_feedback']/stats['total_feedback']*100:.0f}%")
                
                st.subheader("Rating Distribution")
                rating_dist = stats["rating_distribution"]
                for rating in range(5, 0, -1):
                    count = rating_dist.get(rating, 0)
                    percentage = (count / stats["total_feedback"] * 100) if stats["total_feedback"] > 0 else 0
                    st.write(f"{'⭐' * rating}: {'█' * int(percentage/2)} {count} ({percentage:.1f}%)")
                
                st.subheader("Recommendations")
                recommendations = FeedbackAnalyzer.generate_recommendations(stats)
                for rec in recommendations:
                    st.info(f"💡 {rec}")
                
                if st.button("📥 Export Feedback Data"):
                    export_file = feedback_system.export_for_analysis()
                    st.success(f"✅ Exported to {export_file}")
            else:
                st.info("📭 No user feedback collected yet. Submit ratings in the 'Ask Question' tab!")

with tab4:
    st.header("📚 Browse Legal Corpus")
    
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
                        st.write(f"**Type:** {metadata.get('document_type', 'unknown')}")
                        st.write(f"**Text:** {doc_text[:300]}...")
            except Exception as e:
                st.error(f"Error loading documents: {e}")
        else:
            st.info("📭 No documents in the database yet. Use the Manage Data tab to load data.")
    else:
        st.warning("⚠️ Please configure your Mistral API key")

with tab5:
    st.header("💬 User Feedback Overview")
    
    if feedback_system:
        stats = feedback_system.get_statistics()
        
        if stats["total_feedback"] > 0:
            st.subheader("Recent Feedback")
            
            all_feedback = feedback_system.load_all_feedback()
            recent = all_feedback[-10:][::-1]
            
            for fb in recent:
                with st.expander(f"{'⭐' * fb['rating']} - {fb['question'][:100]}..."):
                    st.write(f"**Rating:** {'⭐' * fb['rating']}")
                    st.write(f"**Question:** {fb['question']}")
                    if fb.get('comments'):
                        st.write(f"**Comments:** {fb['comments']}")
                    st.write(f"**Jurisdiction:** {fb.get('jurisdiction', 'N/A')}")
                    st.write(f"**Date:** {fb.get('timestamp', 'N/A')}")
            
            st.divider()
            st.subheader("Low-Rated Questions (Need Improvement)")
            low_rated = feedback_system.get_low_rated_questions(threshold=2)
            
            if low_rated:
                for fb in low_rated[:5]:
                    st.warning(f"⚠️ {fb['rating']}⭐: {fb['question']}")
                    if fb.get('comments'):
                        st.write(f"   💬 {fb['comments']}")
            else:
                st.success("✅ No low-rated feedback - great job!")
        else:
            st.info("📭 No feedback collected yet")
    else:
        st.warning("⚠️ Feedback system not initialized")

with tab6:
    st.header("🔧 Manage Legal Corpus")
    
    if vector_store:
        data_tab1, data_tab2 = st.tabs(["Sample Data", "External APIs"])
        
        with data_tab1:
            st.subheader("Load Sample Data")
            st.write("Load pre-configured sample legal documents for demonstration.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Load Sample Case Law"):
                    with st.spinner("Loading sample data..."):
                        sample_docs = create_sample_legal_data()
                        vector_store.add_documents(sample_docs)
                        st.cache_resource.clear()
                    st.success(f"✅ Loaded {len(sample_docs)} case law documents")
                    st.rerun()
            
            with col2:
                if st.button("📥 Load Sample Regulations"):
                    with st.spinner("Loading regulatory data..."):
                        sample_regs = create_sample_regulatory_data()
                        vector_store.add_documents(sample_regs)
                        st.cache_resource.clear()
                    st.success(f"✅ Loaded {len(sample_regs)} regulatory documents")
                    st.rerun()
        
        with data_tab2:
            st.subheader("Fetch from External APIs")
            
            api_source = st.selectbox(
                "Select Data Source",
                ["CourtListener", "eCFR", "GovInfo", "Regulations.gov"]
            )
            
            search_query = st.text_input("Search Query", value="contract law")
            max_results = st.number_input("Max Results", min_value=1, max_value=50, value=10)
            
            if st.button(f"🌐 Fetch from {api_source}"):
                with st.spinner(f"Fetching from {api_source}..."):
                    docs = []
                    
                    if api_source == "CourtListener":
                        ingestion = CourtListenerIngestion()
                        docs = ingestion.search_opinions(search_query, max_results=max_results)
                    elif api_source == "eCFR":
                        ingestion = ECFRIngestion()
                        docs = ingestion.search_cfr(search_query, max_results=max_results)
                    elif api_source == "GovInfo":
                        ingestion = GovInfoIngestion()
                        docs = ingestion.search_regulations(search_query, max_results=max_results)
                    elif api_source == "Regulations.gov":
                        ingestion = RegulationsGovIngestion()
                        docs = ingestion.search_documents(search_query, max_results=max_results)
                    
                    if docs:
                        vector_store.add_documents(docs)
                        st.success(f"✅ Added {len(docs)} documents from {api_source}")
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
st.caption("LexRAG - Advanced Legal Research System | Powered by Mistral AI | Features: Agentic Reasoning • AI Evaluation • Multi-Source Integration")
