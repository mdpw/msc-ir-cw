"""
Brandix ISPS - Run Analysis Page
Single-click analysis execution: Embeddings → Alignment → Metrics → Insights
"""

import streamlit as st
import sys
import json
import numpy as np
import time
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from document_processor import DocumentProcessor
from embedding_engine import EmbeddingEngine
from vector_store import VectorStore
from synchronization_engine import SynchronizationEngine
from llm_engine import LLMEngine
from executive_summary import ExecutiveSummaryGenerator

st.set_page_config(page_title="Run Analysis", page_icon="None", layout="wide")

# Dark Theme Compatible CSS + Font Awesome Icons
st.markdown("""
    <style>
    /* Import Font Awesome */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* Main background */
    .main {
        background-color: #0e1117;
    }
    
    /* Icon styling */
    .fa-icon {
        color: #4da6ff;
        margin-right: 8px;
    }
    
    .fa-icon-large {
        font-size: 1.2em;
        color: #4da6ff;
        margin-right: 10px;
    }
    
    .fa-icon-small {
        font-size: 0.9em;
        color: #4da6ff;
        margin-right: 6px;
    }
    
    /* Info boxes */
    .info-box {
        background-color: rgba(28, 131, 225, 0.08);
        border: 1px solid rgba(28, 131, 225, 0.2);
        border-left: 4px solid #4da6ff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #e0e0e0;
    }
    
    /* Headers */
    h1 {
        color: #4da6ff !important;
        font-weight: 600 !important;
    }
    
    h2, h3, h4 {
        color: #66b3ff !important;
    }
    
    /* Sidebar explicit dark mode */
    [data-testid="stSidebar"] {
        background-color: #0e1117 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-testid="stSidebarNav"] span {
        color: #e0e0e0 !important;
        font-weight: 500 !important;
    }
    
    /* Top Header */
    header[data-testid="stHeader"] {
        background-color: #0e1117 !important;
        background: transparent !important;
    }
    
    /* Dropdown/Selectbox dark mode */
    div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    
    ul[data-testid="stSelectboxVirtualList"] {
        background-color: #1e2129 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: rgba(28, 131, 225, 0.2);
        color: #e0e0e0;
        border: 1px solid rgba(28, 131, 225, 0.3);
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: rgba(28, 131, 225, 0.3);
        border: 1px solid rgba(28, 131, 225, 0.5);
        transform: translateY(-2px);
    }
    
    /* Primary buttons */
    .stButton>button[kind="primary"] {
        height: 60px;
        font-size: 18px;
        font-weight: bold;
        background-color: rgba(28, 131, 225, 0.4);
        border: 1px solid #1c83e1;
    }
    
    /* Metric boxes */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        padding: 15px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    /* Fixed height log viewer */
    div[data-testid="stCodeBlock"] {
        height: 400px !important;
        max-height: 400px !important;
        overflow-y: auto !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    div[data-testid="stCodeBlock"] pre {
        background-color: #1e2129 !important;
        height: 380px !important;
        max-height: 380px !important;
        overflow-y: auto !important;
        margin: 0 !important;
    }
    
    /* Scrollbar styling */
    div[data-testid="stCodeBlock"] pre::-webkit-scrollbar {
        width: 10px;
    }
    div[data-testid="stCodeBlock"] pre::-webkit-scrollbar-track {
        background: #0e1117;
    }
    div[data-testid="stCodeBlock"] pre::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 5px;
    }
    div[data-testid="stCodeBlock"] pre::-webkit-scrollbar-thumb:hover {
        background: #444;
    }
    
    /* Text colors */
    p, span, label, li {
        color: #e0e0e0 !important;
    }
    
    strong {
        color: #ffffff !important;
    }
    
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1><i class="fas fa-cog fa-icon-large"></i>Strategic Alignment Analysis</h1>', unsafe_allow_html=True)
st.markdown("### Single-Click AI Analysis Pipeline")
st.markdown("---")

# Configuration
AVAILABLE_YEARS = ["2026", "2027", "2028", "2029", "2030"]
UPLOAD_BASE = Path("data/uploaded")
OUTPUTS_BASE = Path("outputs")

# Initialize session state
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = "2026"
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = {}
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
if 'run_mode' not in st.session_state:
    st.session_state.run_mode = "core"

# Year Selection
st.markdown('<h3><i class="fas fa-calendar-alt fa-icon"></i>Step 1: Select Year to Analyze</h3>', unsafe_allow_html=True)
col1, col2 = st.columns([1, 3])

with col1:
    selected_year = st.selectbox(
        "Choose Year",
        AVAILABLE_YEARS,
        index=AVAILABLE_YEARS.index(st.session_state.selected_year),
        key='analysis_year_selector'
    )

with col2:
    if selected_year != st.session_state.selected_year:
        st.session_state.selected_year = selected_year
        st.rerun()
    
    st.info(f"Analyzing Year: **{selected_year}**")

# Check if documents exist
year_path = UPLOAD_BASE / selected_year
strategic_path = year_path / "strategic_plan.docx"
action_path = year_path / "action_plan.docx"

strategic_exists = strategic_path.exists()
action_exists = action_path.exists()

st.markdown("---")

# Document Status
st.markdown('<h3><i class="fas fa-folder-open fa-icon"></i>Step 2: Verify Documents</h3>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    if strategic_exists:
        st.success("Strategic Plan Ready")
    else:
        st.error("Strategic Plan Missing")

with col2:
    if action_exists:
        st.success("Action Plan Ready")
    else:
        st.error("Action Plan Missing")

with col3:
    if strategic_exists and action_exists:
        st.success("Ready to Analyze")
    else:
        st.warning("Upload Required")

if not (strategic_exists and action_exists):
    st.error(f"Documents missing for year {selected_year}!")
    st.info("Go to **'Admin Upload'** page to upload documents")
    st.stop()

# Create output directory
output_dir = OUTPUTS_BASE / selected_year
output_dir.mkdir(parents=True, exist_ok=True)

st.markdown("---")

# Analysis Section
st.markdown('<h3><i class="fas fa-play-circle fa-icon"></i>Step 3: Run Analysis</h3>', unsafe_allow_html=True)

# Check existing results
analysis_complete_key = f'analysis_complete_{selected_year}'
already_analyzed = st.session_state.analysis_complete.get(selected_year, False)
sync_report_exists = (output_dir / "synchronization_report.json").exists()

# --- Core Analysis (Stages 1-4) ---
st.markdown("""
<div class="info-box">
<i class="fas fa-info-circle fa-icon-small"></i><strong>Core Analysis (~4 seconds):</strong><br>
1. <i class="fas fa-file-alt fa-icon-small"></i> Load & Process Documents (Extract objectives & actions)<br>
2. <i class="fas fa-microchip fa-icon-small"></i> Generate AI Embeddings (Convert text to vectors)<br>
3. <i class="fas fa-search fa-icon-small"></i> Analyze Alignment (Calculate similarity scores)<br>
4. <i class="fas fa-chart-bar fa-icon-small"></i> Calculate Metrics (KPIs and statistics)
</div>
""", unsafe_allow_html=True)

if already_analyzed or sync_report_exists:
    st.success("Core analysis already completed for this year!")

col1, col2 = st.columns([2, 1])
with col1:
    if st.button("RUN CORE ANALYSIS", type="primary", disabled=st.session_state.analysis_running, use_container_width=True):
        st.session_state.analysis_running = True
        st.session_state.run_mode = "core"
with col2:
    if st.button("RESET RESULTS", type="secondary", disabled=st.session_state.analysis_running, use_container_width=True):
        import shutil
        output_dir = Path(f"outputs/{selected_year}")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        st.session_state.analysis_complete[selected_year] = False
        st.cache_data.clear()
        st.success(f"Results for {selected_year} cleared.")
        time.sleep(1)
        st.rerun()

# --- AI Suggestions (Stages 5-6) ---
st.markdown("---")
st.markdown("""
<div class="info-box">
<i class="fas fa-robot fa-icon-small"></i><strong>AI Suggestions (20-30 minutes, requires Ollama):</strong><br>
5. <i class="fas fa-magic fa-icon-small"></i> Generate Improvements (LLM-powered gap suggestions)<br>
6. <i class="fas fa-clipboard-list fa-icon-small"></i> Create Executive Summary (Professional report)
</div>
""", unsafe_allow_html=True)

if not sync_report_exists and not already_analyzed:
    st.warning("Run core analysis first before generating AI suggestions.")

improvements_exist = (output_dir / "improvements.json").exists()
summary_exists = (output_dir / "executive_summary.json").exists()
if improvements_exist and summary_exists:
    st.success("AI suggestions already generated for this year!")

ai_disabled = st.session_state.analysis_running or (not sync_report_exists and not already_analyzed)
if st.button("GENERATE AI SUGGESTIONS", type="primary", disabled=ai_disabled, use_container_width=True):
    st.session_state.analysis_running = True
    st.session_state.run_mode = "ai"

if st.session_state.analysis_running:

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Create scrollable log viewer with fixed height
    st.markdown("---")
    st.markdown('### <i class="fas fa-terminal fa-icon-small"></i> Analysis Log', unsafe_allow_html=True)
    log_placeholder = st.empty()

    # Initialize log
    log_messages = []

    def add_log(message, icon=""):
        """Add message to scrollable log (no icons)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_messages.append(f"[{timestamp}] {message}")

        # Keep last 50 messages
        if len(log_messages) > 50:
            log_messages.pop(0)

        # Display in fixed-height scrollable container
        log_text = "\n".join(log_messages)
        log_placeholder.code(log_text, language="log")

    metrics_container = st.container()
    run_mode = st.session_state.get('run_mode', 'core')

    try:
        # ==============================================================
        # CORE ANALYSIS MODE (Stages 1-4)
        # ==============================================================
        if run_mode == "core":
            # ============================================================
            # STAGE 1: Load & Process Documents (0-25%)
            # ============================================================
            add_log("="*60)
            add_log("STAGE 1/4: Load & Process Documents")
            add_log("="*60)

            status_text.info("Stage 1/4: Loading & Processing Documents...")
            progress_bar.progress(5)
            time.sleep(0.3)

            add_log("Initializing document processor...")
            processor = DocumentProcessor()

            status_text.info("Loading strategic plan...")
            progress_bar.progress(10)
            add_log(f"Loading strategic plan from: {strategic_path.name}")
            objectives = processor.load_strategic_plan(str(strategic_path))
            add_log(f"Extracted {len(objectives)} strategic objectives")

            status_text.info("Loading action plan...")
            progress_bar.progress(18)
            add_log(f"Loading action plan from: {action_path.name}")
            actions = processor.load_action_plan(str(action_path))
            add_log(f"Extracted {len(actions)} action items")

            progress_bar.progress(25)
            add_log("Stage 1 Complete!")
            status_text.success(f"Stage 1 Complete: {len(objectives)} objectives, {len(actions)} actions")
            time.sleep(0.5)

            # Show metrics
            with metrics_container:
                col1, col2 = st.columns(2)
                col1.metric("Strategic Objectives", len(objectives))
                col2.metric("Action Items", len(actions))

            # ============================================================
            # STAGE 2: Generate Embeddings (25-50%)
            # ============================================================
            add_log("="*60)
            add_log("STAGE 2/4: Generate AI Embeddings")
            add_log("="*60)

            status_text.info("Stage 2/4: Generating AI Embeddings...")
            progress_bar.progress(28)
            time.sleep(0.3)

            add_log("Initializing embedding model (sentence-transformers)...")
            add_log("Model: all-MiniLM-L6-v2 (384 dimensions)")
            embedding_engine = EmbeddingEngine()
            add_log("Embedding model loaded successfully")

            status_text.info("Encoding objectives...")
            progress_bar.progress(35)
            add_log(f"Encoding {len(objectives)} objectives to 384D vectors...")
            embedding_engine.embed_objectives(objectives)
            add_log("Objective embeddings generated")

            status_text.info("Encoding actions...")
            progress_bar.progress(42)
            add_log(f"Encoding {len(actions)} actions to 384D vectors...")
            embedding_engine.embed_actions(actions)
            add_log("Action embeddings generated")

            progress_bar.progress(50)
            add_log("Stage 2 Complete!")
            status_text.success("Stage 2 Complete: Embeddings generated")
            time.sleep(0.5)

            # ============================================================
            # STAGE 3: Analyze Alignment (50-75%)
            # ============================================================
            add_log("="*60)
            add_log("STAGE 3/4: Analyze Strategic Alignment")
            add_log("="*60)

            status_text.info("Stage 3/4: Analyzing Strategic Alignment...")
            progress_bar.progress(52)
            time.sleep(0.3)

            # Create vector store
            status_text.info("Creating vector database...")
            progress_bar.progress(55)
            add_log("Initializing FAISS vector store (L2 distance)...")
            vector_store = VectorStore(dimension=384)
            add_log("Vector store initialized")

            # Prepare action embeddings
            status_text.info("Preparing action embeddings...")
            progress_bar.progress(58)
            add_log("Converting action embeddings to numpy arrays...")
            action_texts = [action['text'] for action in actions]
            action_embeddings = embedding_engine.model.encode(action_texts)
            action_embeddings = np.array(action_embeddings)
            add_log(f"Action embeddings ready: shape {action_embeddings.shape}")

            # Add to vector store
            status_text.info("Building vector index...")
            progress_bar.progress(62)
            add_log(f"Adding {len(actions)} action vectors to FAISS index...")
            action_metadata = [
                {
                    'id': action['id'],
                    'title': action.get('title', action['text'][:50]),
                    'text': action['text'],
                    'pillar': action.get('pillar', 'Unknown')
                }
                for action in actions
            ]
            vector_store.add_vectors(action_embeddings, action_metadata)
            add_log(f"Vector index built with {vector_store.index.ntotal} vectors")

            # Initialize sync engine
            status_text.info("Initializing synchronization engine...")
            progress_bar.progress(65)
            add_log("Creating synchronization engine...")
            sync_engine = SynchronizationEngine(
                doc_processor=processor,
                embedding_engine=embedding_engine,
                vector_store=vector_store
            )
            add_log("Synchronization engine ready")

            # Run synchronization analysis
            status_text.info("Running alignment analysis...")
            progress_bar.progress(68)
            add_log("Calculating similarity matrix...")
            add_log(f"Matrix size: {len(objectives)} objectives × {len(actions)} actions")
            add_log("Computing cosine similarities...")

            results = sync_engine.analyze_synchronization(objectives, actions)

            add_log("Similarity matrix calculated")
            add_log("Alignment scores computed")
            add_log("Gap analysis completed")
            add_log("Pillar statistics generated")

            progress_bar.progress(75)
            add_log("Stage 3 Complete!")
            status_text.success("Stage 3 Complete: Alignment calculated")
            time.sleep(0.5)

            # ============================================================
            # STAGE 4: Calculate Metrics & Save (75-100%)
            # ============================================================
            add_log("="*60)
            add_log("STAGE 4/4: Calculate Metrics & Save Report")
            add_log("="*60)

            status_text.info("Stage 4/4: Calculating Performance Metrics...")
            progress_bar.progress(80)

            # Add metadata to results
            add_log(f"Adding metadata for year {selected_year}...")
            results['year'] = selected_year
            results['analysis_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            add_log(f"Analysis date: {results['analysis_date']}")

            # Save synchronization report
            add_log("Saving synchronization report...")
            results_file = output_dir / "synchronization_report.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            add_log(f"Saved: {results_file}")

            progress_bar.progress(100)
            time.sleep(0.5)

            # Show key metrics
            overall = results['overall_alignment']
            with metrics_container:
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Overall Score", f"{overall['overall_score']:.1f}%")
                col2.metric("Classification", overall['classification'])
                col3.metric("Strong Alignments", overall['distribution']['strong'])
                col4.metric("Coverage Rate", f"{overall['coverage_rate']:.1f}%")

            # SUCCESS
            add_log("="*60)
            add_log("CORE ANALYSIS COMPLETE!")
            add_log("="*60)
            add_log(f"Year: {selected_year}")
            add_log(f"Overall Alignment: {overall['overall_score']:.1f}%")
            add_log(f"Classification: {overall['classification']}")
            add_log(f"Strong Alignments: {overall['distribution']['strong']}")
            add_log(f"Gaps Found: {overall['distribution']['weak']}")
            add_log("Report saved. Ready to view results!")
            add_log("Run 'Generate AI Suggestions' for LLM-powered improvements.")

            status_text.empty()
            progress_bar.empty()

            st.markdown(f'### <i class="fas fa-check-circle fa-icon-small" style="color: #4caf50;"></i> Core Analysis Complete for {selected_year}!', unsafe_allow_html=True)
            st.success("""
            Core analysis completed successfully:
            - Documents processed
            - Embeddings generated
            - Alignment analyzed
            - Metrics calculated
            - Report saved
            """)
            st.info("Click **'Generate AI Suggestions'** above to run LLM-powered improvements and executive summary (requires Ollama).")

            # Final metrics
            st.markdown("---")
            st.markdown('<h3><i class="fas fa-chart-line fa-icon"></i>Results Summary</h3>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Alignment", f"{overall['overall_score']:.1f}%")
            with col2:
                st.metric("Classification", overall['classification'])
            with col3:
                st.metric("Well-Covered",
                         f"{overall['distribution']['strong'] + overall['distribution']['moderate']}")
            with col4:
                st.metric("Gaps Found", overall['distribution']['weak'])

            # Mark as complete and clear cached data so View Results picks up new results
            st.session_state.analysis_complete[selected_year] = True
            st.session_state.analysis_running = False
            st.cache_data.clear()

        # ==============================================================
        # AI SUGGESTIONS MODE (Stages 5-6)
        # ==============================================================
        elif run_mode == "ai":
            # Load saved sync report
            add_log("="*60)
            add_log("Loading saved analysis report...")
            add_log("="*60)

            results_file = output_dir / "synchronization_report.json"
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            add_log(f"Loaded report for year {selected_year}")

            # Rebuild sync engine from saved data
            add_log("Rebuilding analysis engine from saved data...")
            processor = DocumentProcessor()
            objectives = processor.load_strategic_plan(str(strategic_path))
            actions = processor.load_action_plan(str(action_path))
            add_log(f"Loaded {len(objectives)} objectives and {len(actions)} actions")

            embedding_engine = EmbeddingEngine()
            embedding_engine.embed_objectives(objectives)
            embedding_engine.embed_actions(actions)
            add_log("Embeddings regenerated")

            vector_store = VectorStore(dimension=384)
            action_texts = [action['text'] for action in actions]
            action_embeddings = np.array(embedding_engine.model.encode(action_texts))
            action_metadata = [
                {
                    'id': action['id'],
                    'title': action.get('title', action['text'][:50]),
                    'text': action['text'],
                    'pillar': action.get('pillar', 'Unknown')
                }
                for action in actions
            ]
            vector_store.add_vectors(action_embeddings, action_metadata)

            sync_engine = SynchronizationEngine(
                doc_processor=processor,
                embedding_engine=embedding_engine,
                vector_store=vector_store
            )
            sync_engine.analyze_synchronization(objectives, actions)
            add_log("Analysis engine rebuilt successfully")

            progress_bar.progress(15)

            # ============================================================
            # STAGE 5: Generate AI Improvements (15-60%)
            # ============================================================
            add_log("="*60)
            add_log("STAGE 5: Generate AI-Powered Improvements")
            add_log("="*60)

            status_text.info("Stage 5: Generating AI-Powered Improvements...")
            progress_bar.progress(20)
            time.sleep(0.3)

            try:
                from rag_pipeline import ImprovementGenerator

                add_log("Checking Ollama connection...")
                llm = LLMEngine(model_name="phi3:mini")
                if llm.test_connection():
                    add_log("Ollama connection successful")
                    add_log("Model: phi3:mini")

                    status_text.info("Initializing improvement generator...")
                    progress_bar.progress(25)

                    add_log("Creating RAG pipeline...")
                    add_log("Creating document chunks for context retrieval...")
                    improvement_gen = ImprovementGenerator(sync_engine, llm)
                    add_log(f"Created {len(improvement_gen.rag_pipeline.chunks)} document chunks")
                    add_log("Generated embeddings for all chunks")

                    status_text.info("Generating improvement suggestions (this may take 10-15 minutes)...")
                    progress_bar.progress(30)

                    add_log("Identifying gap objectives...")
                    improvements = improvement_gen.generate_improvements_for_gaps(
                        threshold=0.50,
                        max_objectives=10
                    )

                    num_gaps = improvements['summary']['total_gaps']
                    num_processed = improvements['summary']['processed']
                    num_suggestions = improvements['summary']['total_suggestions']

                    add_log(f"Found {num_gaps} gap objectives")
                    add_log(f"Processed top {num_processed} objectives")
                    add_log(f"Generated {num_suggestions} improvement suggestions")

                    severity = improvements['summary']['by_severity']
                    add_log(f"Critical gaps: {severity.get('critical', 0)}")
                    add_log(f"High priority: {severity.get('high', 0)}")
                    add_log(f"Medium priority: {severity.get('medium', 0)}")

                    add_log("Saving improvement suggestions...")
                    improvements_file = output_dir / "improvements.json"
                    with open(improvements_file, 'w', encoding='utf-8') as f:
                        json.dump(improvements, f, indent=2)
                    add_log(f"Saved to: {improvements_file.name}")

                    progress_bar.progress(55)
                    add_log("Stage 5 Complete!")
                    status_text.success(f"Stage 5 Complete: Generated {num_suggestions} suggestions")
                else:
                    add_log("ERROR: Ollama not available")
                    status_text.error("Ollama not available. Please start Ollama and try again.")
                    st.session_state.analysis_running = False
                    st.stop()

            except Exception as e:
                add_log(f"ERROR: {str(e)}")
                status_text.error(f"Could not generate improvements: {str(e)}")
                st.session_state.analysis_running = False
                st.stop()

            time.sleep(0.5)

            # ============================================================
            # STAGE 6: Generate Executive Summary (60-100%)
            # ============================================================
            add_log("="*60)
            add_log("STAGE 6: Create Executive Summary")
            add_log("="*60)

            status_text.info("Stage 6: Creating Executive Summary...")
            progress_bar.progress(60)
            time.sleep(0.3)

            try:
                add_log("Checking LLM availability...")
                llm = LLMEngine(model_name="phi3:mini")
                if llm.test_connection():
                    add_log("LLM connection successful")

                    status_text.info("Generating executive insights with AI...")
                    progress_bar.progress(70)

                    add_log("Initializing executive summary generator...")
                    summary_gen = ExecutiveSummaryGenerator(llm)

                    add_log("Generating summary sections:")
                    add_log("  1. Executive Overview")
                    add_log("  2. Key Findings")
                    add_log("  3. Critical Gaps")
                    add_log("  4. Strategic Recommendations")
                    add_log("  5. Risk Assessment")
                    add_log("  6. Next Steps")

                    summary = summary_gen.generate_executive_summary(results)

                    add_log("All 6 sections generated successfully")

                    add_log("Saving executive summary...")
                    summary_file = output_dir / "executive_summary.json"
                    summary_gen.save_summary(summary, str(summary_file))
                    add_log(f"Saved JSON: {summary_file.name}")

                    add_log("Creating markdown report...")
                    md_content = f"""# Executive Summary - {selected_year}

    ## Overview
    {summary['overview']}

    ## Key Findings
    {summary['key_findings']}

    ## Critical Gaps
    {summary['critical_gaps']}

    ## Strategic Recommendations
    {summary['recommendations']}

    ## Risk Assessment
    {summary['risk_assessment']}

    ## Next Steps
    {summary['next_steps']}
    """
                    md_file = output_dir / "executive_summary.md"
                    with open(md_file, 'w', encoding='utf-8') as f:
                        f.write(md_content)
                    add_log(f"Saved Markdown: {md_file.name}")

                    progress_bar.progress(98)
                    add_log("Stage 6 Complete!")
                    status_text.success("Stage 6 Complete: Executive summary generated")
                else:
                    add_log("WARNING: Ollama not available - skipping executive summary")
                    status_text.warning("Ollama not available - skipping executive summary")
                    progress_bar.progress(98)

            except Exception as e:
                add_log(f"WARNING: Error in summary generation: {str(e)}")
                status_text.warning(f"Could not generate executive summary: {str(e)}")
                progress_bar.progress(98)

            progress_bar.progress(100)
            time.sleep(0.5)

            # SUCCESS
            add_log("="*60)
            add_log("AI SUGGESTIONS COMPLETE!")
            add_log("="*60)
            add_log("Improvements and executive summary generated!")

            status_text.empty()
            progress_bar.empty()

            st.markdown(f'### <i class="fas fa-check-circle fa-icon-small" style="color: #4caf50;"></i> AI Suggestions Complete for {selected_year}!', unsafe_allow_html=True)
            st.success("""
            AI generation completed successfully:
            - Improvement suggestions generated
            - Executive summary created
            """)

            st.session_state.analysis_running = False
            st.cache_data.clear()

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error during analysis: {str(e)}")
        st.exception(e)
        st.session_state.analysis_running = False
        st.stop()

# Navigation after completion
if st.session_state.analysis_complete.get(selected_year):
    st.markdown("---")
    st.markdown('<h3><i class="fas fa-directions fa-icon"></i>Next Steps</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("View Detailed Results", use_container_width=True):
            st.switch_page("pages/03_View_Results.py")
    
    with col2:
        if st.button("Read Executive Summary", use_container_width=True):
            st.switch_page("pages/04_Executive_Summary.py")
    
    with col3:
        if st.button("Compare Years", use_container_width=True):
            st.switch_page("pages/05_Multi_Year_Comparison.py")
    
    # Download section
    st.markdown("---")
    st.markdown('<h3><i class="fas fa-download fa-icon"></i>Download Reports</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        results_file = output_dir / "synchronization_report.json"
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                st.download_button(
                    label="Download Analysis Results (JSON)",
                    data=f.read(),
                    file_name=f"synchronization_report_{selected_year}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    with col2:
        md_file = output_dir / "executive_summary.md"
        if md_file.exists():
            with open(md_file, 'r', encoding='utf-8') as f:
                st.download_button(
                    label="Download Executive Summary (MD)",
                    data=f.read(),
                    file_name=f"executive_summary_{selected_year}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

st.markdown("---")
st.markdown("""
<div class="info-box">
<i class="fas fa-lightbulb fa-icon-small"></i><strong>Tip:</strong> Analysis results are automatically saved and can be viewed anytime from the View Results page
</div>
""", unsafe_allow_html=True)
