import streamlit as st
import sys
import json
import numpy as np
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from document_processor import DocumentProcessor
from embedding_engine import EmbeddingEngine
from vector_store import VectorStore
from synchronization_engine import SynchronizationEngine
from executive_summary import ExecutiveSummaryGenerator

st.set_page_config(page_title="Run Analysis", page_icon="⚙️", layout="wide")

st.title("⚙️ Strategic Alignment Analysis")
st.markdown("---")

# Available years
AVAILABLE_YEARS = ["2026", "2027", "2028", "2029", "2030"]
UPLOAD_BASE = Path("data/uploaded")
OUTPUTS_BASE = Path("outputs")

# Initialize session state
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = "2026"

# Year Selection
st.subheader("📅 Select Year to Analyze")
selected_year = st.selectbox(
    "Choose Year",
    AVAILABLE_YEARS,
    index=AVAILABLE_YEARS.index(st.session_state.selected_year),
    key='analysis_year_selector'
)

if selected_year != st.session_state.selected_year:
    st.session_state.selected_year = selected_year
    # Reset analysis state when year changes
    analysis_key = f'analysis_step_{selected_year}'
    if analysis_key in st.session_state:
        st.session_state[analysis_key] = 0
    st.rerun()

# Check if documents exist for selected year
year_path = UPLOAD_BASE / selected_year
strategic_path = year_path / "strategic_plan.docx"
action_path = year_path / "action_plan.docx"

strategic_exists = strategic_path.exists()
action_exists = action_path.exists()

if not (strategic_exists and action_exists):
    st.error(f"⚠️ Documents missing for year {selected_year}!")
    
    col1, col2 = st.columns(2)
    with col1:
        if not strategic_exists:
            st.warning("❌ Strategic Plan not uploaded")
        else:
            st.success("✅ Strategic Plan ready")
    
    with col2:
        if not action_exists:
            st.warning("❌ Action Plan not uploaded")
        else:
            st.success("✅ Action Plan ready")
    
    st.info("👉 Go to **'📤 Admin Upload'** page to upload documents")
    st.stop()

st.success(f"✅ Analyzing Year: **{selected_year}**")

# Create output directory for this year
output_dir = OUTPUTS_BASE / selected_year
output_dir.mkdir(parents=True, exist_ok=True)

st.markdown("---")

# Initialize progress tracking (year-specific)
analysis_key = f'analysis_step_{selected_year}'
if analysis_key not in st.session_state:
    st.session_state[analysis_key] = 0

# Progress steps
progress_steps = [
    "📄 Load & Process Documents",
    "🧮 Generate Embeddings",
    "🔍 Analyze Alignment",
    "📊 Calculate Metrics",
    "✨ Generate Insights"
]

st.subheader("Analysis Progress")
progress_cols = st.columns(len(progress_steps))
for idx, (col, step) in enumerate(zip(progress_cols, progress_steps)):
    with col:
        if idx < st.session_state[analysis_key]:
            st.success(f"✅\n{step}")
        elif idx == st.session_state[analysis_key]:
            st.info(f"▶️\n{step}")
        else:
            st.text(f"⏳\n{step}")

st.markdown("---")

# ============================================================
# STEP 1: Load & Process Documents
# ============================================================
if st.session_state[analysis_key] == 0:
    st.subheader("📄 Step 1: Load & Process Documents")
    st.write(f"Extract objectives and actions from {selected_year} documents.")
    
    if st.button("▶️ Start Processing", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Initialize processor
            status_text.info("🔧 Initializing document processor...")
            progress_bar.progress(20)
            time.sleep(0.3)
            processor = DocumentProcessor()
            
            # Step 2: Load strategic plan
            status_text.info("📋 Loading strategic plan...")
            progress_bar.progress(50)
            objectives = processor.load_strategic_plan(str(strategic_path))
            
            # Step 3: Load action plan
            status_text.info("📅 Loading action plan...")
            progress_bar.progress(80)
            actions = processor.load_action_plan(str(action_path))
            
            # Step 4: Complete
            progress_bar.progress(100)
            status_text.success(f"✅ Processing complete! Found {len(objectives)} objectives and {len(actions)} actions.")
            time.sleep(1)
            
            # Save to session state
            st.session_state[f'objectives_{selected_year}'] = objectives
            st.session_state[f'actions_{selected_year}'] = actions
            st.session_state[analysis_key] = 1
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")
            st.exception(e)

# ============================================================
# STEP 2: Generate Embeddings
# ============================================================
elif st.session_state[analysis_key] == 1:
    st.subheader("🧮 Step 2: Generate AI Embeddings")
    st.write("Converting text to AI-readable vectors...")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Strategic Objectives", len(st.session_state[f'objectives_{selected_year}']))
    with col2:
        st.metric("Action Items", len(st.session_state[f'actions_{selected_year}']))
    
    if st.button("▶️ Generate Embeddings", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load model
            status_text.info("🤖 Loading embedding model...")
            progress_bar.progress(30)
            time.sleep(0.3)
            
            # Step 2: Create engine
            status_text.info("⚙️ Initializing embedding engine...")
            progress_bar.progress(60)
            embedding_engine = EmbeddingEngine()
            
            # Step 3: Complete
            progress_bar.progress(100)
            status_text.success("✅ Embedding engine ready!")
            time.sleep(1)
            
            # Save to session state
            st.session_state[f'embedding_engine_{selected_year}'] = embedding_engine
            st.session_state[analysis_key] = 2
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error generating embeddings: {str(e)}")
            st.exception(e)

# ============================================================
# STEP 3: Analyze Strategic Alignment
# ============================================================
elif st.session_state[analysis_key] == 2:
    st.subheader("🔍 Step 3: Analyze Strategic Alignment")
    st.write("Calculating alignment between objectives and actions...")
    
    if st.button("▶️ Run Alignment Analysis", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Get data from session state
            embedding_engine = st.session_state[f'embedding_engine_{selected_year}']
            objectives = st.session_state[f'objectives_{selected_year}']
            actions = st.session_state[f'actions_{selected_year}']
            
            # Step 1: Create document processor
            status_text.info("📋 Creating document processor...")
            progress_bar.progress(10)
            time.sleep(0.2)
            doc_processor = DocumentProcessor()
            
            # Step 2: Create vector store
            status_text.info("📦 Creating vector store...")
            progress_bar.progress(15)
            time.sleep(0.2)
            vector_store = VectorStore(dimension=384)
            
            # Step 3: Generate action embeddings
            status_text.info("🧮 Generating action embeddings...")
            progress_bar.progress(30)
            action_texts = [action['text'] for action in actions]
            action_embeddings = embedding_engine.model.encode(action_texts)
            action_embeddings = np.array(action_embeddings)
            
            # Step 4: Prepare metadata
            status_text.info("📝 Preparing metadata...")
            progress_bar.progress(40)
            time.sleep(0.2)
            action_metadata = [
                {
                    'id': action['id'],
                    'title': action.get('title', action['text'][:50]),
                    'text': action['text'],
                    'pillar': action.get('pillar', 'Unknown')
                }
                for action in actions
            ]
            
            # Step 5: Add to vector store
            status_text.info("💾 Adding actions to vector store...")
            progress_bar.progress(50)
            time.sleep(0.2)
            vector_store.add_vectors(action_embeddings, action_metadata)
            
            # Step 6: Initialize sync engine
            status_text.info("⚙️ Initializing synchronization engine...")
            progress_bar.progress(60)
            time.sleep(0.2)
            sync_engine = SynchronizationEngine(
                doc_processor=doc_processor,
                embedding_engine=embedding_engine,
                vector_store=vector_store
            )
            
            # Step 7: Run analysis
            status_text.info("🔍 Analyzing synchronization (this may take 1-2 minutes)...")
            progress_bar.progress(75)
            results = sync_engine.analyze_synchronization(objectives, actions)
            
            # Step 8: Add metadata
            status_text.info("📊 Finalizing results...")
            progress_bar.progress(90)
            time.sleep(0.2)
            results['year'] = selected_year
            results['analysis_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Step 9: Save results
            progress_bar.progress(95)
            st.session_state[f'sync_results_{selected_year}'] = results
            st.session_state[f'vector_store_{selected_year}'] = vector_store
            
            # Step 10: Complete
            progress_bar.progress(100)
            overall = results.get('overall_alignment', {})
            status_text.success(f"✅ Alignment analysis complete! Overall Score: {overall.get('overall_score', 0):.1f}%")
            time.sleep(1)
            
            # Update progress
            st.session_state[analysis_key] = 3
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)

# ============================================================
# STEP 4: Calculate Metrics
# ============================================================
elif st.session_state[analysis_key] == 3:
    st.subheader("📊 Step 4: Calculate Performance Metrics")
    st.write("Computing KPIs...")
    
    if f'sync_results_{selected_year}' in st.session_state:
        results = st.session_state[f'sync_results_{selected_year}']
        
        # Display preview metrics
        overall = results.get('overall_alignment', {})
        distribution = overall.get('distribution', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Alignment", f"{overall.get('overall_score', 0):.1f}%")
        with col2:
            st.metric("Strong Alignments", distribution.get('strong', 0))
        with col3:
            st.metric("Critical Gaps", distribution.get('weak', 0))
    
    if st.button("▶️ Generate Metrics", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(0, 101, 20):
            status_text.info(f"📊 Calculating metrics... {i}%")
            progress_bar.progress(i)
            time.sleep(0.1)
        
        progress_bar.progress(100)
        status_text.success("✅ Metrics calculation complete!")
        time.sleep(1)
        
        st.session_state[analysis_key] = 4
        st.rerun()

# ============================================================
# STEP 5: Generate Executive Insights
# ============================================================
elif st.session_state[analysis_key] == 4:
    st.subheader("✨ Step 5: Generate Executive Insights")
    st.write("Creating AI-powered recommendations...")
    
    if st.button("▶️ Generate Insights", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Initialize generator
            status_text.info("🤖 Initializing AI generator...")
            progress_bar.progress(20)
            time.sleep(0.3)
            
            # Step 2: Generate summary
            status_text.info("✨ Generating executive insights with AI...")
            progress_bar.progress(60)
            
            try:
                summary_gen = ExecutiveSummaryGenerator()
                summary = summary_gen.generate_summary(
                    st.session_state[f'sync_results_{selected_year}']
                )
                st.session_state[f'executive_summary_{selected_year}'] = summary
            except Exception as e:
                st.warning(f"⚠️ Could not generate executive summary: {str(e)}")
                summary = "Executive summary generation not available."
            
            # Step 3: Save files
            status_text.info("💾 Saving reports...")
            progress_bar.progress(80)
            time.sleep(0.2)
            
            results_file = output_dir / "synchronization_report.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(st.session_state[f'sync_results_{selected_year}'], f, indent=2)
            
            summary_file = output_dir / "executive_summary.md"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            # Step 4: Complete
            progress_bar.progress(100)
            status_text.success("✅ All reports generated successfully!")
            time.sleep(1)
            
            # Update state
            st.session_state[f'analysis_complete_{selected_year}'] = True
            st.session_state[analysis_key] = 5
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error generating insights: {str(e)}")
            st.exception(e)

# ============================================================
# STEP 6: Analysis Complete
# ============================================================
elif st.session_state[analysis_key] == 5:
    st.success(f"### 🎉 Analysis Complete for {selected_year}!")
    
    # Display summary
    if f'sync_results_{selected_year}' in st.session_state:
        results = st.session_state[f'sync_results_{selected_year}']
        overall = results.get('overall_alignment', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Score", f"{overall.get('overall_score', 0):.1f}%")
        with col2:
            st.metric("Classification", overall.get('classification', 'N/A'))
        with col3:
            st.metric("Coverage Rate", f"{overall.get('coverage_rate', 0):.1f}%")
        with col4:
            st.metric("Total Comparisons", 
                     f"{overall.get('total_objectives', 0)} × {overall.get('total_actions', 0)}")
    
    st.info("👉 Go to **'📊 View Results'** page to see detailed insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Analyze Another Year", use_container_width=True):
            st.info("👉 Change year selector above and start new analysis")
    
    with col2:
        if st.button("📊 View Results Now", type="primary", use_container_width=True):
            st.switch_page("pages/view_results.py")

# ============================================================
# Download Results Section
# ============================================================
if st.session_state.get(f'analysis_complete_{selected_year}'):
    st.markdown("---")
    st.subheader("📥 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        results_file = output_dir / "synchronization_report.json"
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                st.download_button(
                    label="📄 Download Analysis Results (JSON)",
                    data=f.read(),
                    file_name=f"synchronization_report_{selected_year}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    with col2:
        summary_file = output_dir / "executive_summary.md"
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                st.download_button(
                    label="📋 Download Executive Summary (MD)",
                    data=f.read(),
                    file_name=f"executive_summary_{selected_year}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(f"💡 **Tip:** Analysis results are saved in `outputs/{selected_year}/` directory")