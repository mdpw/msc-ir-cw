"""
Testing & Evaluation Dashboard
Streamlit page for running and viewing system tests
"""

import streamlit as st
import sys
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from testing_framework import TestingFramework

st.set_page_config(page_title="Testing & Evaluation", page_icon=None, layout="wide")

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
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: none;
        color: #a0a0a0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(77, 166, 255, 0.1) !important;
        color: #4da6ff !important;
        border-top: 3px solid #4da6ff !important;
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
        background-color: rgba(28, 131, 225, 0.4);
        border: 1px solid #1c83e1;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
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

st.markdown('<h1><i class="fas fa-flask fa-icon-large"></i>Testing & Evaluation Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### System Accuracy Testing & Performance Validation")
st.markdown("---")

# Available years
AVAILABLE_YEARS = ["2026", "2027", "2028", "2029", "2030"]
OUTPUTS_BASE = Path("outputs")
GT_BASE = Path("data/ground_truth")

# Year selection
col1, col2 = st.columns([1, 3])

with col1:
    selected_year = st.selectbox(
        "Select Year",
        AVAILABLE_YEARS,
        index=0,
        key='test_year_selector'
    )

with col2:
    st.info(f"Testing results for **Year {selected_year}**")

st.markdown("---")

# Check if test results exist
test_results_path = OUTPUTS_BASE / selected_year / "test_results.json"
sync_report_path = OUTPUTS_BASE / selected_year / "synchronization_report.json"
gt_path = GT_BASE / f"{selected_year}_ground_truth.json"

test_exists = test_results_path.exists()
sync_exists = sync_report_path.exists()
gt_exists = gt_path.exists()

# Status overview
st.markdown('<h3><i class="fas fa-list-check fa-icon"></i>Testing Status</h3>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if sync_exists:
        st.success("Analysis Complete")
    else:
        st.warning("Analysis Pending")

with col2:
    if gt_exists:
        st.success("Ground Truth Ready")
    else:
        st.warning("Ground Truth Needed")

with col3:
    if test_exists:
        st.success("Tests Complete")
    else:
        st.info("Tests Not Run")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Run Tests",
    "Test Results",
    "Ground Truth",
    "Performance"
])

# ============================================================
# TAB 1: Run Tests
# ============================================================

with tab1:
    st.markdown('<h3><i class="fas fa-play-circle fa-icon"></i>Run Comprehensive Tests</h3>', unsafe_allow_html=True)
    
    if not sync_exists:
        st.error("No analysis results found!")
        st.info("Go to **'Run Analysis'** page and complete analysis first")
        st.stop()
    
    st.write("Execute the complete testing suite to validate system accuracy")
    
    # Test options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('### <i class="fas fa-tasks fa-icon-small"></i>Available Tests', unsafe_allow_html=True)
        st.markdown("""
        1. **Alignment Classification** - Accuracy of Strong/Moderate/Weak labels
        2. **Similarity Scores** - Numerical score accuracy (MSE, MAE, correlation)
        3. **LLM Improvements** - Quality of AI-generated suggestions
        4. **System Performance** - Speed and efficiency benchmarks
        5. **Coverage Analysis** - Test coverage statistics
        """)
    
    with col2:
        st.markdown('### <i class="fas fa-check-double fa-icon-small"></i>Prerequisites', unsafe_allow_html=True)
        
        prereqs = []
        if sync_exists:
            prereqs.append("Synchronization analysis complete")
        else:
            prereqs.append("Synchronization analysis required")
        
        if gt_exists:
            # Count ground truth pairs
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
                gt_count = len(gt_data.get('objective_action_pairs', []))
            prereqs.append(f"Ground truth ready ({gt_count} pairs)")
        else:
            prereqs.append("Ground truth recommended (optional)")
        
        improvements_path = OUTPUTS_BASE / selected_year / "improvements.json"
        if improvements_path.exists():
            prereqs.append("LLM improvements available")
        else:
            prereqs.append("LLM improvements optional")
        
        for prereq in prereqs:
            st.markdown(f"- {prereq}")
    
    st.markdown("---")
    
    # Run button
    if st.button("Run Complete Test Suite", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Import required modules
            from synchronization_engine import SynchronizationEngine
            from document_processor import DocumentProcessor
            from embedding_engine import EmbeddingEngine
            from vector_store import VectorStore
            
            # Load synchronization report
            status_text.info("Loading synchronization report...")
            progress_bar.progress(10)
            
            with open(sync_report_path, 'r', encoding='utf-8') as f:
                sync_report = json.load(f)
            
            # Load improvements if available
            improvements_data = None
            if improvements_path.exists():
                with open(improvements_path, 'r', encoding='utf-8') as f:
                    improvements_data = json.load(f)
            
            # Initialize system components
            status_text.info("Initializing system components...")
            progress_bar.progress(20)
            
            upload_dir = Path("data/uploaded") / selected_year
            strategic_path = upload_dir / "strategic_plan.docx"
            action_path = upload_dir / "action_plan.docx"
            
            processor = DocumentProcessor()
            objectives = processor.load_strategic_plan(str(strategic_path))
            actions = processor.load_action_plan(str(action_path))
            
            engine = EmbeddingEngine()
            vs = VectorStore(dimension=384)
            
            sync_engine = SynchronizationEngine(processor, engine, vs)
            
            # Embed documents
            status_text.info("Generating embeddings...")
            progress_bar.progress(40)
            
            engine.embed_objectives(objectives)
            engine.embed_actions(actions)
            
            # Initialize testing framework
            status_text.info("Initializing testing framework...")
            progress_bar.progress(60)
            
            testing_framework = TestingFramework()
            
            # Run tests
            status_text.info("Running comprehensive tests (this may take 1-2 minutes)...")
            progress_bar.progress(70)
            
            test_results = testing_framework.run_comprehensive_tests(
                sync_report=sync_report,
                sync_engine=sync_engine,
                improvements_data=improvements_data,
                ground_truth_path=str(gt_path),
                expert_feedback=None
            )
            
            # Save results
            status_text.info("Saving test results...")
            progress_bar.progress(90)
            
            testing_framework.save_test_results(test_results, str(test_results_path))
            
            # Complete
            progress_bar.progress(100)
            status_text.success("All tests complete!")
            
            st.success("### Testing Complete!")
            
            # Show summary
            if 'overall_assessment' in test_results:
                assessment = test_results['overall_assessment']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Tests Passed", assessment['tests_passed'])
                with col2:
                    st.metric("Tests Failed", assessment['tests_failed'])
                with col3:
                    st.metric("Overall Grade", assessment['overall_grade'])
            
            st.info("Switch to **'Test Results'** tab to view detailed results")
            st.rerun()
            
        except Exception as e:
            progress_bar.progress(0)
            status_text.error(f"Error during testing: {str(e)}")
            st.exception(e)

# ============================================================
# TAB 2: Test Results
# ============================================================

with tab2:
    st.markdown('<h3><i class="fas fa-chart-pie fa-icon"></i>Test Results & Metrics</h3>', unsafe_allow_html=True)
    
    if not test_exists:
        st.warning("No test results available yet")
        st.info("Go to **'Run Tests'** tab to execute tests")
    else:
        # Load test results
        with open(test_results_path, 'r', encoding='utf-8') as f:
            test_results = json.load(f)
        
        # Overall Assessment
        if 'overall_assessment' in test_results:
            st.markdown("### Overall Assessment")
            
            assessment = test_results['overall_assessment']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Tests Run", assessment['tests_completed'])
            with col2:
                st.metric("Passed", assessment['tests_passed'], 
                         delta="Good" if assessment['tests_passed'] > assessment['tests_failed'] else None)
            with col3:
                st.metric("Failed", assessment['tests_failed'],
                         delta_color="inverse")
            with col4:
                grade = assessment['overall_grade']
                st.metric("Grade", grade)
            
            # Recommendations
            if assessment.get('recommendations'):
                st.markdown('#### <i class="fas fa-thumbtack fa-icon-small"></i>Recommendations', unsafe_allow_html=True)
                for i, rec in enumerate(assessment['recommendations'], 1):
                    st.warning(f"{i}. {rec}")
            
            st.markdown("---")
        
        # Individual Test Results
        st.markdown("### Detailed Test Results")
        
        # Test 1: Alignment Classification
        if 'alignment_classification' in test_results:
            result = test_results['alignment_classification']
            
            if result.get('status') != 'skipped' and 'overall_accuracy' in result:
                with st.expander("Test 1: Alignment Classification Accuracy", expanded=True):
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Overall Accuracy", 
                                 f"{result['overall_accuracy']:.1%}")
                    with col2:
                        st.metric("Weighted Precision",
                                 f"{result['weighted_metrics']['precision']:.1%}")
                    with col3:
                        st.metric("Weighted F1-Score",
                                 f"{result['weighted_metrics']['f1_score']:.1%}")
                    
                    # Per-class metrics table
                    st.markdown("#### Per-Class Performance")
                    
                    class_data = []
                    for class_name, metrics in result['per_class_metrics'].items():
                        class_data.append({
                            'Class': class_name,
                            'Precision': f"{metrics['precision']:.2%}",
                            'Recall': f"{metrics['recall']:.2%}",
                            'F1-Score': f"{metrics['f1_score']:.2%}",
                            'Support': metrics['support']
                        })
                    
                    df = pd.DataFrame(class_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Confusion Matrix
                    st.markdown("#### Confusion Matrix")
                    
                    cm = result['confusion_matrix']['matrix']
                    labels = result['confusion_matrix']['labels']
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=labels,
                        y=labels,
                        colorscale='Blues',
                        text=cm,
                        texttemplate='%{text}',
                        textfont={"size": 16, "color": "white"},
                        hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title="Confusion Matrix",
                        xaxis_title="Predicted Class",
                        yaxis_title="Actual Class",
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e0e0e0'),
                        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Test 2: Similarity Scores
        if 'similarity_scores' in test_results:
            result = test_results['similarity_scores']
            
            if result.get('status') != 'skipped' and 'mean_absolute_error' in result:
                with st.expander("Test 2: Similarity Score Accuracy", expanded=True):
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("MAE", f"{result['mean_absolute_error']:.4f}")
                    with col2:
                        st.metric("RMSE", f"{result['root_mean_squared_error']:.4f}")
                    with col3:
                        st.metric("Correlation", f"{result['correlation_coefficient']:.4f}")
                    with col4:
                        st.metric("Within ±10%", f"{result['within_10_percent']:.1%}")
                    
                    st.markdown("#### Score Statistics")
                    stats = result['score_statistics']
                    
                    stats_data = pd.DataFrame({
                        'Metric': ['Mean', 'Std Dev'],
                        'Ground Truth': [f"{stats['mean_true']:.4f}", f"{stats['std_true']:.4f}"],
                        'Predictions': [f"{stats['mean_pred']:.4f}", f"{stats['std_pred']:.4f}"]
                    })
                    
                    st.dataframe(stats_data, use_container_width=True, hide_index=True)
        
        # Test 3: LLM Improvements
        if 'llm_improvements' in test_results:
            result = test_results['llm_improvements']
            
            if result.get('status') != 'skipped' and 'objectives_processed' in result:
                with st.expander("Test 3: LLM Improvement Quality", expanded=True):
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Objectives Processed", result['objectives_processed'])
                    with col2:
                        st.metric("Avg Suggestions/Obj", 
                                 f"{result['average_suggestions_per_objective']:.1f}")
                    with col3:
                        st.metric("Quality", result['quality_assessment'])
                    
                    st.markdown("#### Category Coverage")
                    
                    coverage = result['category_coverage_rate']
                    categories = list(coverage.keys())
                    rates = [coverage[cat] for cat in categories]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[cat.replace('_', ' ').title() for cat in categories],
                            y=rates,
                            text=[f"{r:.0%}" for r in rates],
                            textposition='auto',
                            marker_color='#4da6ff'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Suggestion Category Coverage",
                        xaxis_title="Category",
                        yaxis_title="Coverage Rate",
                        yaxis_tickformat='.0%',
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e0e0e0'),
                        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Test 4: Performance
        if 'performance' in test_results:
            result = test_results['performance']
            
            with st.expander("⚡ Test 4: System Performance", expanded=True):
                
                benchmarks = result['benchmarks']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Processing Times")
                    st.metric("Document Processing", f"{benchmarks['document_processing_time']:.3f}s")
                    st.metric("Embedding (10 items)", f"{benchmarks['embedding_time_per_10_items']:.3f}s")
                    st.metric("Similarity Matrix", f"{benchmarks['similarity_matrix_calculation_time']:.3f}s")
                
                with col2:
                    st.markdown("#### Scalability")
                    scalability = benchmarks['scalability']
                    st.metric("Total Comparisons", f"{scalability['comparisons_required']:,}")
                    st.metric("Time/Comparison", f"{scalability['time_per_comparison_ms']:.4f}ms")
                    st.metric("Throughput", f"{benchmarks['embedding_throughput']:.1f} items/s")
        
        # Download results
        st.markdown("---")
        st.markdown('<h3><i class="fas fa-download fa-icon"></i>Download Results</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            with open(test_results_path, 'r') as f:
                st.download_button(
                    label="Download Full Results (JSON)",
                    data=f.read(),
                    file_name=f"test_results_{selected_year}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col2:
            summary_path = test_results_path.with_name('test_results_summary.md')
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    st.download_button(
                        label="Download Summary (Markdown)",
                        data=f.read(),
                        file_name=f"test_summary_{selected_year}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )

# ============================================================
# TAB 3: Ground Truth Management
# ============================================================

with tab3:
    st.markdown('<h3><i class="fas fa-file-signature fa-icon"></i>Ground Truth Management</h3>', unsafe_allow_html=True)
    
    st.info("""
    **Ground truth** is expert-validated data used to test system accuracy.
    It consists of objective-action pairs with expected alignment scores.
    """)
    
    if gt_exists:
        st.success("Ground truth file exists!")
        
        # Load and display
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        # Stats
        col1, col2, col3 = st.columns(3)
        
        pairs = gt_data.get('objective_action_pairs', [])
        annotated = [p for p in pairs if p.get('expected_alignment') != 'TO_BE_ANNOTATED']
        
        with col1:
            st.metric("Total Pairs", len(pairs))
        with col2:
            st.metric("Annotated", len(annotated))
        with col3:
            completion = len(annotated) / len(pairs) * 100 if pairs else 0
            st.metric("Completion", f"{completion:.0f}%")
        
        # Display sample
        st.markdown("#### Sample Ground Truth Data")
        
        sample_pairs = pairs[:5]
        sample_data = []
        
        for pair in sample_pairs:
            # Safe access to system_prediction
            sys_pred = pair.get('system_prediction', {})
            
            sample_data.append({
                'Objective ID': pair.get('objective_id', 'N/A'),
                'Action ID': pair.get('action_id', 'N/A'),
                'Expected': pair.get('expected_alignment', 'TO_BE_ANNOTATED'),
                'Score': pair.get('expected_score', 0.0),
                'System Pred': sys_pred.get('alignment', 'N/A') if sys_pred else 'N/A'
            })
        
        df = pd.DataFrame(sample_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Download
        with open(gt_path, 'r') as f:
            st.download_button(
                label="Download Ground Truth File",
                data=f.read(),
                file_name=f"{selected_year}_ground_truth.json",
                mime="application/json",
                use_container_width=True
            )
    
    else:
        st.warning("No ground truth file found")
        
        st.markdown("""
        ### Create Ground Truth Template
        
        Generate a template with system predictions that experts can annotate.
        """)
        
        if st.button("Create Ground Truth Template", type="primary", use_container_width=True):
            
            if not sync_exists:
                st.error("Run analysis first to create template")
            else:
                try:
                    # Load sync report
                    with open(sync_report_path, 'r') as f:
                        sync_report = json.load(f)
                    
                    # Create template
                    from run_tests import create_sample_ground_truth
                    import io
                    from contextlib import redirect_stdout
                    
                    f = io.StringIO()
                    with redirect_stdout(f):
                        create_sample_ground_truth(selected_year)
                    
                    st.success("Ground truth template created!")
                    st.info(f"File saved to: {gt_path}")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.exception(e)

# ============================================================
# TAB 4: Performance Benchmarks
# ============================================================

with tab4:
    st.markdown('<h3><i class="fas fa-tachometer-alt fa-icon"></i>Performance Benchmarks</h3>', unsafe_allow_html=True)
    
    if not test_exists:
        st.warning("Run tests first to see performance benchmarks")
    else:
        with open(test_results_path, 'r') as f:
            test_results = json.load(f)
        
        if 'performance' in test_results:
            result = test_results['performance']
            benchmarks = result['benchmarks']
            
            # Processing time breakdown
            st.markdown("### Processing Time Breakdown")
            
            times = {
                'Document Processing': benchmarks['document_processing_time'],
                'Embedding Generation': benchmarks['embedding_time_per_10_items'] * 
                    (benchmarks['scalability']['objectives_count'] / 10),
                'Similarity Calculation': benchmarks['similarity_matrix_calculation_time']
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(times.keys()),
                    y=list(times.values()),
                    text=[f"{v:.2f}s" for v in times.values()],
                    textposition='auto',
                    marker_color='#4da6ff'
                )
            ])
            
            fig.update_layout(
                title="Component Processing Times",
                xaxis_title="Component",
                yaxis_title="Time (seconds)",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Scalability analysis
            st.markdown("### Scalability Analysis")
            
            scalability = benchmarks['scalability']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Objectives Processed", scalability['objectives_count'])
                st.metric("Actions Processed", scalability['actions_count'])
                st.metric("Total Comparisons", f"{scalability['comparisons_required']:,}")
            
            with col2:
                st.metric("Embedding Throughput", f"{benchmarks['embedding_throughput']:.1f} items/s")
                st.metric("Time per Comparison", f"{scalability['time_per_comparison_ms']:.4f}ms")
                st.metric("Total Analysis Time", f"{benchmarks['estimated_total_analysis_time']:.2f}s")

st.markdown("---")
st.markdown("""
<div class="info-box">
<i class="fas fa-info-circle fa-icon-small"></i><strong>Note:</strong> Testing Framework validates system accuracy and performance | Coursework Requirement 3.8
</div>
""", unsafe_allow_html=True)
