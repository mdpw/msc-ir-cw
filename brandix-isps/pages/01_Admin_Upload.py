"""
Brandix ISPS - Admin Upload Page
Year-specific document upload and management
"""

import streamlit as st
import os
import json
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Admin Upload", page_icon="None", layout="wide")

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
    
    /* Info boxes - styled like feature-box but more descriptive */
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
    
    /* Streamlit notification components */
    div[data-baseweb="notification"] {
        background-color: rgba(28, 131, 225, 0.15) !important;
        border-left: 4px solid #1c83e1 !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
    }
    
    /* Success messages */
    .element-container div[data-testid="stMarkdown"] > div:has(> div[data-testid="stNotification"][kind="success"]) {
        background-color: rgba(76, 175, 80, 0.15) !important;
        border-left: 4px solid #4caf50 !important;
    }
    
    /* Warning messages */
    .element-container div[data-testid="stMarkdown"] > div:has(> div[data-testid="stNotification"][kind="warning"]) {
        background-color: rgba(255, 152, 0, 0.15) !important;
        border-left: 4px solid #ff9800 !important;
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
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px dashed rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 20px;
    }
    
    /* Text visibility */
    p, span, label, div {
        color: #e0e0e0 !important;
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
    
    /* Strong/bold text */
    strong {
        color: #ffffff !important;
    }
    
    /* Dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Download buttons */
    .stDownloadButton>button {
        background-color: rgba(76, 175, 80, 0.2) !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        color: #81c784 !important;
    }
    
    .stDownloadButton>button:hover {
        background-color: rgba(76, 175, 80, 0.3) !important;
        border: 1px solid #4caf50 !important;
    }
    
    /* Delete buttons */
    div.stButton > button:has(div[data-testid="stMarkdownContainer"] p:contains("Delete")) {
        background-color: rgba(244, 67, 54, 0.2) !important;
        border: 1px solid rgba(244, 67, 54, 0.3) !important;
        color: #ef5350 !important;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1><i class="fas fa-upload fa-icon-large"></i>Document Upload & Management</h1>', unsafe_allow_html=True)
st.markdown("### Upload Strategic Plan and Action Plans by Year")
st.markdown("---")

# Available years (Strategic Plan 2025-2030)
AVAILABLE_YEARS = ["2026", "2027", "2028", "2029", "2030"]
UPLOAD_BASE = Path(__file__).parent.parent / "data" / "uploaded"
UPLOAD_BASE.mkdir(parents=True, exist_ok=True)

# Initialize session state
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = "2026"

# Year Selection
st.markdown('<h3><i class="fas fa-calendar-alt fa-icon"></i>Step 1: Select Planning Year</h3>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
<i class="fas fa-info-circle fa-icon-small"></i><strong>Information:</strong> The Strategic Plan covers 2025-2030. Select the action plan year you want to upload and analyze.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])

with col1:
    selected_year = st.selectbox(
        "Select Year",
        AVAILABLE_YEARS,
        index=AVAILABLE_YEARS.index(st.session_state.selected_year),
        key='year_selector'
    )

with col2:
    # Update session state
    if selected_year != st.session_state.selected_year:
        st.session_state.selected_year = selected_year
        st.rerun()
    
    st.success(f"Selected Year: **{selected_year}**")

# Create year-specific directory
year_path = UPLOAD_BASE / selected_year
year_path.mkdir(parents=True, exist_ok=True)

st.markdown("---")

# Check existing files
strategic_path = year_path / "strategic_plan.docx"
action_path = year_path / "action_plan.docx"
metadata_path = year_path / "metadata.json"

strategic_exists = strategic_path.exists()
action_exists = action_path.exists()

# Load existing metadata
existing_metadata = {}
if metadata_path.exists():
    with open(metadata_path, 'r') as f:
        existing_metadata = json.load(f)

# File Upload Section
st.markdown('<h3><i class="fas fa-folder-open fa-icon"></i>Step 2: Upload Documents</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('### <i class="fas fa-file-alt fa-icon-small"></i>Strategic Plan (2025-2030)', unsafe_allow_html=True)
    
    if strategic_exists:
        st.warning("Strategic Plan already exists for this year!")
        
        # Show existing file info
        file_size = strategic_path.stat().st_size / 1024  # KB
        upload_date = existing_metadata.get('strategic_plan_upload_date', 'Unknown')
        
        st.info(f"""
        **Existing File:**
        - **Filename:** `{existing_metadata.get('strategic_plan_filename', 'strategic_plan.docx')}`
        - Upload Date: {upload_date}
        - File Size: {file_size:.1f} KB
        """)
        
        # Download existing file
        with open(strategic_path, 'rb') as f:
            st.download_button(
                label="Download Existing Strategic Plan",
                data=f,
                file_name=f"strategic_plan_{selected_year}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Option to replace
        replace_strategic = st.checkbox("Replace existing Strategic Plan", key='replace_strategic')
        
        if replace_strategic:
            st.warning("Uploading a new file will **overwrite** the existing Strategic Plan!")
    else:
        replace_strategic = True  # No file exists, so always allow upload
    
    if replace_strategic or not strategic_exists:
        strategic_file = st.file_uploader(
            "Choose Strategic Plan (.docx)",
            type=['docx'],
            key='strategic_uploader'
        )
        
        if strategic_file:
            # Save file
            with open(strategic_path, "wb") as f:
                f.write(strategic_file.getbuffer())
            
            # Update metadata
            existing_metadata['strategic_plan_upload_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            existing_metadata['strategic_plan_filename'] = strategic_file.name
            existing_metadata['strategic_plan_size'] = strategic_file.size
            
            with open(metadata_path, 'w') as f:
                json.dump(existing_metadata, f, indent=2)
            
            st.success("Strategic Plan uploaded successfully!")
            st.metric("File Size", f"{strategic_file.size / 1024:.1f} KB")
            st.rerun()

with col2:
    st.markdown(f'### <i class="fas fa-calendar-check fa-icon-small"></i>Action Plan for {selected_year}', unsafe_allow_html=True)
    
    if action_exists:
        st.warning("Action Plan already exists for this year!")
        
        # Show existing file info
        file_size = action_path.stat().st_size / 1024  # KB
        upload_date = existing_metadata.get('action_plan_upload_date', 'Unknown')
        
        st.info(f"""
        **Existing File:**
        - **Filename:** `{existing_metadata.get('action_plan_filename', 'action_plan.docx')}`
        - Upload Date: {upload_date}
        - File Size: {file_size:.1f} KB
        """)
        
        # Download existing file
        with open(action_path, 'rb') as f:
            st.download_button(
                label=f"Download Existing Action Plan {selected_year}",
                data=f,
                file_name=f"action_plan_{selected_year}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Option to replace
        replace_action = st.checkbox("Replace existing Action Plan", key='replace_action')
        
        if replace_action:
            st.warning("Uploading a new file will **overwrite** the existing Action Plan!")
    else:
        replace_action = True  # No file exists, so always allow upload
    
    if replace_action or not action_exists:
        action_file = st.file_uploader(
            f"Choose Action Plan for {selected_year} (.docx)",
            type=['docx'],
            key='action_uploader'
        )
        
        if action_file:
            # Save file
            with open(action_path, "wb") as f:
                f.write(action_file.getbuffer())
            
            # Update metadata
            existing_metadata['action_plan_upload_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            existing_metadata['action_plan_filename'] = action_file.name
            existing_metadata['action_plan_size'] = action_file.size
            existing_metadata['upload_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            existing_metadata['year'] = selected_year
            
            with open(metadata_path, 'w') as f:
                json.dump(existing_metadata, f, indent=2)
            
            st.success(f"Action Plan {selected_year} uploaded successfully!")
            st.metric("File Size", f"{action_file.size / 1024:.1f} KB")
            st.rerun()

st.markdown("---")

# Status Overview
st.markdown(f'<h3><i class="fas fa-chart-pie fa-icon"></i>Upload Status for Year {selected_year}</h3>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if strategic_exists:
        st.success("Strategic Plan Ready")
    else:
        st.warning("Strategic Plan Pending")

with col2:
    if action_exists:
        st.success("Action Plan Ready")
    else:
        st.warning("Action Plan Pending")

with col3:
    both_ready = strategic_exists and action_exists
    if both_ready:
        st.success("Ready to Analyze")
    else:
        st.info("Upload Both Documents")

# Document Management
if strategic_exists or action_exists:
    st.markdown("---")
    st.markdown('<h3><i class="fas fa-tasks fa-icon"></i>Document Management</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if strategic_exists:
            st.markdown("#### Strategic Plan")
            with open(strategic_path, 'rb') as f:
                st.download_button(
                    label="Download Strategic Plan",
                    data=f,
                    file_name=f"strategic_plan_{selected_year}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
            
            if st.button("Delete Strategic Plan", use_container_width=True):
                strategic_path.unlink()
                st.success("Strategic Plan deleted!")
                st.rerun()
    
    with col2:
        if action_exists:
            st.markdown("#### Action Plan")
            with open(action_path, 'rb') as f:
                st.download_button(
                    label=f"Download Action Plan {selected_year}",
                    data=f,
                    file_name=f"action_plan_{selected_year}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
            
            if st.button(f"Delete Action Plan {selected_year}", use_container_width=True):
                action_path.unlink()
                st.success("Action Plan deleted!")
                st.rerun()

    # Full Reset Option
    st.markdown("---")
    st.markdown('### <i class="fas fa-exclamation-triangle fa-icon" style="color: #f44336;"></i> Danger Zone', unsafe_allow_html=True)
    with st.expander("Reset All Data for this Year"):
        st.error(f"This will delete ALL data for {selected_year}, including uploaded documents and analysis results.")
        if st.button(f"CONFIRM FULL RESET FOR {selected_year}", type="secondary", use_container_width=True):
            import shutil
            
            # 1. Delete source uploads
            if year_path.exists():
                shutil.rmtree(year_path)
            
            # 2. Delete analysis outputs
            output_dir = Path(f"outputs/{selected_year}")
            if output_dir.exists():
                shutil.rmtree(output_dir)
            
            # 3. Clear session state related to this year
            if 'analysis_complete' in st.session_state:
                st.session_state.analysis_complete[selected_year] = False

            # 4. Clear cached data so other pages refresh
            st.cache_data.clear()

            st.success(f"All data for {selected_year} has been reset.")
            st.rerun()

# Next Steps
if both_ready:
    st.markdown("---")
    st.markdown('### <i class="fas fa-check-circle fa-icon-small" style="color: #4caf50;"></i> Documents Ready for Analysis!', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <i class="fas fa-arrow-right fa-icon-small"></i><strong>Next Step:</strong> Go to <strong>Run Analysis</strong> page to process your documents
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Start Analysis Now", type="primary", use_container_width=True):
        st.switch_page("pages/02_Run_Analysis.py")

# All uploaded years summary
st.markdown("---")
st.markdown('<h3><i class="fas fa-list fa-icon"></i>All Uploaded Years Summary</h3>', unsafe_allow_html=True)

all_years_data = []
for year in AVAILABLE_YEARS:
    y_path = UPLOAD_BASE / year
    s_exists = (y_path / "strategic_plan.docx").exists()
    a_exists = (y_path / "action_plan.docx").exists()
    
    if s_exists or a_exists:
        all_years_data.append({
            'Year': year,
            'Strategic Plan': '<i class="fas fa-check" style="color: #4caf50;"></i>' if s_exists else '<i class="fas fa-times" style="color: #f44336;"></i>',
            'Action Plan': '<i class="fas fa-check" style="color: #4caf50;"></i>' if a_exists else '<i class="fas fa-times" style="color: #f44336;"></i>',
            'Status': '<i class="fas fa-check-circle" style="color: #4caf50;"></i> Ready' if (s_exists and a_exists) else '<i class="fas fa-clock" style="color: #ff9800;"></i> Partial'
        })

if all_years_data:
    import pandas as pd
    df = pd.DataFrame(all_years_data)
    # Using markdown to render HTML in dataframe
    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="info-box">
    <i class="fas fa-info-circle fa-icon-small"></i><strong>No documents uploaded yet.</strong> Start by selecting a year and uploading documents above.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div class="info-box">
<i class="fas fa-lightbulb fa-icon-small"></i><strong>Tip:</strong> You can manage different years independently. Strategic Plan is shared across years, while Action Plans are year-specific.
</div>
""", unsafe_allow_html=True)
