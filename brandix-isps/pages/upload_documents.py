import streamlit as st
import os
from pathlib import Path

st.set_page_config(page_title="Upload Documents", page_icon="📤", layout="wide")

st.title("📤 Document Upload & Management")
st.markdown("---")

# Create upload directory if it doesn't exist
UPLOAD_DIR = Path("data/uploaded")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize session state for uploaded files
if 'strategic_plan_uploaded' not in st.session_state:
    st.session_state.strategic_plan_uploaded = False
if 'action_plan_uploaded' not in st.session_state:
    st.session_state.action_plan_uploaded = False

col1, col2 = st.columns(2)

with col1:
    st.subheader("Strategic Plan (2025-2030)")
    st.info("Upload your 5-year strategic plan document")
    
    strategic_file = st.file_uploader(
        "Choose Strategic Plan (.docx)",
        type=['docx'],
        key='strategic_uploader'
    )
    
    if strategic_file:
        # Save uploaded file
        strategic_path = UPLOAD_DIR / "strategic_plan.docx"
        with open(strategic_path, "wb") as f:
            f.write(strategic_file.getbuffer())
        
        st.session_state.strategic_plan_uploaded = True
        st.session_state.strategic_plan_path = str(strategic_path)
        st.success(f"Uploaded: {strategic_file.name}")
        st.metric("File Size", f"{strategic_file.size / 1024:.1f} KB")

with col2:
    st.subheader("Action Plan (Year 1)")
    st.info("Upload your annual action plan document")
    
    action_file = st.file_uploader(
        "Choose Action Plan (.docx)",
        type=['docx'],
        key='action_uploader'
    )
    
    if action_file:
        # Save uploaded file
        action_path = UPLOAD_DIR / "action_plan.docx"
        with open(action_path, "wb") as f:
            f.write(action_file.getbuffer())
        
        st.session_state.action_plan_uploaded = True
        st.session_state.action_plan_path = str(action_path)
        st.success(f"Uploaded: {action_file.name}")
        st.metric("File Size", f"{action_file.size / 1024:.1f} KB")

st.markdown("---")

# Status Overview
st.subheader("Upload Status")
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.strategic_plan_uploaded:
        st.success("Strategic Plan Ready")
    else:
        st.warning("Strategic Plan Pending")

with col2:
    if st.session_state.action_plan_uploaded:
        st.success("Action Plan Ready")
    else:
        st.warning("Action Plan Pending")

with col3:
    both_ready = (st.session_state.strategic_plan_uploaded and 
                  st.session_state.action_plan_uploaded)
    if both_ready:
        st.success("Ready to Analyze")
    else:
        st.info("Upload Both Documents")

# Next Steps
if both_ready:
    st.markdown("---")
    st.success("All Documents Uploaded!")
    st.info("Go to 'Run Analysis' page to process your documents")