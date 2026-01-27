"""
Brandix ISPS - Admin Upload Page
Year-specific document upload and management
"""

import streamlit as st
import os
import json
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Admin Upload", page_icon="📤", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {width: 100%;}
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📤 Document Upload & Management")
st.markdown("### Upload Strategic Plan and Action Plans by Year")
st.markdown("---")

# Available years (Strategic Plan 2025-2030)
AVAILABLE_YEARS = ["2026", "2027", "2028", "2029", "2030"]
UPLOAD_BASE = Path("data/uploaded")
UPLOAD_BASE.mkdir(parents=True, exist_ok=True)

# Initialize session state
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = "2026"

# Year Selection
st.subheader("📅 Step 1: Select Planning Year")
st.markdown("""
<div class="info-box">
💡 <strong>Information:</strong> The Strategic Plan covers 2025-2030. Select the action plan year you want to upload and analyze.
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
    
    st.success(f"✅ Selected Year: **{selected_year}**")

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
st.subheader("📂 Step 2: Upload Documents")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Strategic Plan (2025-2030)")
    
    if strategic_exists:
        st.warning("⚠️ **Strategic Plan already exists for this year!**")
        
        # Show existing file info
        file_size = strategic_path.stat().st_size / 1024  # KB
        upload_date = existing_metadata.get('strategic_plan_upload_date', 'Unknown')
        
        st.info(f"""
        **Existing File:**
        - Upload Date: {upload_date}
        - File Size: {file_size:.1f} KB
        """)
        
        # Download existing file
        with open(strategic_path, 'rb') as f:
            st.download_button(
                label="📥 Download Existing Strategic Plan",
                data=f,
                file_name=f"strategic_plan_{selected_year}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Option to replace
        replace_strategic = st.checkbox("🔄 Replace existing Strategic Plan", key='replace_strategic')
        
        if replace_strategic:
            st.warning("⚠️ Uploading a new file will **overwrite** the existing Strategic Plan!")
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
            
            st.success(f"✅ Strategic Plan uploaded successfully!")
            st.metric("File Size", f"{strategic_file.size / 1024:.1f} KB")
            st.rerun()

with col2:
    st.markdown("### 📅 Action Plan (Year-Specific)")
    
    if action_exists:
        st.warning("⚠️ **Action Plan already exists for this year!**")
        
        # Show existing file info
        file_size = action_path.stat().st_size / 1024  # KB
        upload_date = existing_metadata.get('action_plan_upload_date', 'Unknown')
        
        st.info(f"""
        **Existing File:**
        - Upload Date: {upload_date}
        - File Size: {file_size:.1f} KB
        """)
        
        # Download existing file
        with open(action_path, 'rb') as f:
            st.download_button(
                label=f"📥 Download Existing Action Plan {selected_year}",
                data=f,
                file_name=f"action_plan_{selected_year}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Option to replace
        replace_action = st.checkbox("🔄 Replace existing Action Plan", key='replace_action')
        
        if replace_action:
            st.warning("⚠️ Uploading a new file will **overwrite** the existing Action Plan!")
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
            
            st.success(f"✅ Action Plan {selected_year} uploaded successfully!")
            st.metric("File Size", f"{action_file.size / 1024:.1f} KB")
            st.rerun()

st.markdown("---")

# Status Overview
st.subheader("📊 Upload Status for Year " + selected_year)

col1, col2, col3 = st.columns(3)

with col1:
    if strategic_exists:
        st.success("✅ **Strategic Plan Ready**")
    else:
        st.warning("⏳ **Strategic Plan Pending**")

with col2:
    if action_exists:
        st.success("✅ **Action Plan Ready**")
    else:
        st.warning("⏳ **Action Plan Pending**")

with col3:
    both_ready = strategic_exists and action_exists
    if both_ready:
        st.success("✅ **Ready to Analyze**")
    else:
        st.info("⏳ **Upload Both Documents**")

# Document Management
if strategic_exists or action_exists:
    st.markdown("---")
    st.subheader("🗂️ Document Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if strategic_exists:
            st.markdown("#### Strategic Plan")
            with open(strategic_path, 'rb') as f:
                st.download_button(
                    label="📥 Download Strategic Plan",
                    data=f,
                    file_name=f"strategic_plan_{selected_year}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
            
            if st.button("🗑️ Delete Strategic Plan", use_container_width=True):
                strategic_path.unlink()
                st.success("Strategic Plan deleted!")
                st.rerun()
    
    with col2:
        if action_exists:
            st.markdown("#### Action Plan")
            with open(action_path, 'rb') as f:
                st.download_button(
                    label=f"📥 Download Action Plan {selected_year}",
                    data=f,
                    file_name=f"action_plan_{selected_year}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
            
            if st.button(f"🗑️ Delete Action Plan {selected_year}", use_container_width=True):
                action_path.unlink()
                st.success("Action Plan deleted!")
                st.rerun()

# Next Steps
if both_ready:
    st.markdown("---")
    st.success("### 🎯 Documents Ready for Analysis!")
    st.info("👉 Go to **'⚙️ Run Analysis'** page to process your documents")
    
    if st.button("▶️ Start Analysis Now →", type="primary", use_container_width=True):
        st.switch_page("pages/02_⚙️_Run_Analysis.py")

# All uploaded years summary
st.markdown("---")
st.subheader("📋 All Uploaded Years Summary")

all_years_data = []
for year in AVAILABLE_YEARS:
    y_path = UPLOAD_BASE / year
    s_exists = (y_path / "strategic_plan.docx").exists()
    a_exists = (y_path / "action_plan.docx").exists()
    
    if s_exists or a_exists:
        all_years_data.append({
            'Year': year,
            'Strategic Plan': '✅' if s_exists else '❌',
            'Action Plan': '✅' if a_exists else '❌',
            'Status': '✅ Ready' if (s_exists and a_exists) else '⏳ Partial'
        })

if all_years_data:
    import pandas as pd
    df = pd.DataFrame(all_years_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No documents uploaded yet. Start by selecting a year and uploading documents above.")

st.markdown("---")
st.caption("💡 **Tip:** You can manage different years independently. Strategic Plan is shared across years, while Action Plans are year-specific.")