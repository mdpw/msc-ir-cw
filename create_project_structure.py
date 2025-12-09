"""
Brandix ISPS Project Structure Generator
Run this script to automatically create all folders and placeholder files
"""

import os

def create_structure():
    """Create complete project structure"""
    
    # Define structure
    structure = {
        'brandix-isps': {
            'data': {
                'README.md': '# Data Directory\n\nPlace your document files here:\n- BRANDIX_STRATEGIC_PLAN_2025.docx\n- BRANDIX_ACTION_PLAN.docx'
            },
            'src': {
                '__init__.py': '# Brandix ISPS Source Code',
                'document_processor.py': '# TODO: Implement document processing',
                'embedding_engine.py': '# TODO: Implement embedding engine',
                'llm_engine.py': '# TODO: Implement LLM engine',
                'vector_store.py': '# TODO: Implement vector store',
                'synchronization_engine.py': '# TODO: Implement synchronization engine',
                'ontology.py': '# TODO: Define ontology',
                'ontology_mapper.py': '# TODO: Implement ontology mapper',
                'rag_pipeline.py': '# TODO: Implement RAG pipeline',
                'improvement_generator.py': '# TODO: Implement improvement generator',
                'isps_system.py': '# TODO: Main system controller'
            },
            'tests': {
                '__init__.py': '# Tests',
                'test_components.py': '# TODO: Unit tests',
                'evaluation.py': '# TODO: Accuracy evaluation'
            },
            'outputs': {
                'README.md': '# Outputs Directory\n\nGenerated analysis results will be saved here.'
            },
            'models': {
                'README.md': '# Models Directory\n\nFAISS vector store files will be saved here.'
            },
            '.streamlit': {
                'config.toml': '''[theme]
primaryColor = "#1f4788"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f5f7fa"
textColor = "#262730"

[server]
maxUploadSize = 200
'''
            },
            'app.py': '''"""
Brandix ISPS - Streamlit Dashboard
Main application file
"""

import streamlit as st

st.set_page_config(
    page_title="Brandix ISPS",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Brandix ISPS")
st.markdown("### Intelligent Strategic Planning Synchronization System")

st.info("👷 Under Development - Follow the 21-day implementation plan")

# TODO: Implement dashboard
''',
            'requirements.txt': '''sentence-transformers==2.2.2
faiss-cpu==1.7.4
PyPDF2==3.0.1
python-docx==1.1.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
langchain==0.1.0
langchain-community==0.0.10
streamlit==1.28.0
plotly==5.17.0
''',
            'config.py': '''"""
Configuration file for Brandix ISPS
"""

# File paths
STRATEGIC_PLAN_PATH = "data/BRANDIX_STRATEGIC_PLAN_2025.docx"
ACTION_PLAN_PATH = "data/BRANDIX_ACTION_PLAN.docx"

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1"
EMBEDDING_DIMENSION = 384

# Analysis settings
TOP_K_MATCHES = 5
ALIGNMENT_THRESHOLD_STRONG = 0.75
ALIGNMENT_THRESHOLD_MODERATE = 0.60
ALIGNMENT_THRESHOLD_WEAK = 0.40

# Output settings
OUTPUT_DIR = "outputs"
MODELS_DIR = "models"
''',
            'README.md': '''# 🎯 Brandix ISPS - Intelligent Strategic Planning Synchronization System

## Overview
AI-powered system for analyzing synchronization between Strategic Plans and Action Plans.

## Features
- Overall alignment analysis
- Strategy-wise synchronization
- Intelligent improvement suggestions
- Interactive dashboard
- Ontology-based mapping
- RAG-powered contextual analysis

## Technology Stack
- **Embeddings:** sentence-transformers
- **Vector DB:** FAISS
- **LLM:** Ollama (Llama 3.1)
- **RAG:** LangChain
- **Dashboard:** Streamlit
- **Visualization:** Plotly

## Setup

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Ollama
Download from: https://ollama.ai/download
```bash
ollama pull llama3.1
```

### 4. Add your documents
Place your documents in the `data/` folder:
- BRANDIX_STRATEGIC_PLAN_2025.docx
- BRANDIX_ACTION_PLAN.docx

## Usage

### Run Dashboard
```bash
streamlit run app.py
```

### Run Analysis
```bash
python src/isps_system.py
```

## Project Structure
```
brandix-isps/
├── data/               # Input documents
├── src/                # Source code
├── tests/              # Unit tests
├── outputs/            # Analysis results
├── models/             # Saved models
├── .streamlit/         # Streamlit config
├── app.py              # Dashboard
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Implementation Timeline
Follow the 21-day plan:
- **Phase 1 (Days 1-4):** Foundation & Setup
- **Phase 2 (Days 5-12):** Core System Development
- **Phase 3 (Days 13-21):** Dashboard, Deployment & Documentation

## Author
[Your Name]
MSc Computer Science - Information Retrieval
''',
            '.gitignore': '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# ISPS specific
data/*.docx
data/*.pdf
outputs/*.json
outputs/*.txt
models/*.faiss
models/*.meta

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml
'''
        }
    }
    
    def create_recursive(base_path, structure):
        """Recursively create folders and files"""
        for name, content in structure.items():
            path = os.path.join(base_path, name)
            
            if isinstance(content, dict):
                # It's a folder
                os.makedirs(path, exist_ok=True)
                print(f"📁 Created folder: {path}")
                create_recursive(path, content)
            else:
                # It's a file
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"📄 Created file: {path}")
    
    # Create structure
    print("🚀 Creating Brandix ISPS project structure...\n")
    create_recursive('.', structure)
    print("\n✅ Project structure created successfully!")
    print("\n📋 Next steps:")
    print("1. cd brandix-isps")
    print("2. python -m venv venv")
    print("3. source venv/bin/activate  (or venv\\Scripts\\activate on Windows)")
    print("4. pip install -r requirements.txt")
    print("5. Place your DOCX files in the data/ folder")
    print("6. Start with Day 1 implementation!")

if __name__ == "__main__":
    create_structure()