# 🎯 Brandix ISPS - Intelligent Strategic Planning Synchronization System

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
source venv/bin/activate  # Windows: venv\Scripts\activate
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
