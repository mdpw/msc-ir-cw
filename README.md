# Brandix ISPS - Intelligent Strategic Planning Synchronization System

An AI-powered system that analyzes the synchronization between corporate strategic plans and operational action plans using NLP, semantic embeddings, and retrieval-augmented generation (RAG).

Built as a coursework project for **MSc Computer Science - Information Retrieval** module.

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://brandix-isps.streamlit.app/)

## Problem Statement

Organizations often struggle to ensure their annual action plans are properly aligned with long-term strategic objectives. Manual evaluation is time-consuming and subjective. This system automates the synchronization analysis using information retrieval techniques, providing quantitative alignment scores, gap identification, and actionable improvement suggestions.

## System Architecture

```
                    +-----------------------+
                    |   Document Upload     |
                    |  (.docx Strategic &   |
                    |   Action Plans)       |
                    +----------+------------+
                               |
                    +----------v------------+
                    |  Document Processor   |
                    |  (Parsing, Cleaning,  |
                    |   Objective/Action    |
                    |   Extraction)         |
                    +----------+------------+
                               |
                +-------- -----v-----------+
                |  Embedding Engine         |
                |  (all-MiniLM-L6-v2)      |
                |  384-dim Sentence Vectors |
                +----------+---------------+
                           |
              +------------v-------------+
              |     FAISS Vector Store    |
              |  (Similarity Search &    |
              |   Cosine Distance)       |
              +------------+-------------+
                           |
            +--------------v---------------+
            |   Synchronization Engine     |
            |  (Alignment Scoring,         |
            |   Classification,            |
            |   Pillar-level Analysis)     |
            +--------------+---------------+
                           |
          +-------+--------+--------+-------+
          |       |                 |       |
    +-----v---+  +v---------+  +---v-----+ |
    |Knowledge|  |Executive  |  |   RAG   | |
    |  Graph  |  | Summary   |  |Pipeline | |
    |(NetworkX)|  |(LLM)     |  |(Context)| |
    +---------+  +----------+  +---+-----+ |
                                   |       |
                            +------v-------v--+
                            |   LLM Engine     |
                            |  (Ollama/Phi-3)  |
                            |  Improvement     |
                            |  Suggestions     |
                            +---------+--------+
                                      |
                            +---------v--------+
                            | Streamlit Dashboard|
                            | (7-Page Multi-Page |
                            |  Application)      |
                            +--------------------+
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Semantic Alignment Analysis** | Cosine similarity scoring between strategic objectives and action items using sentence-transformers |
| **Multi-level Classification** | Strong (>0.70), Moderate (0.50-0.70), and Weak (<0.50) alignment categories |
| **Pillar-level Breakdown** | Per-pillar analysis across all strategic pillars (e.g., Environmental Leadership, Operational Excellence) |
| **Knowledge Graph** | NetworkX-based graph visualization showing objective-action relationships |
| **RAG-powered Suggestions** | Context-aware improvement recommendations using retrieved document segments |
| **Executive Summary** | LLM-generated executive report with key findings, gaps, risks, and next steps |
| **Multi-year Comparison** | Track synchronization changes across planning years |
| **Automated Testing** | Comprehensive evaluation framework with ground truth comparison |

## Technology Stack

| Component | Technology |
|-----------|------------|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2, 384-dim) |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| LLM | Ollama (Phi-3 Mini) |
| RAG Framework | Custom pipeline with contextual retrieval |
| Frontend | Streamlit (multi-page application) |
| Visualization | Plotly (interactive charts) |
| Graph Analysis | NetworkX |
| Document Processing | python-docx |

## Project Structure

```
msc-ir-cw/
├── brandix-isps/                  # Main application
│   ├── Home.py                    # Landing page & system overview
│   ├── config.py                  # Configuration (thresholds, model settings)
│   ├── requirements.txt           # Python dependencies
│   ├── run_tests.py               # Test runner script
│   │
│   ├── pages/                     # Streamlit multi-page app
│   │   ├── 01_Admin_Upload.py     # Document upload & management
│   │   ├── 02_Run_Analysis.py     # Core analysis & AI suggestions pipeline
│   │   ├── 03_View_Results.py     # Interactive results dashboard
│   │   ├── 04_Executive_Summary.py# LLM-generated executive report
│   │   ├── 05_Multi_Year_Comparison.py # Year-over-year tracking
│   │   ├── Knowledge_Graph.py     # Graph visualization
│   │   └── Testing_Evaluation.py  # Automated test results & metrics
│   │
│   ├── src/                       # Core system modules
│   │   ├── document_processor.py  # .docx parsing & objective extraction
│   │   ├── embedding_engine.py    # Sentence-transformer embeddings
│   │   ├── vector_store.py        # FAISS index management
│   │   ├── synchronization_engine.py # Alignment scoring & classification
│   │   ├── knowledge_graph.py     # NetworkX graph construction
│   │   ├── llm_engine.py          # Ollama LLM integration
│   │   ├── rag_pipeline.py        # RAG contextual retrieval
│   │   ├── executive_summary.py   # Summary generation
│   │   └── testing_framework.py   # Evaluation framework
│   │
│   ├── data/                      # Input documents & ground truth
│   │   ├── uploaded/              # User-uploaded plan documents
│   │   └── ground_truth/          # Manual annotations for evaluation
│   │
│   ├── outputs/                   # Analysis results (per year)
│   │   └── 2026/
│   │       ├── synchronization_report.json
│   │       ├── executive_summary.json
│   │       ├── test_results.json
│   │       └── ...
│   │
│   ├── models/                    # Cached embedding models
│   ├── tests/                     # Unit & component tests
│   └── .streamlit/config.toml     # Theme & server configuration
│
├── docs/                          # Coursework report & presentation
└── .devcontainer/                 # GitHub Codespaces configuration
```

## Setup & Installation

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai/download) (for LLM features)

### 1. Clone the repository
```bash
git clone https://github.com/mdpw/msc-ir-cw.git
cd msc-ir-cw/brandix-isps
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull the LLM model (for AI suggestions)
```bash
ollama pull phi3:mini
```

### 5. Run the application
```bash
streamlit run Home.py
```

The dashboard will open at `http://localhost:8501`.

## Usage Workflow

1. **Upload Documents** - Upload your Strategic Plan and Action Plan (.docx) via the Admin Upload page
2. **Run Core Analysis** - Execute the synchronization analysis pipeline (Stages 1-4, ~4 seconds)
3. **Generate AI Suggestions** *(optional)* - Run LLM-powered improvement suggestions (Stages 5-6, requires Ollama)
4. **View Results** - Explore alignment scores, pillar breakdowns, and similarity matrices
5. **Executive Summary** - Review the AI-generated executive report
6. **Knowledge Graph** - Visualize objective-action relationships
7. **Testing Evaluation** - View automated test results against ground truth annotations

## Analysis Pipeline

The system runs a 6-stage pipeline:

| Stage | Name | Description |
|-------|------|-------------|
| 1 | Document Processing | Parse .docx files, extract objectives and action items |
| 2 | Embedding Generation | Generate 384-dim sentence embeddings using all-MiniLM-L6-v2 |
| 3 | Similarity Analysis | Build FAISS index, compute cosine similarity matrix |
| 4 | Synchronization Scoring | Classify alignments, calculate pillar scores, generate report |
| 5 | AI Improvement Suggestions | RAG-powered contextual suggestions via Ollama LLM |
| 6 | Executive Summary | LLM-generated comprehensive report with recommendations |

> Stages 1-4 run as **Core Analysis** (~4 seconds). Stages 5-6 run separately as **AI Suggestions** (requires Ollama, ~20-30 minutes).

## Evaluation Results

Based on the latest analysis run against ground truth annotations:

| Metric | Score |
|--------|-------|
| Top-K Classification Accuracy | 70.0% |
| Comprehensive Classification Accuracy | 46.7% |
| Similarity Score Correlation | 0.492 |
| Predictions within 10% of ground truth | 50.0% |
| Predictions within 20% of ground truth | 80.0% |
| Overall Grade | **B+ (Very Good)** |

## Configuration

Key parameters in [config.py](brandix-isps/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence-transformer model |
| `LLM_MODEL` | phi3:mini | Ollama LLM for suggestions |
| `TOP_K_MATCHES` | 5 | Number of top matches per objective |
| `ALIGNMENT_THRESHOLD_STRONG` | 0.70 | Strong alignment threshold |
| `ALIGNMENT_THRESHOLD_MODERATE` | 0.50 | Moderate alignment threshold |
| `ALIGNMENT_THRESHOLD_WEAK` | 0.30 | Weak alignment threshold |

## License

This project was developed for academic purposes as part of the MSc Computer Science programme.
