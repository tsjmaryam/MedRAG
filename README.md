# MedRAG: Dataset-Agnostic Medical RAG Pipeline for Clinical Diagnosis Support

MedRAG is a modular, config-driven Retrieval-Augmented Generation (RAG) pipeline for clinical diagnosis support, developed at Children's National Hospital. The system combines semantic patient retrieval via FAISS, knowledge graph enrichment, and LLM-generated structured diagnosis reports to assist clinicians with differential diagnosis.

---

## Key Features

**Dataset-agnostic architecture** - all dataset-specific settings including file paths, column names, diagnosis lists, and disease taxonomy mappings live in a single YAML config file. Adding a new dataset requires only a new config file and a preprocessor adapter, with zero changes to the core pipeline.

**Local embeddings** - uses `all-MiniLM-L6-v2` via sentence-transformers for semantic patient similarity search, eliminating any external API dependency for embeddings.

**Knowledge graph enrichment** - retrieves relevant disease context from a medical knowledge graph using NetworkX and cosine similarity, injecting structured clinical knowledge into the LLM prompt.

**FAISS semantic retrieval** - indexes all training patient records as dense vectors and retrieves the top-k most similar cases for each new patient query.

## Models

- **Embeddings:** all-MiniLM-L6-v2 (local, no API dependency)
- **LLM:** Qwen 2.5 3B running locally via Ollama (no API key required)
- The pipeline is model-agnostic and can be switched to any Ollama-compatible model via a one-line change in `main_MedRAG_v2.py`

**Evaluation pipeline** - compares generated diagnoses against ground truth labels and reports top-1 accuracy with mismatch analysis.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS (faiss-cpu) |
| Knowledge Graph | NetworkX, pandas, openpyxl |
| LLM (primary) | Qwen/Qwen2.5-72B-Instruct via HuggingFace |
| LLM (fallback) | mistralai/Mistral-Nemo-Instruct-2407 |
| Config Management | PyYAML |
| NLP | NLTK, scikit-learn |
| Data Processing | pandas, NumPy |
| Environment | conda, VS Code, PowerShell |

---

## Repository Layout
```
MedRAG/
│
├── config_ddxplus.yaml          # All dataset-specific settings: paths, columns, diagnoses, taxonomy
├── config_loader.py             # Loads YAML config into any pipeline component
├── preprocess_ddxplus_v2.py     # Converts raw DDXPlus CSV into pipeline-ready JSON and CSV
├── KG_Retrieve_v2.py            # Knowledge graph retrieval using sentence transformers
├── main_MedRAG_v2.py            # Core pipeline: FAISS search + KG enrichment + LLM generation
├── run_pipeline.py              # Single entry point to run the full pipeline
├── evaluate.py                  # Evaluates generated diagnoses against ground truth
└── README.md                    # Project documentation
```

---

## How to Run

**Prerequisites**
- Python 3.10+
- HuggingFace account with valid API token
- DDXPlus dataset files

**Installation**
```bash
pip install -r requirements.txt
```

**Set your HuggingFace token in authentication.py**
```python
hf_token = "your_hf_token_here"
```

**Preprocess the dataset**
```bash
python preprocess_ddxplus_v2.py
```

**Run the pipeline**
```bash
python run_pipeline.py
```

**Evaluate results**
```bash
python evaluate.py
```

---

## Adding a New Dataset

1. Create a new config file e.g. `config_newdataset.yaml` with correct column mappings, file paths, categories, and diagnoses list
2. Write a new preprocessor e.g. `preprocess_newdataset.py` that outputs the standard schema
3. Point `run_pipeline.py` to the new config with `cfg = get_config('config_newdataset.yaml')`

Core pipeline files remain completely unchanged.

---

## Results

- **80% top-1 diagnostic accuracy** on initial 10-patient DDXPlus validation with KG enrichment active
- Validated on DDXPlus dataset with 1M+ synthetic patient records across 49 pathologies
- Misclassifications occur between clinically similar conditions (e.g. atrial fibrillation vs. unstable angina)

---

## Models

- **Embeddings:** all-MiniLM-L6-v2 (local, no API dependency)
- **Primary LLM:** Qwen/Qwen2.5-72B-Instruct via HuggingFace Inference API
- **Fallback LLM:** mistralai/Mistral-Nemo-Instruct-2407
- The pipeline is model-agnostic and can be switched to any HuggingFace-compatible or OpenAI model via a one-line change in `main_MedRAG_v2.py`

---

## Dataset

DDXPlus: a large-scale medical diagnosis benchmark dataset with 1M+ synthetic patient records across 49 pathologies, including structured symptom evidence and differential diagnosis labels.

---

## Acknowledgments

This work was completed under the supervision of Dr. Syed Muhammad Anwar at Children's National Hospital. The pipeline was developed as part of an ongoing research effort to build responsible, interpretable AI systems for clinical decision support.

---
