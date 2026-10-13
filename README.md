# Document Q&A System - Dev 1

## Overview
Dev 1 is responsible for the foundation of the team-based document Q&A system:
- **Document Ingestion (`ingest.py` / `dev1/ingest.py`)**: Parsing, loading, and chunking documents with permission tags.
- **Vector Indexing (`index.py` / `dev1/index.py`)**: Embedding chunks and persisting them in local Chroma vector storage.
- **Permission-Aware Retrieval (`retrieve.py` / `dev1/retrieve.py`)**: Querying vector indices with role-based and user-level access filtering.
- **Contracts (`contracts.py`)**: Shared dataclasses and protocol interfaces unifying Dev 1, Dev 2 (generation/LLM), and Dev 3 (API/UI).

---

## Prerequisites
- **Python**: 3.11 (`3.11.x`)
- **Package Manager**: `uv` or `pip`

---

## Setup Instructions

### 1. Create and Activate Virtual Environment
Using `uv`:
```bash
uv venv .venv --python 3.11
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

Or using standard `python`:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# Or using uv:
uv pip install -r requirements.txt
```

### 3. Environment Variables
Copy the template to `.env` (never commit real secrets):
```bash
cp .env.example .env
```
Adjust the parameters in `.env` as needed.

---

## Running Tests

Execute test suite via `pytest`:
```bash
pytest
```

To run with verbose output:
```bash
pytest -v
```

---

## Project Structure
```text
.
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules (venv, caches, Chroma, secrets)
├── contracts.py          # Shared data contracts and protocol definitions
├── dev1/                 # Dev 1 implementation package
│   ├── __init__.py       # Dev 1 package exports
│   ├── ingest.py         # Document ingestion & chunking scaffolding
│   ├── index.py          # Vector database indexing scaffolding
│   └── retrieve.py       # Permission-aware retrieval scaffolding
├── ingest.py             # Root entry point alias for dev1.ingest
├── index.py              # Root entry point alias for dev1.index
├── retrieve.py           # Root entry point alias for dev1.retrieve
├── requirements.txt      # Project dependencies
├── README.md             # Documentation and setup instructions
└── tests/                # Test suite
    ├── __init__.py
    ├── test_contracts.py # Contracts validation tests
    └── test_scaffolding.py # Scaffolding & import validation tests
```
