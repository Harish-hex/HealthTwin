# 🏥 HealthTwin — RAG-Based AI Healthcare Assistant

HealthTwin is a production-ready **Retrieval-Augmented Generation (RAG)** system that answers medical questions by searching a knowledge base built from medical textbooks. It exposes a simple REST API and supports multiple LLM backends (Groq, OpenAI, or a fully open-source fallback).

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Knowledge Base](#-knowledge-base)
- [Setup & Installation](#-setup--installation)
- [Configuration](#-configuration)
- [Running the API Server](#-running-the-api-server)
- [API Reference](#-api-reference)
- [Building Your Own Knowledge Base](#-building-your-own-knowledge-base)
- [Deployment](#-deployment)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🔎 Overview

HealthTwin answers natural-language medical questions by:

1. **Embedding** the user's query with a sentence-transformer model.
2. **Searching** a pre-built FAISS vector index (35,000+ chunks from medical textbooks).
3. **Reranking** the top candidates with a cross-encoder for higher precision.
4. **Generating** a concise, context-grounded answer via Groq (Llama-3), OpenAI (GPT-4o-mini), or a local Flan-T5 fallback.

> ⚠️ **Disclaimer**: HealthTwin is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## 🏗 Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  FastAPI  (app.py)          │
│  POST /ask                  │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  SentenceTransformer        │  ← all-MiniLM-L6-v2 (384-dim)
│  Query Embedding            │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  FAISS Vector Search        │  ← IndexFlatL2 over 35,167 chunks
│  Top-N Candidate Retrieval  │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Cross-Encoder Reranking    │  ← ms-marco-MiniLM-L-6-v2
│  Re-scores & Selects Top-K  │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  LLM Answer Generation      │  ← Groq / OpenAI / Flan-T5
│  Context-grounded response  │
└──────────┬──────────────────┘
           │
           ▼
    JSON Response
    { "answer": "...", "contexts": [...] }
```

### Two-Stage Retrieval

| Stage | Component | Purpose |
|-------|-----------|---------|
| **Stage 1** | FAISS (L2) | Fast broad retrieval of `2 × top_k` candidates |
| **Stage 2** | CrossEncoder | Accurate re-scoring → selects final `top_k` results |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **API Framework** | FastAPI + Uvicorn |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Vector Store** | FAISS (`faiss-cpu`) |
| **LLM (primary)** | Groq — `llama-3.3-70b-versatile` (free & fast) |
| **LLM (secondary)** | OpenAI — `gpt-4o-mini` |
| **LLM (fallback)** | HuggingFace — `google/flan-t5-base` (no API key needed) |
| **PDF Processing** | PyMuPDF (`fitz`) + multiprocessing |
| **Language** | Python 3.11 |
| **Deployment** | Railway / Render / Docker |

---

## 📁 Project Structure

```
HealthTwin-RAG-Based-AI-Healthcare-Assistant/
├── app.py                          # FastAPI server (main entry point)
├── medical_pdf_pipeline_fast.py    # PDF ingestion & FAISS index builder
├── requirements.txt                # Python dependencies
├── Procfile                        # Railway / Heroku process definition
├── runtime.txt                     # Python version pin (3.11.7)
├── env.example                     # Environment variable template
├── outputs/
│   ├── medical_faiss.index         # Pre-built FAISS vector index
│   ├── medical_metadata.json       # Chunk metadata (book, page, text)
│   └── processing_summary.csv      # Per-book processing statistics
├── API_README.md                   # Detailed API documentation
└── QUICK_START.md                  # One-page quick-start reference
```

---

## 📚 Knowledge Base

The pre-built index covers **9 medical textbooks** totalling ~1.6 GB of PDFs:

| Subject | Pages |
|---------|------:|
| Anatomy & Physiology | 1,300 |
| Cardiology | 2,034 |
| Dentistry | 710 |
| Emergency Medicine | 2,727 |
| Gastroenterology | 2,724 |
| General Medicine | 1,428 |
| Infectious Disease | 622 |
| Internal Medicine | 4,171 |
| Nephrology | 2,947 |
| **Total** | **18,663 pages → 35,167 chunks** |

Default chunk settings: `CHUNK_SIZE = 800 words`, `CHUNK_OVERLAP = 200 words`. These can be adjusted when [building your own knowledge base](#-building-your-own-knowledge-base).

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.11
- `pip` (or a virtual-environment manager such as `venv` / `conda`)
- *(Optional)* A [Groq](https://console.groq.com/) or [OpenAI](https://platform.openai.com/api-keys) API key for best answer quality

### 1. Clone the repository

```bash
git clone https://github.com/Harish-hex/HealthTwin-RAG-Based-AI-Healthcare-Assistant.git
cd HealthTwin-RAG-Based-AI-Healthcare-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp env.example .env
# Edit .env and fill in your keys (see Configuration section below)
```

---

## 🔧 Configuration

Copy `env.example` to `.env` and set the variables you need:

```dotenv
# Groq API key (recommended — free, fast, uses Llama-3)
# Get yours at https://console.groq.com/
GROQ_API_KEY=your-groq-api-key-here

# OpenAI API key (used only if GROQ_API_KEY is not set)
# Get yours at https://platform.openai.com/api-keys
OPENAI_API_KEY=your-openai-api-key-here

# Server port (Railway sets this automatically; defaults to 8000 locally)
PORT=8000
```

**LLM selection priority:**
1. Groq (if `GROQ_API_KEY` is set and valid)
2. OpenAI (if `OPENAI_API_KEY` is set and valid)
3. Local `google/flan-t5-base` (no API key required — slower, lower quality)

---

## 🚀 Running the API Server

```bash
python app.py
```

The server starts on `http://localhost:8000` (or the port specified in `.env`).

Alternatively, using Uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 API Reference

### `GET /`

Basic health check and endpoint listing.

**Response**
```json
{
  "status": "healthy",
  "service": "Medical RAG Chatbot API",
  "version": "1.0.0",
  "endpoints": {
    "ask": "POST /ask - Ask a medical question"
  }
}
```

---

### `GET /health`

Detailed component health status.

**Response**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "faiss_loaded": true,
  "metadata_loaded": true,
  "llm_type": "openai"
}
```

---

### `POST /ask` ⭐

Ask a medical question. Also available at `POST /query`.

**Request body**
```json
{
  "query": "What are the contraindications of aspirin?",
  "top_k": 3
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | ✅ | — | The medical question |
| `top_k` | integer | ❌ | `3` | Number of context snippets to retrieve (1–10) |

**Response**
```json
{
  "answer": "Aspirin is contraindicated in patients with active peptic ulcer disease, bleeding disorders, severe liver disease, and in children with viral infections due to Reye's syndrome risk.",
  "contexts": [
    "Context snippet 1 from medical textbook...",
    "Context snippet 2 about contraindications...",
    "Context snippet 3 with clinical guidelines..."
  ]
}
```

**Status codes:** `200` is always returned by design — even on errors. Clients must inspect the `answer` field for error messages (e.g. `"I encountered an error…"`). This behaviour matches the existing application contract and ensures downstream consumers always receive valid JSON.

---

### Example cURL calls

```bash
# Health check
curl http://localhost:8000/health

# Cardiology
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the signs of acute coronary syndrome?", "top_k": 5}'

# Pharmacology
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "When should beta-blockers be avoided?", "top_k": 3}'

# Emergency Medicine
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How do you manage anaphylactic shock?", "top_k": 4}'
```

---

## 🔨 Building Your Own Knowledge Base

Use `medical_pdf_pipeline_fast.py` to index your own PDFs.

### 1. Add PDFs

Place all PDF files in the `medical_books/` directory (create it if it doesn't exist):

```bash
mkdir medical_books
cp /path/to/your/*.pdf medical_books/
```

### 2. (Optional) Tune pipeline settings

At the top of `medical_pdf_pipeline_fast.py`:

```python
CHUNK_SIZE        = 800    # words per chunk
CHUNK_OVERLAP     = 200    # word overlap between chunks
MAX_PDFS          = None   # set to an integer to limit the number of PDFs
MAX_PAGES_PER_PDF = None   # set to an integer to limit pages per PDF
PARALLEL_WORKERS  = 4      # parallel PDF processing workers
SKIP_EMBEDDINGS   = False  # set True to extract text only (no FAISS index)
```

### 3. Run the pipeline

```bash
python medical_pdf_pipeline_fast.py
```

The script will:
1. Extract text from every PDF using PyMuPDF (parallel workers).
2. Split the text into overlapping chunks.
3. Generate 384-dim embeddings with `all-MiniLM-L6-v2`.
4. Build and save a FAISS index to `outputs/medical_faiss.index`.
5. Save chunk metadata to `outputs/medical_metadata.json`.
6. Write a per-book summary to `outputs/processing_summary.csv`.

### 4. Test the index

```bash
python -c "
from medical_pdf_pipeline_fast import search
search('What are the contraindications of penicillin in pregnancy?')
"
```

---

## 🚢 Deployment

### Railway (recommended)

```bash
npm i -g @railway/cli
railway login
railway init
railway variables set GROQ_API_KEY=your-key-here
railway up
```

Your API will be live at `https://your-app.up.railway.app`.

### Render

1. Connect your GitHub repository.
2. Select **Web Service**.
3. **Build command**: `pip install -r requirements.txt`
4. **Start command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `GROQ_API_KEY` (or `OPENAI_API_KEY`).

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t healthtwin .
docker run -p 8000:8000 -e GROQ_API_KEY=your-key healthtwin
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Average response time (Groq) | ~2–5 s |
| Average response time (OpenAI) | ~2–5 s |
| Average response time (fallback) | ~10–20 s |
| Max guaranteed response time | < 60 s |
| Vector search latency | < 100 ms |
| Context retrieval (FAISS + reranking) | < 200 ms |

Recommended server: **2 GB RAM** minimum (models are loaded in memory at startup).

---

## 🔍 Troubleshooting

| Problem | Solution |
|---------|----------|
| `FAISS index not found` | Ensure `outputs/medical_faiss.index` exists. Run the PDF pipeline to generate it. |
| `Module not found` | Run `pip install -r requirements.txt` inside your virtual environment. |
| Slow responses | Lower `top_k` to 3. Use Groq instead of the fallback model. |
| OpenAI / Groq API error | The server auto-falls back to `flan-t5-base`. Check your API key in `.env`. |
| Out of memory | Deploy on a host with ≥ 2 GB RAM. Ensure only one worker process runs. |
| `No PDF files found` | Place `.pdf` files in `medical_books/` before running the pipeline. |

---

## 🔐 Security Notes

- API keys are loaded exclusively from environment variables (never hard-coded).
- CORS is currently set to `allow_origins=["*"]`; restrict this in production by replacing the wildcard with your actual frontend domain in `app.py`:
  ```python
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["https://your-frontend-domain.com"],
      ...
  )
  ```
- All user inputs are validated before processing.
- No sensitive data is written to logs.

---

## 📄 License

This project is for **educational and research purposes only**. It is not intended for clinical use and does not constitute medical advice.

---

*Built with ❤️ for medical education and research.*
