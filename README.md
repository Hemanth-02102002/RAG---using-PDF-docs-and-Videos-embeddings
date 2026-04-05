# 💬 Academic QA — Traditional RAG System

A **Retrieval-Augmented Generation (RAG)** application that lets you ask natural-language questions against your own documents (PDFs and YouTube videos) and receive concise, sourced answers — powered by **Groq's Llama 3.1**, **ChromaDB**, **LangChain**, and **LangGraph**, with a clean **Streamlit** chat interface.

---

## 📖 Project Description

Traditional RAG combines the strengths of **semantic search** and **large language models**. Instead of relying on the LLM's parametric memory alone, each user question is first used to retrieve the most relevant chunks from a local vector database, and only those chunks are injected as context into the LLM prompt. This keeps answers grounded, accurate, and traceable back to their source.

This project implements the classic RAG pipeline end-to-end:

- **Ingest** → load PDFs and YouTube video transcripts, chunk the text, embed it, and persist to ChromaDB.
- **Query** → embed the user question, perform similarity search, build a context-augmented prompt, and stream the response through Groq's fast inference API.
- **Cite** → display the source document name and exact page numbers alongside every answer.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📄 Multi-source ingestion | PDF files + YouTube videos (auto-transcribed via Whisper) |
| 🔍 Semantic similarity search | Top-3 most relevant chunks retrieved per query |
| 🧠 Stateful conversation | LangGraph `MemorySaver` persists the conversation within a session |
| ⚡ Fast LLM inference | Groq API with `llama-3.1-8b-instant` model |
| 🗂 Local vector store | ChromaDB persisted to disk — no external vector DB required |
| 📌 Source citations | Every answer shows the source file and reference page numbers |
| 🖥 Interactive UI | Streamlit chat interface with real-time streaming |

---

## 🛠 Tools & Technologies

### Core Framework
| Tool | Role |
|---|---|
| **Python 3.12+** | Runtime |
| **Streamlit** | Chat web UI |
| **LangChain** | Document loading, text splitting, vector store abstraction |
| **LangGraph** | Stateful conversational workflow with in-memory checkpointing |

### LLM & Embeddings
| Tool | Role |
|---|---|
| **Groq API** (`langchain-groq`) | Ultra-fast LLM inference |
| **llama-3.1-8b-instant** | The language model used for answer generation |
| **HuggingFace Sentence Transformers** | Local embedding model (`intfloat/multilingual-e5-large`) |

### Vector Database
| Tool | Role |
|---|---|
| **ChromaDB** | Local persistent vector store for document embeddings |

### Document Loaders & Parsing
| Tool | Role |
|---|---|
| **PyPDF** / `PyPDFLoader` | PDF ingestion and page-level loading |
| **yt-dlp** | YouTube audio download |
| **faster-whisper** | Local audio-to-text transcription |
| **pydub** | Audio file processing |

### Utilities
| Tool | Role |
|---|---|
| **uv** | Fast Python package manager and virtual environment tool |
| **python-dotenv** | Environment variable management (API keys) |
| **tiktoken** | Token counting utilities |
| **pandas** | Tabular processing of retrieved document chunks |

---

## 🏗 Workflow Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                    │
│                     (ingest.py)                          │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐                    │
│  │  PDF Files  │    │ YouTube URL  │                    │
│  │  (data/)    │    │              │                    │
│  └──────┬──────┘    └──────┬───────┘                    │
│         │                  │                            │
│  PyPDFLoader          YoutubeAudioLoader                │
│         │            + FasterWhisperParser              │
│         │                  │                            │
│         └────────┬─────────┘                            │
│                  ▼                                       │
│       RecursiveCharacterTextSplitter                    │
│       (chunk_size=2028, overlap=250)                    │
│                  │                                       │
│                  ▼                                       │
│     HuggingFaceEmbeddings                               │
│     (intfloat/multilingual-e5-large)                    │
│                  │                                       │
│                  ▼                                       │
│         ChromaDB (docs/chroma/)  ←── persisted to disk  │
└──────────────────────────────────────────────────────────┘

                          │
                          │  (run once, then reuse)
                          ▼

┌──────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                        │
│                      (app.py)                            │
│                                                          │
│  User Question                                           │
│       │                                                  │
│       ├──► Embed question (same HuggingFace model)       │
│       │                                                  │
│       ├──► ChromaDB similarity_search (top k=3 chunks)   │
│       │         │                                        │
│       │         ▼                                        │
│       │    Retrieved context chunks                      │
│       │         │                                        │
│       └─────────┴──► Build HumanMessage:                 │
│                      "Context: <chunks>\n\nQ: <prompt>"  │
│                               │                          │
│                               ▼                          │
│                    ┌─────────────────┐                   │
│                    │   LangGraph     │                   │
│                    │  StateGraph     │                   │
│                    │                 │                   │
│                    │ MemorySaver ──► │ Conversation      │
│                    │ (per thread_id) │ History           │
│                    └────────┬────────┘                   │
│                             │                            │
│                             ▼                            │
│                   Groq API (llama-3.1-8b-instant)        │
│                             │                            │
│                             ▼                            │
│                   AI Answer + Source Citation            │
│                   (source doc, page numbers)             │
│                             │                            │
│                             ▼                            │
│                    Streamlit Chat UI                     │
└──────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **Stateful memory via LangGraph**: `MemorySaver` with a fixed `thread_id` per Streamlit session means the model has access to the full conversation history, enabling multi-turn dialogue.
- **Context injected per turn**: Retrieved chunks are prepended to each user message — the LLM always has the freshest, most relevant document context.
- **Local embeddings**: Using `intfloat/multilingual-e5-large` locally avoids per-token embedding costs and supports multilingual documents.
- **Persistent ChromaDB**: The vector store is written to `docs/chroma/` once during ingestion and reloaded cheaply at app startup.

---

## 📁 Project Structure

```
Traditional_RAG/
├── app.py              # Streamlit chat app & LangGraph query pipeline
├── ingest.py           # Document ingestion pipeline (PDF + YouTube)
├── config.py           # Centralized configuration (paths, model names, chunk settings)
├── pyproject.toml      # Project metadata and dependencies
├── .env                # API keys (not committed to version control)
├── .gitignore
├── data/               # Place your PDF source documents here
│   └── Hands-On_LLM.pdf
└── docs/
    ├── chroma/         # Persisted ChromaDB vector store (auto-created)
    └── youtube/        # Temporary YouTube audio files (auto-created)
```

---

## 🚀 How to Run the Project

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager  
  Install with: `pip install uv` or `winget install --id=astral-sh.uv`
- A **[Groq API Key](https://console.groq.com/)** (free tier available)

---

### Step 1 — Clone the Repository

```bash
git clone <your-repo-url>
cd Traditional_RAG
```

---

### Step 2 — Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# .env
GROQ_API_KEY="your_groq_api_key_here"
```

---

### Step 3 — Install Dependencies

```bash
uv sync
```

This creates a virtual environment (`.venv/`) and installs all dependencies from `pyproject.toml`.

---

### Step 4 — Add Your Documents

Place your PDF files inside the `data/` directory:

```
data/
└── your_document.pdf
```

> Optionally, update `config.py` to point to a different YouTube video URL for transcription ingestion.

---

### Step 5 — Run the Ingestion Pipeline

> ⚠️ **Run this only once** (or whenever you update your documents). This downloads/transcribes YouTube audio, loads PDFs, embeds all content, and persists the ChromaDB vector store to `docs/chroma/`.

```bash
uv run python ingest.py
```

You should see progress logs like:
```
Starting PDF document loading from 'data'...
Loaded 412 pages from PDF documents.
Split documents into 891 chunks.
Initializing embeddings with model: intfloat/multilingual-e5-large
Creating and persisting Chroma DB to 'docs/chroma'...
Successfully processed 891 document chunks and persisted Chroma DB.
```

---

### Step 6 — Launch the Streamlit App

```bash
uv run streamlit run app.py
```

Open your browser at **http://localhost:8501** and start asking questions!

---

## ⚙️ Configuration

Edit `config.py` to customize the pipeline:

```python
class Config:
    YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=uFhDGagZzjs"
    YOUTUBE_AUDIO_SAVE_DIRECTORY = "docs/youtube/"

    PDF_SOURCE_DIRECTORY = "data"
    CHROMA_PERSIST_DIRECTORY = "docs/chroma"

    EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
    CHUNK_SIZE = 2028
    CHUNK_OVERLAP = 250
```

| Parameter | Description |
|---|---|
| `YOUTUBE_VIDEO_URL` | YouTube video to transcribe and ingest |
| `PDF_SOURCE_DIRECTORY` | Folder containing your PDF files |
| `CHROMA_PERSIST_DIRECTORY` | Where the vector DB is saved |
| `EMBEDDING_MODEL_NAME` | HuggingFace embedding model |
| `CHUNK_SIZE` | Max characters per document chunk |
| `CHUNK_OVERLAP` | Overlap between consecutive chunks |

---

## 📝 Example Usage

After launching the app, you can ask questions like:

- *"What is the difference between encoder and decoder transformers?"*
- *"Explain fine-tuning in the context of LLMs."*
- *"What are the main challenges in RAG systems?"*

The assistant will respond with a concise answer and cite the **source document** and **reference page numbers** so you can verify the information directly.

---

## 🔧 Troubleshooting

| Issue | Solution |
|---|---|
| `ChromaDB not found` error on startup | Run `python ingest.py` first to build the vector store |
| `GROQ_API_KEY` missing | Ensure `.env` file exists with a valid API key |
| YouTube download fails | Check your internet connection; ensure `yt-dlp` is up to date |
| Embedding model slow to load | First run downloads the model (~500MB); subsequent runs use the cache |
| No PDFs found | Place `.pdf` files in the `data/` directory |

---

## 📦 Dependencies Summary

```toml
chromadb>=1.0.12
faster-whisper>=1.1.1
groq>=0.26.0
langchain>=0.3.25
langchain-community>=0.3.24
langchain-groq>=0.3.2
langgraph>=0.4.7
pydub>=0.25.1
pypdf>=5.6.0
sentence-transformers>=4.1.0
streamlit>=1.45.1
tiktoken>=0.9.0
transformers>=4.52.4
yt-dlp>=2025.5.22
```

---

## 📄 License

This project is intended for educational and research purposes.
