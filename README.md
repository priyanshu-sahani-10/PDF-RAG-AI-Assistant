# PDF RAG Chatbot

Upload PDFs and ask questions about them. Answers are generated locally by **Gemma 2B (Ollama)** from passages retrieved from your documents, with page-level source references.

- **Frontend:** React + Vite + Tailwind CSS (light theme)
- **Backend:** FastAPI, LangChain (text splitting), PyPDF, HuggingFace `all-MiniLM-L6-v2` embeddings, ChromaDB, SQLite
- **LLM:** `gemma:2b` through a local Ollama server. No cloud APIs are used.

```
pdf-rag-chatbot/
├── frontend/   React app (http://localhost:5173)
└── backend/    FastAPI app (http://localhost:8000)
    ├── app/    api/, services/, schemas/, core/, database/
    ├── data/   SQLite + ChromaDB (created at runtime, git-ignored)
    ├── uploads/  stored PDFs (git-ignored)
    └── tests/
```

## 1. Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11 or 3.12 (developed on 3.12; 3.13 may lack wheels for some dependencies) |
| Node.js | 18 or newer (built with 22) |
| Ollama | latest, from https://ollama.com |
| Disk / network | ~2 GB for Python packages (PyTorch), ~1.7 GB for `gemma:2b`, ~90 MB for the embedding model. Internet is needed for these first downloads only. |

## 2. Install Ollama and the Gemma 2B model

**Windows (PowerShell)**
```powershell
winget install Ollama.Ollama        # or download the installer from https://ollama.com/download
# Open a NEW terminal afterwards so `ollama` is on PATH
ollama pull gemma:2b
ollama list                          # gemma:2b should be listed
ollama run gemma:2b "Reply with the single word: ready"
Invoke-RestMethod http://localhost:11434/api/tags   # confirms the server is reachable
```

**macOS / Linux**
```bash
# macOS: brew install ollama   |   Linux: curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma:2b
ollama list
ollama run gemma:2b "Reply with the single word: ready"
curl http://localhost:11434/api/tags
```

On Windows and macOS the Ollama desktop app starts the server automatically. On Linux (or if the check above fails) run `ollama serve` in a separate terminal and leave it running.

## 3. Backend

**Windows (PowerShell)**
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip

# Optional but recommended on machines without an NVIDIA GPU: a much smaller CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
Copy-Item .env.example .env          # edit if you need different settings
python -m uvicorn app.main:create_app --factory --reload --port 8000
```

**Linux / macOS**
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional, Linux CPU-only
pip install -r requirements.txt
cp .env.example .env
python -m uvicorn app.main:create_app --factory --reload --port 8000
```

On first start the backend downloads the embedding model in the background (watch the log for `Embedding model ready`). Uploads work during the download and finish once it completes.

## 4. Frontend

In a second terminal:

**Windows (PowerShell)**
```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

**Linux / macOS**
```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

## 5. URLs

| What | URL |
|------|-----|
| App | http://localhost:5173 |
| API | http://localhost:8000/api |
| Swagger docs | http://localhost:8000/docs |
| Health check | http://localhost:8000/api/health |

## 6. Configuration

Backend (`backend/.env`, see `.env.example`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `gemma:2b` | Model used for answers |
| `OLLAMA_TIMEOUT_SECONDS` | `180` | Max wait for one answer |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model. **Changing it requires re-uploading your PDFs** |
| `DATA_DIR`, `UPLOAD_DIR` | `data`, `uploads` | Storage (relative to `backend/`) |
| `CHROMA_DIR`, `SQLITE_PATH` | under `DATA_DIR` | Optional overrides |
| `MAX_UPLOAD_MB` | `25` | Upload size limit |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | `1000`, `150` | Text splitting (characters) |
| `TOP_K` | `4` | Chunks retrieved per question |
| `RELEVANCE_THRESHOLD` | `0.25` | Minimum relevance to use a chunk |
| `CORS_ORIGINS` | Vite dev URLs | Comma-separated allowed origins |

Frontend (`frontend/.env`): `VITE_API_BASE_URL=http://localhost:8000`

**Relevance score.** The Chroma collection uses cosine distance (`0` = identical). The backend reports `relevance = 1 - distance`, clamped to 0..1. Chunks below `RELEVANCE_THRESHOLD` are discarded. If none remain, the app replies that nothing was found and does **not** call the LLM. If you get too many "couldn't find" replies, lower the threshold (e.g. `0.15`); if answers cite loosely related text, raise it.

## 7. How it works

1. **Upload:** the file is validated (extension, MIME type, `%PDF-` header, size), streamed to disk, and hashed. The first 16 hex characters of the SHA-256 become the stable document ID, so re-uploading the same file is rejected as a duplicate (`409`).
2. **Index (background):** PyPDF extracts text per page, `RecursiveCharacterTextSplitter` makes ~1000-character chunks, each is embedded and stored in ChromaDB with `document_id`, `filename`, `page_number`, `chunk_index`. If anything fails, vectors and the stored file are removed and the document is marked **Failed** with the reason.
3. **Ask:** the question is embedded with the same model, ChromaDB searches with a `document_id` filter applied inside the vector query, weak matches are dropped, and the remaining chunks go into the prompt for Gemma.
4. **Cite:** sources are built from the retrieved chunk metadata, never from text written by the model.

The system prompt tells the model to stay within the context, never invent facts or citations, say when information is missing, treat text inside PDFs as untrusted data, and separate stated facts from inference.

Writes to ChromaDB and SQLite are serialized with a lock. Documents stuck "processing" when the server stops are marked Failed on the next start.

## 8. API reference

Errors always look like `{"error": {"code": "...", "message": "..."}}`.

| Method & path | Request | Response |
|---------------|---------|----------|
| `GET /api/health` | | `{status, backend, ollama:{available, model, model_available, message}, embedding_model, max_upload_mb, top_k, relevance_threshold}` |
| `POST /api/documents/upload` | multipart field `file` | `202` document (see below). Status starts as `processing` |
| `GET /api/documents` | | `{documents: [document]}` |
| `GET /api/documents/{id}` | | document |
| `DELETE /api/documents/{id}` | | `{deleted: true, id}` |
| `POST /api/chat` | `{question, document_ids?: [id], session_id?}` | `{answer, sources, session_id, grounded}` |
| `GET /api/chat/sessions` | | `{sessions: [{id, title, created_at, updated_at}]}` |
| `POST /api/chat/sessions` | `{title?}` | `201` session |
| `GET /api/chat/sessions/{id}` | | session + `messages: [{id, role, content, sources, created_at}]` |
| `DELETE /api/chat/sessions/{id}` | | `{deleted: true, id}` |

`document`: `{id, filename, status: "processing"|"ready"|"failed", page_count, chunk_count, size_bytes, error, created_at}`

`source`: `{document_id, filename, page_number, chunk_index, excerpt, text, relevance}`

Common error codes: `invalid_file_type`, `file_too_large` (413), `duplicate_document` (409), `no_documents`, `documents_not_ready`, `ollama_unavailable` (503), `model_missing` (503), `ollama_timeout` (504), `generation_failed` (502), `embedding_model_unavailable` (503).

## 9. Tests

```bash
cd backend
pip install -r requirements-dev.txt     # adds pytest and reportlab
python -m pytest -q
```

The tests use a small deterministic fake embedder and a fake Ollama client, so they run offline in seconds. They cover PDF extraction and failures, chunk metadata, cosine relevance and filtering, upload/duplicate/size/type validation, chat with citations and scope, the no-relevant-content path, sessions, deletion, restart persistence, the Ollama HTTP client, and CORS.

Frontend production build: `cd frontend && npm run build`.

## 10. Troubleshooting

| Problem | Fix |
|---------|-----|
| Sidebar shows **Backend offline** | Backend isn't running or `VITE_API_BASE_URL` is wrong. Restart `npm run dev` after editing `.env`. |
| **Ollama not running** | Start the Ollama app, or run `ollama serve`. Check `OLLAMA_BASE_URL`. |
| **Model not downloaded** | `ollama pull gemma:2b` (must match `OLLAMA_MODEL`). |
| First answer is slow | Gemma loads into memory on first use. Raise `OLLAMA_TIMEOUT_SECONDS` on slow machines. |
| `pip install` is huge or fails on PyTorch | Install CPU-only torch first (step 3), then `pip install -r requirements.txt`. Use Python 3.11/3.12. |
| `Could not load the embedding model` | The first run needs internet to fetch it from HuggingFace. Behind a proxy set `HTTPS_PROXY`. The error message includes the underlying cause. |
| `Address already in use` | Use another port: `--port 8001`, then set `VITE_API_BASE_URL=http://localhost:8001`. Vite will pick the next free port automatically; add that origin to `CORS_ORIGINS` if it isn't 5173. |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| PDF marked Failed: "scanned or image-only" | OCR is not supported in this version. Use a PDF with selectable text. |
| PDF marked Failed: "password protected" | Remove the password and upload again. |
| Answers say "couldn't find" too often | Lower `RELEVANCE_THRESHOLD`, or raise `TOP_K`. |
| Reset everything | Stop the backend, delete `backend/data/*` and `backend/uploads/*` (keep `.gitkeep`). |

## Limitations

- No OCR for scanned PDFs.
- Answers are returned in one piece (no streaming).
- Gemma 2B is a small model: keep questions specific, and check the sources for anything important.
- Single-user, local use only: there is no authentication.
