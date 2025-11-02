# CollegeBot (LangChain + Flask + Chroma)

A conventional college chatbot that answers **only** from your approved college data (PDFs, pasted text, and whitelisted webpages). If insufficient context is found, it replies: **"Sorry, I don't have enough information regarding this."**

## Quick start
```bash
python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env  # set your values
python app.py
# open http://localhost:5000
```

- LLM: OpenRouter (OpenAI-compatible). Put your OpenRouter key in `.env` as `OPENAI_API_KEY` and `OPENAI_API_BASE=https://openrouter.ai/api/v1`.
- Embeddings (local CPU): `sentence-transformers/all-MiniLM-L6-v2`.
- Vector store: Chroma (persisted in `./chroma_db`).

## Feeding data
- **Manual text**: paste policy/FAQ/schedule and click **Add Text**.
- **PDF/TXT**: upload documents.
- **Web URLs**: add pages on allowed domains (set in `.env` as `ALLOWED_DOMAINS`).

## Notes
- Retrieval uses a **similarity score threshold** (default 0.6) to avoid weak matches.
- Follow-ups work via a **question condensation** step.
- You can brand via `COLLEGE_NAME` in `.env`.
