---
title: PCTE Chatbot
emoji: 🏫
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

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