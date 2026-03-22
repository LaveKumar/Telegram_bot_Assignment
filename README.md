# 🖼 Vision Captioning Telegram Bot

A lightweight GenAI Telegram bot (Option B) that accepts image uploads and returns a **natural-language caption + 3 keyword tags**, powered entirely by **open-source, locally-running models**.

---

## 🚀 Tech Stack (as specified)

| Layer | Tool | Detail |
|---|---|---|
| **Bot interface** | `python-telegram-bot` v21 | Async, long-polling |
| **Vision LLM** | `ollama` → **LLaVA** | Runs locally, no GPU required for 7B |
| **Embeddings** | `sentence-transformers` → `all-MiniLM-L6-v2` | 22 MB, CPU-only |
| **Vector storage** | `sqlite-vec` + plain SQLite | Zero-dependency vector DB |
| **Debug UI** | `Gradio` | Live at `localhost:7860` |

---

## ⚙️ How to Run Locally

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/download) installed and running
- A Telegram Bot Token from [@BotFather](https://t.me/botfather)

### Step 1 — Pull the vision model

```bash
ollama pull llava          # 4 GB, one-time download
# or for better quality:
ollama pull llava:13b
```

### Step 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Set environment variable

```bash
export TELEGRAM_BOT_TOKEN="your-token-here"
```

### Step 4 — Run the bot

```bash
python bot.py
```

### Step 5 (optional) — Run the Gradio debug UI

```bash
python gradio_ui.py
# → Open http://localhost:7860
```

---

## 🐳 Docker Compose (all-in-one)

```bash
# Create .env
echo "TELEGRAM_BOT_TOKEN=your-token-here" > .env

# Start Ollama + Bot + Gradio UI
docker compose up --build
```

Services started:
| Container | Port | Purpose |
|---|---|---|
| `ollama` | 11434 | LLaVA model server |
| `vision-bot` | — | Telegram polling bot |
| `gradio-ui` | 7860 | Debug / test UI |

---

## 🏗 System Design

```
User (Telegram)
      │  sends photo
      ▼
python-telegram-bot  ──► download image bytes
      │
      ├─► SHA-256 hash ──► SQLite exact cache ──► HIT: return instantly
      │
      ├─► sentence-transformers (all-MiniLM-L6-v2)
      │         │  embed last caption
      │         ▼
      │    sqlite-vec cosine search ──► semantic HIT (≥0.92): return
      │
      └─► ollama REST API (localhost:11434)
                │  base64 image + system prompt → LLaVA
                ▼
           { "caption": "...", "tags": [...] }
                │
                ├─► store in SQLite (hash + vector)
                └─► reply to user via Telegram

Gradio UI (localhost:7860)
  ├── Tab 1: Upload image → same pipeline (no Telegram needed)
  ├── Tab 2: Browse cache (10 most recent captions)
  └── Tab 3: Semantic similarity tester (compare two captions)
```

---

## 🗂 File Structure

```
vision_bot/
├── bot.py          # Telegram bot — handlers, history, formatting
├── vision.py       # Ollama / LLaVA integration
├── cache.py        # Two-layer cache: SHA-256 + sqlite-vec embeddings
├── gradio_ui.py    # Gradio debug UI (3 tabs)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🧠 Model Rationale

| Choice | Reason |
|---|---|
| **LLaVA via Ollama** | Open-source, runs locally (no API key), supports vision natively, easy `ollama pull` setup |
| **all-MiniLM-L6-v2** | 22 MB, CPU-only, 384-dim embeddings — perfect balance of speed and quality for semantic caching |
| **sqlite-vec** | No separate vector DB process; embeddings live in the same SQLite file as the rest of the cache |
| **Gradio** | Zero-config local debug UI; test the full pipeline without a Telegram account |

---

## ✅ Optional Enhancements Implemented

| Feature | Where |
|---|---|
| Message history (last 3 turns per user) | `bot.py` → `user_history` dict |
| Exact image caching (SHA-256) | `cache.py` → `get_by_hash()` |
| Semantic caption caching (sqlite-vec) | `cache.py` → `get_by_embedding()` |
| `/summarize` command | `bot.py` → `cmd_summarize()` |
| Gradio local debug UI | `gradio_ui.py` |

---


## Sample Output: 
https://shorturl.at/NfXij
https://shorturl.at/Zld9b
https://shorturl.at/OjAN4
