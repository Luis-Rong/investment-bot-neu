# 🤖 Agentic RAG Robo-Advisor

A chat-based investment advisor that turns a short conversation into a personalized,
**evidence-grounded** portfolio recommendation. It combines an LLM front-end with a
deterministic finance core, **live market data**, and **retrieval-augmented generation**
over real fund factsheets.

> Originally built as a bachelor-thesis prototype, now extended into a full
> **agentic RAG** application that applies the architecture from the
> [IBM RAG and Agentic AI Professional Certificate](https://www.coursera.org/professional-certificates/ibm-rag-and-agentic-ai)
> to a real fintech use case.

---

## ✨ What it does

- **Conversational profiling** — the bot asks one question at a time to learn your goal,
  time horizon, risk tolerance, ESG preference, and how much you want to invest.
- **Real risk metrics** — instead of hard-coded numbers, it computes volatility, max
  drawdown, and Sharpe ratio from live historical prices (via `yfinance`).
- **Grounded explanations** — recommendations cite real fund factsheets retrieved from a
  vector store, so the advice is traceable, not hallucinated.
- **Agentic reasoning** — a LangGraph state machine routes each query, fetches data or
  retrieves documents as needed, and runs a self-reflection loop before answering.
- **Interactive** — adjust the recommendation in natural language
  (*"make it slightly less risky"*).

## 🏗️ Architecture

```mermaid
flowchart LR
    U[User chat] --> UI[Streamlit UI]
    UI --> G{LangGraph agent}
    G -->|route| R[Retrieve factsheets]
    G -->|route| M[Fetch market data]
    G -->|route| C[Compute portfolio]
    R --> RAG[(Chroma vector store)]
    M --> Y[yfinance]
    C --> L[logic: scoring / allocation / contributions]
    R & M & C --> RF[Reflect / self-check]
    RF -->|ok| A[Answer]
    RF -->|revise| G
```

**Layer separation** (kept clean from the thesis version):

| Layer | Package | Responsibility |
|-------|---------|----------------|
| UI | `ui/` | Streamlit chat + recommendation rendering |
| Agent | `agent/` | LangGraph graph, tools, state (routing + reflection) |
| RAG | `rag/` | Ingest factsheets → Chroma, retriever with citations |
| Data | `data_sources/` | `yfinance` wrapper, risk-metric calculations |
| Core | `logic/` | Pure, deterministic scoring / allocation / contributions |
| LLM | `llm/` | Multi-provider model factory + prompts |

### 🎓 IBM course mapping

| Course topic | Where it lives |
|---|---|
| RAG pipeline, Chroma/FAISS, chunking | `rag/ingest.py`, `rag/retriever.py` |
| Agentic RAG with query routing | `agent/graph.py` (route node) |
| ReAct / Reflection architectures | `agent/graph.py` (reflect node + loop) |
| LangGraph state machines | `agent/graph.py`, `agent/state.py` |
| MCP / FastMCP servers | `mcp_server/server.py` |
| Evaluation & LLM-as-judge | `eval/extraction.py`, `eval/judge.py` |

## 🚀 Getting started

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows:  .\venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

# 2. Install (editable, with the extras you want)
pip install -e ".[providers,rag,mcp,dev]"

# 3. Configure secrets
cp .env.example .env      # then fill in your provider API key

# 4. Build the RAG corpus + vector store (optional but recommended —
#    enables cited, grounded recommendations)
python -m rag.build_docs   # generates fund profiles from live market data
python -m rag.ingest       # chunks + embeds them into chroma_db/

# 5. Run
streamlit run app.py
```

### 📚 How grounding works

Official factsheets/KIIDs are copyrighted, so the default corpus is **generated
from data the project can legally reproduce**: `rag/build_docs.py` writes one
Markdown fund profile per ETF (live fundamentals + computed 5-year risk metrics,
with an as-of date). `rag/ingest.py` chunks the documents (paragraph-aware, with
contextual title headers) and embeds them with Chroma's built-in local ONNX
MiniLM model — no API key, no torch. You can additionally drop official PDF
factsheets into `data/factsheets/`; the ingest step picks up both. At answer
time the app retrieves per-fund passages and the LLM must cite them (`[1]`) for
every fund claim; the UI shows the cited passages in a *Sources* panel.

### 🔑 Choosing an LLM provider

The model is selected via one env var — no code change needed:

```env
LLM_MODEL=google_genai:gemini-2.5-flash   # or anthropic:claude-sonnet-5, openai:gpt-4o-mini, ...
```

### 🔌 MCP server

The same data / risk / RAG capabilities the agent uses internally are also
published as a standalone **Model Context Protocol** service (`mcp_server/server.py`),
so any MCP client — Claude Desktop, another agent, an IDE — can call them without
importing this codebase.

```bash
# stdio transport (the default MCP wiring)
python -m mcp_server.server

# or over HTTP for quick manual testing
python -m mcp_server.server --transport streamable-http
```

Exposed tools: `get_prices`, `compute_risk_metrics`, `list_universe`,
`target_volatility`, `plan_contributions`, and `retrieve_factsheet`. To register
it with Claude Desktop, add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "robo-advisor": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/investment_bot_neu"
    }
  }
}
```

### 📏 Evaluation harness

The LLM pipeline is measured, not just written. `eval/` holds a labelled test
set of chat transcripts → expected `UserProfile` and an **LLM-as-judge** pass
that scores recommendation explanations on grounding, relevance, clarity and
safety.

```bash
python -m eval.run              # profile-extraction accuracy (per-field + overall)
python -m eval.run --judge      # + LLM-as-judge over live recommendations
```

Extraction hits the configured model once per case; the judge pass runs the full
advisor graph end-to-end and grades each explanation 1–5. The scoring logic
(field comparison, aggregation, judge parsing) is pure and unit-tested offline in
`tests/test_eval.py`, so CI verifies the harness without an API key.

## ☁️ Deploying a public demo

The repo is deploy-ready for **Streamlit Community Cloud** (or any host that runs
`streamlit run app.py` from a `requirements.txt`). Three things make a public
demo robust and cheap to run:

- **Snapshot data (no live network needed).** Yahoo blocks datacenter IPs, so the
  deployed app loads `data/universe_snapshot.json` — a committed, point-in-time
  snapshot of the enriched ETF universe — instead of calling `yfinance` on every
  cold start. Regenerate locally with `python -m data_sources.snapshot`; set
  `USE_LIVE_DATA=1` locally to prefer live metrics. The app falls back to the
  snapshot automatically if a live fetch fails.
- **Bring-your-own-key (BYOK).** The public instance ships **no** shared LLM key,
  so visitors don't drain the owner's quota. A sidebar panel lets each visitor
  paste their own key (Google / Anthropic / OpenAI); it stays in their session
  and is passed straight to the provider, never to a shared env var. Before a key
  is entered, a committed **example recommendation** (`data/demo_recommendation.json`)
  is shown so the product is legible with zero setup.
- **Grounding still works.** `chroma_db/` is gitignored, so the app rebuilds the
  vector store from the committed factsheets on first boot. A SQLite shim
  (`rag/_compat.py` + `pysqlite3-binary`, Linux-only) covers hosts whose system
  SQLite is too old for Chroma.

Deploy steps: point Streamlit Community Cloud at `app.py`; it installs from
`requirements.txt`. Optionally add an owner key under **App → Settings → Secrets**
(`GOOGLE_API_KEY = "…"`) if you want live chat without BYOK — otherwise leave it
out and visitors bring their own.

## 🧪 Development

```bash
pytest            # run the test suite
ruff check .      # lint
ruff format .     # format
```

## ⚠️ Disclaimer

This is an educational project. It does **not** constitute financial or investment advice.
No recommendation produced by this software should be acted upon without consulting a
licensed professional.

## 📄 License

MIT
