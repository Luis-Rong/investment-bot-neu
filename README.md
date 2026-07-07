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

## 🚀 Getting started

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows:  .\venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

# 2. Install (editable, with the extras you want)
pip install -e ".[providers,rag,dev]"

# 3. Configure secrets
cp .env.example .env      # then fill in your provider API key

# 4. Run
streamlit run app.py
```

### 🔑 Choosing an LLM provider

The model is selected via one env var — no code change needed:

```env
LLM_MODEL=google_genai:gemini-2.5-flash   # or anthropic:claude-sonnet-5, openai:gpt-4o-mini, ...
```

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
