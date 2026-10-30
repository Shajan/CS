# Simple Hello-World Examples

One tiny, runnable script per tool taught in this workshop.

Each script is self-contained — no FinTech agent, no RAG, no LangGraph. Just the
smallest example that shows what the tool does.

## Run order (suggested)

| # | Script | Tool | What it shows |
|---|---|---|---|
| 1 | `01_langsmith.py` | LangSmith | Trace a single LLM call |
| 1b | `01b_langsmith_no_langchain.py` | LangSmith | Trace plain Python functions (no LangChain, no LLM) |
| 1c | `01c_langsmith_google_adk.py` | LangSmith + Google ADK | Trace ADK agents (no LLM). Uses its own venv — see the script header |
| 2 | `02_tiktoken.py` | tiktoken | Count tokens locally |
| 3 | `03_moderation.py` | OpenAI Moderation | Flag harmful input |
| 4 | `04_presidio.py` | Presidio | Detect & redact PII |
| 5 | `05_guardrails.py` | Guardrails AI | Block SSNs in output |
| 6 | `06_deepeval.py` | DeepEval | Score faithfulness |
| 7 | `07_geval.py` | G-Eval | Score with custom criteria |

## Setup

You should already have the workshop deps installed:

```bash
pip install -r requirements.txt
```

Make sure `.env` has `OPENAI_API_KEY` (and `LANGCHAIN_API_KEY` for scripts 1, 1b and 1c).

## Run

```bash
python simple/01_langsmith.py
python simple/02_tiktoken.py
# ...etc
```
