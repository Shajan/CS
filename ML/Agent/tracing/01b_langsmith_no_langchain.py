"""
Hello-world: LangSmith tracing with no LangChain and no LLM.

LangSmith can trace any Python function. Add @traceable and each call is
logged as a "run" with its inputs, outputs, timing, and errors. A traced
function that calls another traced function creates a nested trace.

With LANGCHAIN_TRACING_V2=true, you'll see one trace at
https://smith.langchain.com:

    greet
    ├── make_greeting
    └── shout
"""

import os
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "simple-hello-world")


@traceable
def make_greeting(name: str) -> str:
    return f"Hello, {name}!"


@traceable
def shout(text: str) -> str:
    return text.upper()


@traceable
def greet(name: str) -> str:
    return shout(make_greeting(name))


print("Result:", greet("world"))
print("\nOpen https://smith.langchain.com to see the trace.")
print(f"Project: {os.environ['LANGCHAIN_PROJECT']}")
