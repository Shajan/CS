"""
Hello-world: LangSmith tracing.

Makes one LLM call. With LANGCHAIN_TRACING_V2=true, the call appears
as a trace at https://smith.langchain.com under your project.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "simple-hello-world")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm.invoke("Say hello in one short sentence.")

print("Response:", response.content)
print("\nOpen https://smith.langchain.com to see the trace.")
print(f"Project: {os.environ['LANGCHAIN_PROJECT']}")
