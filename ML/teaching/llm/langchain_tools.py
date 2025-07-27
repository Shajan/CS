#pip install langchain langchain-openai python-dotenv

import os
import re
import sys
import getpass
from typing import Tuple, Optional, List
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

API_VERSION = "2024-02-15-preview"  # bump if you're on something newer

# ----------------------------
# Local "tools"
# ----------------------------
def add(a: float, b: float) -> float:
    return a + b
def multiply(a: float, b: float) -> float:
    return a * b
def get_username() -> str:
    return getpass.getuser()

# ----------------------------
# Parse:  Tool: name(args)
# ----------------------------
TOOL_RE = re.compile(r"^\s*Tool\s*:\s*([a-zA-Z_]\w*)\s*\((.*)\)\s*$", re.IGNORECASE | re.DOTALL)
def parse_tool_call(line: str) -> Optional[Tuple[str, list]]:
    m = TOOL_RE.match(line.strip())
    if not m:
        return None
    name, arg_str = m.group(1), m.group(2).strip()
    if name == "get_username":
        return name, []
    if name in {"add", "multiply"}:
        if not arg_str:
            raise ValueError(f"{name} expects 2 arguments")
        parts = [p.strip() for p in arg_str.split(",")]
        if len(parts) != 2:
            raise ValueError(f"{name} expects exactly 2 arguments")
        try:
            a = float(parts[0])
            b = float(parts[1])
        except ValueError:
            raise ValueError(f"{name} arguments must be numbers")
        return name, [a, b]
    return None

def run_tool(name: str, args: list):
    if name == "add":
        return add(*args)
    if name == "multiply":
        return multiply(*args)
    if name == "get_username":
        return get_username()
    raise ValueError(f"Unknown tool: {name}")

# ----------------------------
# LangChain Azure OpenAI setup
# ----------------------------
def build_llm() -> AzureChatOpenAI:
    load_dotenv()
    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_version=API_VERSION,
        temperature=0.2,
    )

SYSTEM_PROMPT = """You are a concise, helpful assistant.
You can ask me to run one of these tools:
1) add(a, b)         -> number
2) multiply(a, b)    -> number
3) get_username()    -> string   (returns the OS username of this machine)
How to use a tool:
- If a tool is needed, respond with exactly one line that starts with:
  Tool: <tool_name>(<comma separated args>)
  Examples:
  Tool: add(3, 5)
  Tool: multiply(2, 4)
  Tool: get_username()
I will execute the tool and show you the result in a follow-up message that looks like:
"Result: <value>"
After you see that result, give your final, concise answer to the user.
If no tool is needed, answer directly.
"""
def main():
    llm = build_llm()
    history: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    print("Type 'exit' or 'quit' to stop.\n")
    try:
        while True:
            user_input = input("you> ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("bye!")
                break
            history.append(HumanMessage(content=user_input))
            # Round 1: model decides if it wants a tool
            ai_msg: AIMessage = llm.invoke(history)
            print(f"assistant> {ai_msg.content}\n")
            history.append(ai_msg)
            parsed = parse_tool_call(ai_msg.content or "")
            if not parsed:
                # No tool call -> done
                continue
            tool_name, args = parsed
            try:
                result = run_tool(tool_name, args)
            except Exception as e:
                error_text = f"ERROR while running {tool_name}: {e}"
                print(f"(tool error)  => {error_text}\n")
                history.append(HumanMessage(content=error_text))
                final_msg: AIMessage = llm.invoke(history)
                print(f"assistant> {final_msg.content}\n")
                history.append(final_msg)
                continue
            print(f"(tool result) => {result}\n")
            # Feed result back as if the user supplied it
            history.append(HumanMessage(content=f"Result: {result}"))
            # Round 2: final answer
            final_msg: AIMessage = llm.invoke(history)
            print(f"assistant> {final_msg.content}\n")
            history.append(final_msg)
    except (KeyboardInterrupt, EOFError):
        print("\nbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()
