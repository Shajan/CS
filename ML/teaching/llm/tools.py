#pip install openai python-dotenv

import os
import re
import sys
import getpass
from typing import Tuple, Optional
from dotenv import load_dotenv
from openai import AzureOpenAI

API_VERSION = "2024-02-15-preview"  # adjust if you use a newer one


# ----------------------------
# Local "tools"
# ----------------------------
def add(a: float, b: float) -> float:
    return a + b

def multiply(a: float, b: float) -> float:
    return a * b

def get_username() -> str:
    # Use the OS username of the machine running this script
    return getpass.getuser()


# ----------------------------
# Simple parser for: "Tool: name(arg1, arg2, ...)"
# Only supports our three demo tools:
#   add(a, b), multiply(a, b), get_username()
# ----------------------------
TOOL_RE = re.compile(r"^\s*Tool\s*:\s*([a-zA-Z_]\w*)\s*\((.*)\)\s*$", re.IGNORECASE | re.DOTALL)

def parse_tool_call(line: str) -> Optional[Tuple[str, list]]:
    """
    Returns (tool_name, args_list) or None if not a tool call.
    We keep it simple & explicit for the 3 demo tools.
    """
    m = TOOL_RE.match(line.strip())
    if not m:
        return None

    name, arg_str = m.group(1), m.group(2).strip()

    if name == "get_username":
        # no args expected
        return name, []

    # For add/multiply: expect two numeric args "x, y"
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

    # Unknown tool name
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
# Azure OpenAI setup
# ----------------------------
def build_client() -> AzureOpenAI:
    load_dotenv()
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=API_VERSION,
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )


SYSTEM_PROMPT = f"""You are a concise, helpful assistant.

You can ask me to run one of these tools:

1) add(a, b)         -> number
2) multiply(a, b)    -> number
3) get_username()    -> string   (returns the OS username of this machine)

**How to use a tool:**
- If a tool is needed, respond with exactly one line that *starts* with:
  Tool: <tool_name>(<comma separated args>)
  Examples:
  Tool: add(3, 5)
  Tool: multiply(2, 4)
  Tool: get_username()

I will execute the tool and show you the result in a follow-up message with role "tool".
After you see the tool result, give your final, concise answer to the user.
If no tool is needed, answer directly.
"""


def main():
    client = build_client()
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Type 'exit' or 'quit' to stop.\n")

    try:
        while True:
            user_input = input("you> ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("bye!")
                break

            messages.append({"role": "user", "content": user_input})

            # Round 1: ask model what to do (maybe a tool call, maybe final answer)
            resp = client.chat.completions.create(
                model=deployment,
                messages=messages,
                temperature=0.2,
            )
            assistant_msg = resp.choices[0].message.content or ""
            print(f"assistant> {assistant_msg}\n")
            messages.append({"role": "assistant", "content": assistant_msg})

            # Detect "Tool:" prefix
            parsed = parse_tool_call(assistant_msg)
            if not parsed:
                # No tool call -> done
                continue

            tool_name, args = parsed

            # Execute tool locally
            try:
                result = run_tool(tool_name, args)
            except Exception as e:
                # Return the error to the model so it can apologize / correct
                error_text = f"ERROR while running {tool_name}: {e}"
                print(f"(tool error)  => {error_text}\n")
                messages.append({"role": "tool", "content": error_text})

                # Let the model recover/finalize
                followup = client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    temperature=0.2,
                )
                final_msg = followup.choices[0].message.content or ""
                print(f"assistant> {final_msg}\n")
                messages.append({"role": "assistant", "content": final_msg})
                continue

            # Send tool result back to the model
            print(f"(tool result) => {result}\n")
            messages.append({"role": "user", "content": f"Result: {result}"})


            # Round 2: let the model produce the final answer
            followup = client.chat.completions.create(
                model=deployment,
                messages=messages,
                temperature=0.2,
            )
            final_msg = followup.choices[0].message.content or ""
            print(f"assistant> {final_msg}\n")
            messages.append({"role": "assistant", "content": final_msg})

    except (KeyboardInterrupt, EOFError):
        print("\nbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()