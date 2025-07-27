#pip install openai python-dotenv

from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import sys

API_VERSION = "2024-02-15-preview"  # bump if you're on something newer

def build_client() -> AzureOpenAI:
    load_dotenv()
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=API_VERSION,
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )

def chat_loop(client: AzureOpenAI, deployment: str):
    # Start the running conversation with a system message (optional but recommended)
    messages = [
        {"role": "system", "content": "You are a concise, helpful assistant."}
    ]
    print("Type 'exit' or 'quit' (or Ctrl+C) to stop.\n")
    try:
        while True:
            user_input = input("you> ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("bye!")
                break
            messages.append({"role": "user", "content": user_input})
            resp = client.chat.completions.create(
                model=deployment,              # deployment name, not base model name
                messages=messages,
                temperature=0.2,
            )
            assistant_msg = resp.choices[0].message.content
            print(f"assistant> {assistant_msg}\n")
            # Save assistant response back into the running history
            messages.append({"role": "assistant", "content": assistant_msg})
    except (KeyboardInterrupt, EOFError):
        print("\nbye!")
        sys.exit(0)

def main():
    client = build_client()
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    chat_loop(client, deployment)

if __name__ == "__main__":
    main()