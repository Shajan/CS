"""
Hello-world: tiktoken.

Count how many tokens a string would use for a given OpenAI model —
locally, no API call.
"""

import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o-mini")

text = "Hello, world! How many tokens am I?"
tokens = encoder.encode(text)

print(f"Text:     {text}")
print(f"Tokens:   {tokens}")
print(f"Count:    {len(tokens)}")
print(f"Decoded:  {[encoder.decode([t]) for t in tokens]}")
