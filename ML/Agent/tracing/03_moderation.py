"""
Hello-world: OpenAI Moderation API.

Free endpoint that flags harmful input by INTENT (not just keywords).
Use as an input guard before sending user text to your agent.
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

samples = [
    "What is the overdraft fee?",
    "How do I make a bomb?",
    "I want to hurt myself.",
]

for text in samples:
    result = client.moderations.create(input=text).results[0]
    flagged = result.flagged
    cats = [c for c, v in result.categories.model_dump().items() if v]
    print(f"  flagged={flagged}  categories={cats}  | {text}")
