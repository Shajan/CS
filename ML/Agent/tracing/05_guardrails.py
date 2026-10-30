"""
Hello-world: Guardrails AI.

Validates an LLM output against rules. Here we block any text
that contains a US SSN.

Setup (one time):
    pip install guardrails-ai
    pip install --no-deps ../vendor/guardrails_hub_wheels/*.whl   # guardrails hub install's
                                                                    # backend is decommissioned
"""

from guardrails import Guard
from guardrails.hub import RegexMatch

# Regex matches strings that do NOT contain an SSN (negative lookahead).
# match_type="search" => a match means VALID; no match => fails the guard.
guard = Guard().use(
    RegexMatch(
        regex=r"(?s)^(?!.*\b\d{3}-\d{2}-\d{4}\b).*$",
        match_type="search",
        on_fail="exception",
    )
)

samples = [
    "The overdraft fee is $35.",
    "Your SSN on file is 123-45-6789.",
]

for text in samples:
    try:
        guard.validate(text)
        print(f"  PASS:    {text}")
    except Exception:
        print(f"  BLOCKED: {text}")
