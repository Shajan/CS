"""
Hello-world: Presidio.

Detects PII (names, emails, SSNs, credit cards, ...) using spaCy NER
and regex, then redacts it. Runs locally — no API call.

Requires the spaCy model:  python -m spacy download en_core_web_lg
"""

import os
import sys

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Note: Presidio's UsSsnRecognizer explicitly blocklists 123-45-6789 as a
# known dummy/placeholder value, so it won't flag it — use a different number.
text = "My name is Alice Johnson, my SSN is 234-56-7890, email alice@example.com."

# Step 1: detect
results = analyzer.analyze(
    text=text,
    language="en",
    entities=["PERSON", "EMAIL_ADDRESS", "US_SSN"],
)

# Step 2: redact
redacted = anonymizer.anonymize(text=text, analyzer_results=results).text

print("Before:", text)
print("After: ", redacted)
print("Found: ", [r.entity_type for r in results])

# Skip interpreter teardown — chromadb/torch/grpc native libs SIGBUS at exit on macOS arm64
sys.stdout.flush()
os._exit(0)
