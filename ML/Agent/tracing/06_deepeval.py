"""
Hello-world: DeepEval FaithfulnessMetric.

Asks: "Is the answer fully supported by the retrieved context?"
A judge LLM (uses OPENAI_API_KEY) reads context + answer and scores 0–1.
"""

from dotenv import load_dotenv
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric

load_dotenv()

# A faithful answer — the claim "$35" appears in the context.
faithful = LLMTestCase(
    input="What is the overdraft fee?",
    actual_output="The overdraft fee is $35 per transaction.",
    retrieval_context=["SecureBank charges a $35 overdraft fee per transaction."],
)

# A hallucinated answer — "$50" is NOT in the context.
hallucinated = LLMTestCase(
    input="What is the overdraft fee?",
    actual_output="The overdraft fee is $50 per transaction.",
    retrieval_context=["SecureBank charges a $35 overdraft fee per transaction."],
)

metric = FaithfulnessMetric(threshold=0.7)

for label, tc in [("faithful", faithful), ("hallucinated", hallucinated)]:
    metric.measure(tc)
    print(f"  {label:14s} score={metric.score:.2f}  pass={metric.is_successful()}")
    print(f"                 reason: {metric.reason}\n")
