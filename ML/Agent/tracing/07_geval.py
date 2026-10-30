"""
Hello-world: G-Eval.

G-Eval lets you score outputs against ANY criterion you write in plain
English. The judge LLM (uses OPENAI_API_KEY) reads your criteria and
returns a 0–1 score.

Here we score "empathy" of a customer-support reply.
"""

from dotenv import load_dotenv
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

load_dotenv()

empathy = GEval(
    name="Empathy",
    criteria=(
        "Score 1.0 if the response acknowledges the customer's frustration, "
        "validates their feelings, and offers a clear next step. "
        "Score 0.0 if the response is robotic or dismissive."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

cases = [
    ("warm", "I'm so sorry you're dealing with this — I'll connect you with a specialist right now."),
    ("cold", "Per policy section 4.2, your dispute will be reviewed within 10 business days."),
]

for label, output in cases:
    tc = LLMTestCase(input="I'm furious about a fraud charge!", actual_output=output)
    empathy.measure(tc)
    print(f"  {label:6s} score={empathy.score:.2f}  pass={empathy.is_successful()}")
    print(f"          reason: {empathy.reason}\n")
