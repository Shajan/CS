from openai import AzureOpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Access the env vars
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

API_VERSION = "2024-02-15-preview"
client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=API_VERSION,
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
resp = client.chat.completions.create(
    model=deployment,   # <-- deployment name, not the base model name
    messages=[{"role": "user", "content": "Say 'Hello from Azure OpenAI!'"}],
    temperature=0
)
print(resp.choices[0].message.content)
