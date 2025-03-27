# Reference https://modelcontextprotocol.io/quickstart/server 

# Setup python virtual env
python -m venv .venv
source .venv/bin/activate

# Install python packages
pip install -r requirements.txt

# Add LLM keys to .env file
source ./.env

# Install claude
npm install -g @anthropic-ai/claude-code
npx -y @wonderwhy-er/desktop-commander setup

