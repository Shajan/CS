# Setup python virtual env
python -m venv .venv
source .venv/bin/activate

# Install python packages
pip install -r requirements.txt

# Install node.js browser automation library
playwright install chromium

# Add LLM keys to .env file
