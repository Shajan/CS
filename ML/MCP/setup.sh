# Reference https://modelcontextprotocol.io/quickstart/server 

# Setup python virtual env
python -m venv .venv
source .venv/bin/activate

# Install python packages
pip install -r requirements.txt

# Add LLM keys to .env file

# Add this server to the list of mcp servers for the application
#
# Example 
#   claude:
#     Add content of sample1.json to  ~/*/Claude/claude_desktop_config.json
#       On mac it is ~/Library/Application Support/Claude/claude_desktop_config.json
#       Create the file if it is not already there.
#       In the json, update the path of ./run.sh
#     Find the location of Claude\claude_desktop_config.json on your PC
