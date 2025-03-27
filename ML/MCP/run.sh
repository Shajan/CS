#!/bin/bash

# Use first argument if provided, otherwise default to 'server.py'
SCRIPT_TO_RUN="${1:-sample1.py}"

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Push to that directory
pushd "${SCRIPT_DIR}" > /dev/null

python -m venv .venv
source .venv/bin/activate

# Start the server
python "$SCRIPT_TO_RUN"

popd > /dev/null

