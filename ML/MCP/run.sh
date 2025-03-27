#!/bin/bash

# Use first argument if provided, otherwise default to 'sample1.py'
SCRIPT_TO_RUN="${1:-sample1.py}"

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the directory where the .venv is
pushd "${SCRIPT_DIR}" > /dev/null

source .venv/bin/activate

# Start the server
python "$SCRIPT_TO_RUN"

# Go back to where we were 
popd > /dev/null
