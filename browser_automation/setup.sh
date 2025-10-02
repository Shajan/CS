#!/bin/bash

# Install Python backend dependencies
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install

# Install frontend depenencies
cd ../frontend
npm install
