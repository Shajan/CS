#!/bin/bash

# Install Python backend dependencies
cd backend
source .venv/bin/activate

# Start backend
uvicorn main:app --reload &
