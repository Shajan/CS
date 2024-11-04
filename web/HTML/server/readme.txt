Setup:
  Use python 3.11
    conda activate py_3_11

  Create python virtual env in the conda env
    python -m venv .venv
    source .venv/bin/activate 

  Install prereqs
    pip install -r ./requirements.txt

Run:
  uvicorn sse_server:app --reload --port 8001

