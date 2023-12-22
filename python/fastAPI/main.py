from fastapi import FastAPI

app = FastAPI()

l = []

@app.get("/")
def landing_page():
    return {"Hello": "Chimmu"}

@app.get("/list")
def list():
    return {"l": l}

@app.post("/list/add/{name}")
def list(name: str):
  l.append(name)

