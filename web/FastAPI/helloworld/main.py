from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "echo": ""})

@app.post("/", response_class=HTMLResponse)
async def echo_text(request: Request, text: str = Form(...)):
    return templates.TemplateResponse("form.html", {"request": request, "echo": text})

