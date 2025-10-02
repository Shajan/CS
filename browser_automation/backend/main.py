from fastapi import FastAPI, Request
from pydantic import BaseModel
from playwright.async_api import async_playwright
import asyncio

app = FastAPI()

class Task(BaseModel):
    url: str
    instruction: str

@app.post("/execute")
async def execute_task(task: Task):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(task.url)
        # Minimal LLM proxy: echo instruction to console
        print(f"[LLM] Instruction: {task.instruction}")
        await asyncio.sleep(2)  # simulate processing
        await browser.close()
    return {"status": "done"}

