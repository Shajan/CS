from typing import Optional, List
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)

app = FastAPI()

class Item(BaseModel):
  name: str
  age: int
  description: Optional[str] = None


l : List[Item] = []

# Landing page
@app.get("/")
def landing_page():
  return {"Hello": "World!"}

# Simple apis ----------------------------
@app.get("/py-list")
def py_list():
  return l

@app.get("/py-incr/{num}")
def py_increment(num: int):
  return num + 1 

# Using pydantic --------------------------
@app.get("/items/list")
def list_items() -> List[Item]:
  return l

@app.get("/items/{idx}")
def get_item(idx: int) -> Item:
  if idx >= 0 and idx < len(l):
    return l[idx]
  raise HTTPException(status_code=404, detail="Item not found")

@app.post("/items/add")
def add_item(item: Item):
  l.append(item)

# Using async --------------------------
@app.get("/slow-incr/{num}")
async def slow_increment(num: int):
  async def _sleep_blocking():
    import time
    logging.info("begin _sleep_blocking")
    time.sleep(10)
    logging.info("end _sleep_blocking")

  async def _sleep_non_blocking():
    import asyncio
    logging.info("begin _sleep_non_blocking")
    await asyncio.sleep(10)
    logging.info("end _sleep_non_blocking")

  #await _sleep_blocking()
  await _sleep_non_blocking()
  return num + 1
  
