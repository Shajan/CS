from fastapi import FastAPI, WebSocket 
from fastapi.responses import FileResponse, StreamingResponse
import asyncio

app = FastAPI()

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    for i in range(100):
        await websocket.send_json({"message": f"Message {i}"})
        await asyncio.sleep(1)
    await websocket.close()

# Server Sent Event (SSE) endpoint
async def event_generator():
    for i in range(100):
        yield f"data: Message {i}\n\n"
        await asyncio.sleep(1)

@app.get("/events")
async def get_events():
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Serve static HTML files
@app.get("/sse")
async def get_sse_html():
    return FileResponse("templates/sse.html")

@app.get("/ws")
async def get_ws_html():
    return FileResponse("templates/ws.html")

