import asyncio
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

app = FastAPI()

# Configure CORS to allow port 8000 in case the http server is running on that port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # Allow requests from your HTTP server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the HTML file (optional)
@app.get("/", response_class=HTMLResponse)
async def get_html():
    html_content = """
    <!DOCTYPE html>
    <html>
    <body>
      <h1>Server-Sent Events with FastAPI</h1>
      <h4>Specific event</h4>
      <div id="update"></div>
      <h4>Append stream</h4>
      <div id="result"></div>

      <script>

        var source = new EventSource("/events");
        source.onmessage = function(event) {
          document.getElementById("result").innerHTML += event.data + "<br>";
        };

        // Listen for the custom 'update' event
        var source_ex = new EventSource("/events_ex");
        source_ex.addEventListener('my_event', (event) => {
          document.getElementById("update").innerHTML = event.data + "<br>";
        });

      </script>
    </body>
    </html>
    """
    return html_content

@app.get("/events")
async def events():
    async def event_stream():
        while True:
            await asyncio.sleep(1)  # Wait 1 second between messages
            yield f"data: The current time is {time.ctime()}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# stream with id, allows browser to continue if connection is lost
events = [
    {"id": 1, "event": "message", "data": "First event"},
    {"id": 2, "event": "update", "data": "Update event"},
    {"id": 3, "event": "message", "data": "Another event"},
]

@app.get("/events_ex")
async def get_events(request: Request):
    async def event_generator():
        # Check for Last-Event-ID header, browser automatically sets this
        last_event_id = request.headers.get("Last-Event-ID")
        start_id = 0

        if last_event_id:
            start_id = int(last_event_id) + 1

        # Stream events from the appropriate starting point
        for event in events[start_id:]:
            yield f"id: {event['id']}\n"
            yield f"event: my_event\n"
            yield f"data: {event['data']}\n\n"
            await asyncio.sleep(1)  # Simulate delay between events

    return StreamingResponse(event_generator(), media_type="text/event-stream")
