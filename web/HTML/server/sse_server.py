import asyncio
import time
from fastapi import FastAPI
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
      <div id="result"></div>

      <script>
        var source = new EventSource("/events");
        source.onmessage = function(event) {
          document.getElementById("result").innerHTML += event.data + "<br>";
        };
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

