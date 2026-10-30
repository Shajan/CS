"""
Hello-world: LangSmith tracing a Google ADK app, with no LLM.

ADK agents don't have to call a model. Here are three custom agents that
just run Python code: "greet" runs "make_greeting" then "shout". One call to
configure_google_adk() makes LangSmith trace every ADK run.

With LANGCHAIN_TRACING_V2=true, you'll see one trace at
https://smith.langchain.com:

    google_adk.session
    └── greet
        ├── make_greeting
        └── shout

Needs a newer langsmith than the workshop pins, so use its own venv:
    python3 -m venv simple/.venv-adk
    simple/.venv-adk/bin/pip install google-adk "langsmith[google-adk]" python-dotenv
    simple/.venv-adk/bin/python simple/01c_langsmith_google_adk.py
"""

import asyncio
import os

from dotenv import load_dotenv
from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions
from google.adk.runners import InMemoryRunner
from google.genai import types
from langsmith.integrations.google_adk import configure_google_adk

load_dotenv()
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "simple-google-adk")

configure_google_adk()


def text(s: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=s)])


class MakeGreeting(BaseAgent):
    async def _run_async_impl(self, ctx):
        name = ctx.user_content.parts[0].text
        greeting = f"Hello, {name}!"
        yield Event(
            author=self.name,
            content=text(greeting),
            actions=EventActions(state_delta={"greeting": greeting}),
        )


class Shout(BaseAgent):
    async def _run_async_impl(self, ctx):
        yield Event(author=self.name, content=text(ctx.session.state["greeting"].upper()))


class Greet(BaseAgent):
    async def _run_async_impl(self, ctx):
        for agent in self.sub_agents:
            async for event in agent.run_async(ctx):
                yield event


greet = Greet(
    name="greet",
    sub_agents=[MakeGreeting(name="make_greeting"), Shout(name="shout")],
)


async def main():
    runner = InMemoryRunner(agent=greet, app_name="hello")
    session = await runner.session_service.create_session(app_name="hello", user_id="me")

    async for event in runner.run_async(
        user_id="me",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="world")]),
    ):
        print(f"{event.author}: {event.content.parts[0].text}")


asyncio.run(main())
print("\nOpen https://smith.langchain.com to see the trace.")
print(f"Project: {os.environ['LANGCHAIN_PROJECT']}")
