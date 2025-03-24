import asyncio
from browser_use import Agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load the opeain keys
load_dotenv()

async def main():
  agent = Agent(
    task="Find the population of each state in India, sort by pupulation",
    llm=ChatOpenAI(model="gpt-4o"),
  )
  await agent.run()

asyncio.run(main())
