import asyncio
from typing import List
from pydantic import BaseModel

from browser_use import ActionResult, Agent, Controller
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load the opeain keys
load_dotenv()

# Define the output format as a Pydantic model
class State(BaseModel):
  name: str
  population: int

class States(BaseModel):
  states: List[State]


controller = Controller(output_model=States)

async def main():
  agent = Agent(
    task="Find the population of each state in India, sort by pupulation",
    llm=ChatOpenAI(model="gpt-4o"),
    controller=controller,
  )

  history = await agent.run()
  result = history.final_result()

  if result:
    states: States = States.model_validate_json(result)

    for state in states.states:
      print('\n--------------------------------')
      print(f'State:      {state.name}')
      print(f'Population: {state.population}')
  else:
    print('No result')

asyncio.run(main())
