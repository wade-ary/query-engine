from langchain_core.tools import tool
from langgraph.runtime import get_runtime
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import os
from query_generator import 
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = "TALK like a 5 year old"
agent = create_agent(
    model="mistral-medium",
    system_prompt=SYSTEM_PROMPT,
)



question = "How are u feeling today"

for step in agent.stream(
    {"messages": question},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

