from langchain_core.tools import tool
from langgraph.runtime import get_runtime
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import os
from query_generator import generate_search_terms
from sample_workflow import get_raw_papers_from_terms
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
You are a research assistant, when the user gives a query if its too vague you modify it a little if not let it be as is, then you call get_search_terms tool with that query and 
num_terms if user specifies it this will return a list of string that are search terms. 
then you pass that on to the tool get_papers_from_terms, source is always pubmed, topk should be default
unless user specifies how many papers are needed.
Lastly you will get raw json data from get_papers_from_terms, you will neatly format this data with the title and a summaries abstract for each paper 
and a short common theme and differences section underneath
"""
 




@tool
def get_search_terms(query: str, num_terms: int = 3) -> list[str]:
    """
    Generates keyword-style search terms from a natural-language query.

    - Provide the exact user’s research question as `user_query`.
    - Provide how many terms required want as `num_terms` if not specified default 3.
    - This tool returns a list of search terms.
    - The internal model handles all formatting and prompting.

    Always call this tool when you need keyword search terms.
    """

    return generate_search_terms(query, num_terms)


@tool
def get_papers_from_terms(terms: list[str], source: str = "hehe", top_k: int = 5) -> list[dict]:
    """
    Retrieve raw unformatted papers using a list of search terms.

    - Provide the list of keyword search terms as `terms`.
    - Provide `source` as hehe
    - Provide `top_k` for how many top papers to return default to 5.
    - This tool returns a list of {title, abstract} dictionaries.
    - No summarization or formatting is performed here.

    Use this tool after generating search terms.
    """
    return get_raw_papers_from_terms(
        terms=terms,
        source=source,
        top_k=top_k
    )
agent = create_agent(
    model="mistral-small-2402",
    tools = [get_search_terms, get_papers_from_terms],
    system_prompt=SYSTEM_PROMPT,
)

question = "How do LLMs understand context"

for step in agent.stream(
    {"messages": question},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

