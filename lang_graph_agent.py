import uuid
import os
from typing import Literal, TypedDict
from IPython.display import Image, display
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.types import Command, interrupt
from langgraph.graph import END, START, StateGraph
load_dotenv()

from query_generator import generate_search_terms as gen_terms
from query_generator import fix_query
from sample_workflow import get_raw_papers_from_terms
from final_summary import summarize_papers_with_query

class RagAgentState(TypedDict):
    # Query
    user_query: str

    refined_query: str | None

    search_terms: list[str] | None

    returned_docs: list[dict] | None

    summary_response: str | None

llm = ChatOpenAI(model="gpt-5-mini")

def refine_query(state: RagAgentState) -> RagAgentState:
    "Check the user's query and refine it to make appropriate for semantic search"
    refined = fix_query(state["user_query"])
    state["refined_query"] = refined
    return state

def generate_search_terms(state: RagAgentState) -> RagAgentState:
    "Generates a list of search terms for given refined_query"
    terms = gen_terms(state["refined_query"])
    state["search_terms"] = terms
    return state

def retrieve_documents(state: RagAgentState) -> RagAgentState:
    documents = get_raw_papers_from_terms(state["search_terms"])
    state["returned_docs"] = documents
    return state
    

def summarize_results(state: RagAgentState) -> RagAgentState:
    summary = summarize_papers_with_query(state["returned_docs"], state["refined_query"])
    state["summary_response"] = summary
    return state

builder = StateGraph(RagAgentState)

builder.add_node("refine_query", refine_query)
builder.add_node("generate_search_terms", generate_search_terms)
builder.add_node("retrieve_documents", retrieve_documents)
builder.add_node("summarize_results", summarize_results)

builder.add_edge(START, "refine_query")
builder.add_edge("refine_query", "generate_search_terms")
builder.add_edge("generate_search_terms", "retrieve_documents")
builder.add_edge("retrieve_documents", "summarize_results")
builder.add_edge("summarize_results", END)

from langgraph.checkpoint.memory import InMemorySaver
memory = InMemorySaver()
app = builder.compile(checkpointer = memory)

initial_state = {
    "user_query": "Number 1 cause of death in the USA",
}

# Run with a thread_id for persistence
config = {"configurable": {"thread_id": "1"}}
final_state = app.invoke(initial_state, config)
print(final_state["summary_response"])