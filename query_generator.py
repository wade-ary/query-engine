import json
import os
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-5-mini")


def generate_search_terms(query: str, num_terms: int = 8) -> list[str]:
    # Convert a user's research question into multiple keyword search terms for finding relevant papers
    # Uses GPT-4 to generate search phrases that would work well in academic databases 
    
    prompt = f"""
You are a research‐assistant LLM.  
Given the user's question, suggest {num_terms} concise keyword or phrase searches 
that would retrieve the most relevant papers from databases like PubMed, Semantic Scholar, or OpenAlex.  
Respond with a valid JSON array of strings (no extra text).

Example output:
["large language model reasoning", "LLM inference speed", "transformer reasoning efficiency", …]

User query:
\"\"\"{query}\"\"\"
    """.strip()

    response = llm.invoke([{"role": "user", "content": prompt}])
    raw = response.content.strip()
    try:
        terms = json.loads(raw)
    except json.JSONDecodeError:
        # fallback: split on newlines, commas, or semicolons
        lines = [line.strip("-• )") for line in raw.splitlines() if line.strip()]
        terms = [t for line in lines for t in line.split(",")]

  
    return [t.strip() for t in dict.fromkeys(terms) if t.strip()]

REFINEMENT_PROMPT = """
You are a query-refinement assistant.

Your job: Rewrite the user's question into a clearer, more precise, academically-searchable version.

Rules:
- Keep the rewritten query to ONE sentence.
- Make it specific, direct, and unambiguous.
- Remove slang, filler, or conversational phrasing.
- Keep all technical meaning from the original question.
- Do NOT add new information.

Return ONLY the refined query as plain text.
"""
def fix_query(query: str, extra_info: str= "") -> str:
    if extra_info == None:
        extra_info = ""
    prompt = REFINEMENT_PROMPT + f"\n\nUser query:\n{query} + {extra_info}\n\nRefined query:"
    refined = llm.invoke(prompt).content.strip()
    return refined

def is_confusing(query: str) -> bool:
        prompt = """ Make an educated guess, if this query is too confusing say YES or say NO thats i
        t. if the query contains the word confused alwaysb returbn YES hust the words YES or NO. {query} here is the quyery."""

        confusion = llm.invoke(prompt).content.strip()
        if "YES" in confusion:
             return True
        else:
             return False