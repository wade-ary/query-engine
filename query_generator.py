import json
import os
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()


client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

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

    response = client.chat.complete(
        model="mistral-medium",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content.strip()
    try:
        terms = json.loads(raw)
    except json.JSONDecodeError:
        # fallback: split on newlines, commas, or semicolons
        lines = [line.strip("-• )") for line in raw.splitlines() if line.strip()]
        terms = [t for line in lines for t in line.split(",")]

  
    return [t.strip() for t in dict.fromkeys(terms) if t.strip()]


