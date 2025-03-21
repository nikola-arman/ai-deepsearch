from typing import Tuple, Set
from langchain.prompts import PromptTemplate
import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from deepsearch.models import SearchState

# Load environment variables
load_dotenv()

# Get the OpenAI-compatible API base URL and API key
openai_api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8080/v1")
openai_api_key = os.environ.get("OPENAI_API_KEY", "not-needed")

# Define the prompt template for query refinement
QUERY_REFINEMENT_TEMPLATE = """You are a query refinement expert. Your task is to improve
user queries to make them more specific, clear, and effective for web search.

Original query: {original_query}

IMPORTANT: DO NOT change or replace any company names, product names, people names, or specific entities in the query.
If the query mentions specific companies, products, or people, keep these names EXACTLY as they are.

Provide a refined version of this query that:
1. Preserves all named entities (companies, products, people, etc.) exactly as written
2. Adds descriptive terms or context only if absolutely necessary
3. Removes unnecessary words without altering the core entities
4. Is formatted optimally for web search

Only return the refined query without explanations or additional text.
"""

query_refinement_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=QUERY_REFINEMENT_TEMPLATE
)


def init_query_refinement_llm():
    """Initialize the language model for query refinement using OpenAI-compatible API."""
    # Use OpenAI-compatible server for all operations
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base if not openai_api_key or openai_api_key == "not-needed" else None,
        temperature=0.2,
        max_tokens=100
    )
    return llm


def extract_entities(text: str) -> Set[str]:
    """
    Extract potential named entities from text.
    This is a simple heuristic based on capitalized words and common company names.
    """
    words = text.split()

    # Extract whole words for exact matching
    exact_words = set(words)

    # Find all capitalized words that are likely to be named entities
    potential_entities = re.findall(r'\b[A-Z][a-z]*\b', text)

    # Find potential multi-word entities (simple heuristic)
    multi_word = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text)

    # Find company/product names that might be lowercase
    common_companies = ['google', 'facebook', 'amazon', 'microsoft', 'apple', 'netflix',
                        'tesla', 'twitter', 'uber', 'airbnb', 'wiz', 'zoom', 'slack']

    lowercase_entities = [company for company in common_companies if company in text.lower()]

    # Combine all entities
    all_entities = set(potential_entities + multi_word + lowercase_entities).union(exact_words)
    return all_entities


def validate_entities_preserved(original: str, refined: str) -> bool:
    """
    Validate that named entities from the original query are preserved in the refined query.
    Returns True if entities are preserved, False otherwise.
    """
    original_entities = extract_entities(original)

    # If no entities detected, consider it valid
    if not original_entities:
        return True

    # Check for exact word matches in important cases
    original_lower = original.lower()
    refined_lower = refined.lower()

    # Special handling for known company/product/technology names
    critical_entities = ["wiz", "google", "facebook", "amazon", "microsoft", "apple", "tesla"]

    for entity in critical_entities:
        if entity in original_lower and entity not in refined_lower:
            return False

    # Also check for key words that completely change meaning (acquisition terms)
    acquisition_terms = ["bought", "acquired", "purchase", "acquisition"]
    for term in acquisition_terms:
        if term in original_lower and not any(t in refined_lower for t in acquisition_terms):
            return False

    # General entity check
    for entity in original_entities:
        # Skip common words that aren't likely to be entities
        if entity.lower() in {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about'}:
            continue

        # For longer entities (likely company names), require exact match
        if len(entity) > 3 and entity.lower() not in refined_lower:
            return False

    return True


def query_refinement_agent(state: SearchState) -> SearchState:
    """
    Refines the user's original query to make it more effective for search.

    Args:
        state: The current search state with the original query

    Returns:
        Updated state with the refined query
    """
    # Initialize the LLM
    llm = init_query_refinement_llm()

    # Use the newer approach to avoid deprecation warnings
    chain = query_refinement_prompt | llm

    # Generate the refined query
    response = chain.invoke({"original_query": state.original_query})

    # Extract the content if it's a message object
    if hasattr(response, 'content'):
        refined_query = response.content
    else:
        refined_query = response

    # Clean up the refined query
    refined_query = refined_query.strip()

    # Validate that important entities are preserved
    if not validate_entities_preserved(state.original_query, refined_query):
        # If validation fails, use the original query with quotes for better search
        state.refined_query = f'"{state.original_query}"'
    else:
        # Update the state with the refined query
        state.refined_query = refined_query

    return state