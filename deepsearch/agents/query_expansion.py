from typing import List
import os
import json
import logging
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from deepsearch.models import SearchState

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger("deepsearch.query_expansion")

# Get the OpenAI-compatible API base URL and API key
openai_api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8080/v1")
openai_api_key = os.environ.get("OPENAI_API_KEY", "not-needed")

# Define the prompt template for query expansion
QUERY_EXPANSION_TEMPLATE = """You are a query expansion expert. Your task is to generate
multiple diverse and relevant search queries based on the user's original query.

Original query: {original_query}
Refined query: {refined_query}

INSTRUCTIONS:
1. Generate 5 unique search queries that approach the user's question from different angles
2. Each query should focus on specific aspects or dimensions of the original question
3. All queries should be directly relevant to answering the original question
4. Keep all named entities (companies, products, people, etc.) EXACTLY as written
5. Make the queries search-friendly and specific
6. Format your response as a JSON array of strings containing ONLY the queries.

IMPORTANT: Your entire response must be valid parseable JSON, starting with '[' and ending with ']'.
Do not include any text before or after the JSON array.

Example of CORRECT response format:
["query 1", "query 2", "query 3", "query 4", "query 5"]

Example of INCORRECT response format:
Here are the queries: ["query 1", "query 2", "query 3", "query 4", "query 5"]
"""

def init_query_expansion_llm():
    """Initialize the language model for query expansion using OpenAI-compatible API."""
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "no-need"),
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base if not openai_api_key or openai_api_key == "not-needed" else None,
        temperature=0.7,  # Higher temperature for query diversity
        max_tokens=1024
    )
    return llm

def query_expansion_agent(state: SearchState) -> SearchState:
    """
    Generates multiple diverse queries based on the user's original query.

    Args:
        state: The current search state with original and potentially refined query

    Returns:
        Updated state with multiple generated queries
    """
    # Initialize the LLM
    llm = init_query_expansion_llm()

    # Create the prompt
    query_expansion_prompt = PromptTemplate(
        input_variables=["original_query", "refined_query"],
        template=QUERY_EXPANSION_TEMPLATE
    )

    # Use the newer approach to avoid deprecation warnings
    chain = query_expansion_prompt | llm

    # Ensure refined_query is available
    refined_query = state.refined_query if state.refined_query else state.original_query

    # Generate the expanded queries
    response = chain.invoke({
        "original_query": state.original_query,
        "refined_query": refined_query
    })

    # Extract the content if it's a message object
    if hasattr(response, 'content'):
        expanded_queries_text = response.content
    else:
        expanded_queries_text = response

    # Parse the JSON response
    try:
        expanded_queries = json.loads(expanded_queries_text)
        if not isinstance(expanded_queries, list):
            # Handle case where the response isn't a list
            logger.warning("Query expansion output is not a list, using default")
            expanded_queries = [refined_query]
    except json.JSONDecodeError:
        # If parsing fails, extract queries using a simple heuristic
        logger.warning("Failed to parse query expansion JSON output, using heuristic extraction")
        expanded_queries_text = expanded_queries_text.strip()

        # Check if the text contains JSON-like content
        if expanded_queries_text.startswith("[") and expanded_queries_text.endswith("]"):
            # Try to extract anything that looks like a list of queries
            # Remove the outer brackets
            expanded_queries_text = expanded_queries_text[1:-1]

            # Split by commas that are followed by a quote character
            # This helps handle cases where there might be commas inside the queries
            import re
            query_items = re.split(r',\s*"', expanded_queries_text)

            # Clean up each query
            expanded_queries = []
            for i, item in enumerate(query_items):
                # Add back the quote that was removed during splitting (except for the first item)
                if i > 0:
                    item = '"' + item

                # Remove surrounding quotes and any escaping
                item = item.strip().strip('"\'').replace('\\"', '"')

                if item:
                    expanded_queries.append(item)
        else:
            # Try to extract queries using different patterns
            # Look for quoted strings
            import re
            quoted_strings = re.findall(r'"([^"]*)"', expanded_queries_text)

            if quoted_strings:
                expanded_queries = quoted_strings
            else:
                # If we can't find any pattern, try splitting by newlines or numbers
                lines = [line.strip() for line in expanded_queries_text.split('\n') if line.strip()]
                potential_queries = []

                for line in lines:
                    # Remove common prefixes like numbers, bullets, etc.
                    line = re.sub(r'^[\d\.\)\-\*]+\s*', '', line)
                    if line and not line.lower().startswith(('query', 'example', 'correct', 'incorrect')):
                        potential_queries.append(line)

                if potential_queries:
                    expanded_queries = potential_queries
                else:
                    # Fall back to using just the refined query
                    expanded_queries = [refined_query]

    # Ensure we have at least one query
    if not expanded_queries:
        expanded_queries = [refined_query]

    # Filter out any non-string items or invalid items
    expanded_queries = [q for q in expanded_queries if isinstance(q, str) and len(q.strip()) > 0]

    # Deduplicate and add the refined query if it's not already included
    expanded_queries = list(set(expanded_queries))
    if refined_query not in expanded_queries:
        expanded_queries.insert(0, refined_query)

    # Update the state with the generated queries
    state.generated_queries = expanded_queries[:5]  # Limit to 5 queries

    logger.info(f"Generated {len(state.generated_queries)} expanded queries")
    for i, query in enumerate(state.generated_queries):
        logger.info(f"  Query {i+1}: {query}")

    return state