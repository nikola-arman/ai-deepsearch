from typing import List
import os
import json
import logging
import re
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

CRITICAL: Your entire response MUST be valid parseable JSON, starting with '[' and ending with ']'.
Do not include any text before or after the JSON array.
Do not include any explanation, markdown formatting, or code blocks around the JSON.

Example of CORRECT response format:
["query 1", "query 2", "query 3", "query 4", "query 5"]

Example of INCORRECT response format:
Here are the queries: ["query 1", "query 2", "query 3", "query 4", "query 5"]
```
["query 1", "query 2", "query 3", "query 4", "query 5"]
```
"""

def init_query_expansion_llm(temperature: float = 0.7):
    """Initialize the language model for query expansion using OpenAI-compatible API."""
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "no-need"),
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base if not openai_api_key or openai_api_key == "not-needed" else None,
        temperature=temperature,  # Higher temperature for query diversity
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
    llm = init_query_expansion_llm(temperature=0.5)

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

    # Clean up the response text to improve JSON parsing chances
    expanded_queries_text = expanded_queries_text.strip()
    # Remove any markdown code block markers
    expanded_queries_text = re.sub(r'^```json\s*', '', expanded_queries_text)
    expanded_queries_text = re.sub(r'\s*```$', '', expanded_queries_text)
    # Remove any stray markdown characters
    expanded_queries_text = re.sub(r'^#+\s*', '', expanded_queries_text)

    # Parse the JSON response
    try:
        expanded_queries = json.loads(expanded_queries_text)
        if not isinstance(expanded_queries, list):
            # Handle case where the response isn't a list
            logger.warning("Query expansion output is not a list, using default")
            expanded_queries = [refined_query]
    except json.JSONDecodeError:
        # If parsing fails, extract queries using a simple heuristic
        logger.warning(f"Failed to parse query expansion JSON output: {expanded_queries_text[:100]}...")
        expanded_queries_text = expanded_queries_text.strip()

        # First attempt: Check if the text contains JSON-like content with brackets
        if expanded_queries_text.startswith("[") and expanded_queries_text.endswith("]"):
            try:
                # Try to fix common issues like single quotes instead of double quotes
                fixed_text = expanded_queries_text
                # Replace single quotes with double quotes if they're being used for JSON
                if "'" in fixed_text and '"' not in fixed_text:
                    fixed_text = fixed_text.replace("'", '"')

                # Try parsing the fixed text
                expanded_queries = json.loads(fixed_text)
                logger.info("Successfully parsed JSON after basic fixing")

                if not isinstance(expanded_queries, list):
                    expanded_queries = [refined_query]
            except json.JSONDecodeError:
                # If still can't parse, try splitting manually
                logger.info("Attempting to extract queries from bracket-enclosed text")
                # Remove the outer brackets
                expanded_queries_text = expanded_queries_text[1:-1]

                # Split by commas that are followed by a quote character
                # This helps handle cases where there might be commas inside the queries
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

                logger.info(f"Extracted {len(expanded_queries)} queries from bracket-enclosed text")
        else:
            # Second attempt: look for quoted strings that might be queries
            quoted_strings = re.findall(r'"([^"]*)"', expanded_queries_text)

            if quoted_strings:
                expanded_queries = quoted_strings
                logger.info(f"Extracted {len(expanded_queries)} queries from quoted strings")
            else:
                # Third attempt: try splitting by newlines or numbers
                lines = [line.strip() for line in expanded_queries_text.split('\n') if line.strip()]
                potential_queries = []

                for line in lines:
                    # Remove common prefixes like numbers, bullets, etc.
                    line = re.sub(r'^[\d\.\)\-\*]+\s*', '', line)
                    if line and not line.lower().startswith(('query', 'example', 'correct', 'incorrect')):
                        potential_queries.append(line)

                if potential_queries:
                    expanded_queries = potential_queries
                    logger.info(f"Extracted {len(expanded_queries)} queries from line by line analysis")
                else:
                    # Fall back to using just the refined query
                    expanded_queries = [refined_query]
                    logger.warning("Could not extract any queries, using refined query as fallback")

    # Always include the refined query if it's not already in the list
    if refined_query not in expanded_queries:
        expanded_queries.insert(0, refined_query)

    # Filter out any empty queries
    expanded_queries = [query for query in expanded_queries if query and query.strip()]

    # Ensure we have at least one query
    if not expanded_queries:
        expanded_queries = [refined_query]

    # Update the state
    state.generated_queries = expanded_queries

    # Log the generated queries
    logger.info(f"Generated {len(expanded_queries)} expanded queries")
    for i, query in enumerate(expanded_queries):
        logger.info(f"  Query {i+1}: {query}")

    return state