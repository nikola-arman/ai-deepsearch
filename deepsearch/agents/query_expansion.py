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
QUERY_EXPANSION_TEMPLATE = """You are a query expansion expert. Your task is to understand the user's information needs and generate diverse search queries that will help find comprehensive answers.

Original query: {original_query}
Refined query: {refined_query}

# ANALYSIS PROCESS
First, analyze the user's query:
1. What is the core information need behind this query?
2. What specific knowledge or answers is the user looking for?
3. What are the different aspects or dimensions of this topic that would be relevant?
4. What background information would help provide a complete answer?
5. What are potential follow-up questions the user might have?

# QUERY GENERATION INSTRUCTIONS
Based on your analysis, generate 5 HIGHLY DIVERSE search queries that:
1. Target the DIFFERENT ASPECTS of what the user is trying to learn
2. Address potential knowledge gaps the user may have about the topic
3. Focus on distinct information needs related to the original question
4. Include technical, practical, comparative, historical, and future-oriented perspectives
5. Keep all named entities (companies, products, people, etc.) EXACTLY as written

# EXAMPLE
For the query "How does blockchain work?", first identify the information needs:
- Core need: Understanding blockchain technology's fundamental mechanisms
- Aspects: Technical details, real-world applications, historical development, comparisons to other technologies
- Knowledge gaps: How consensus works, scalability issues, security implications

Then generate diverse queries addressing these needs:
["technical explanation of blockchain distributed ledger mechanisms",
"practical applications of blockchain technology beyond cryptocurrency",
"blockchain consensus protocols comparison proof-of-work vs proof-of-stake",
"evolution and history of blockchain technology since 2008",
"blockchain scalability challenges and layer 2 solutions"]

# RESPONSE FORMAT
Your entire response MUST be valid parseable JSON, starting with '[' and ending with ']'.
Do not include any text before or after the JSON array.

Example of CORRECT response format:
["query 1", "query 2", "query 3", "query 4", "query 5"]
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
    # Initialize the LLM with higher temperature for more diversity
    llm = init_query_expansion_llm(temperature=0.8)

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

    # Post-process queries to ensure diversity
    processed_queries = []

    # Include the refined query if it's substantive
    if refined_query and len(refined_query.split()) > 2:
        processed_queries.append(refined_query)

    # Add other queries, ensuring they're diverse from what we already have
    for query in expanded_queries:
        # Skip very short queries or ones that are too similar to others we've already included
        if len(query.split()) < 3:
            continue

        # Check for similarity with existing processed queries
        too_similar = False
        for existing in processed_queries:
            # Simple word overlap calculation
            query_words = set(query.lower().split())
            existing_words = set(existing.lower().split())
            # Calculate Jaccard similarity
            overlap = len(query_words.intersection(existing_words))
            union = len(query_words.union(existing_words))
            similarity = overlap / union if union > 0 else 0

            # If more than 70% similar by word overlap, consider too similar
            if similarity > 0.7:
                too_similar = True
                logger.info(f"Skipping too similar query: {query} (similar to {existing})")
                break

        if not too_similar:
            processed_queries.append(query)

    # Ensure we have at least the refined query if nothing else
    if not processed_queries:
        processed_queries = [refined_query]

    # Limit to 5 diverse queries
    expanded_queries = processed_queries[:5]

    # Update the state
    state.generated_queries = expanded_queries

    # Log the generated queries
    logger.info(f"Generated {len(expanded_queries)} expanded queries")
    for i, query in enumerate(expanded_queries):
        logger.info(f"  Query {i+1}: {query}")

    return state