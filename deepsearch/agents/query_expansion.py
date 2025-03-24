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
First, analyze the refined query carefully:
1. What is the EXACT core information need behind this refined query?
2. What specific knowledge or answers is the user looking for?
3. What are the 3-5 MOST RELEVANT aspects or dimensions of this specific topic?
4. What directly related background information would help provide a complete answer?

# QUERY GENERATION INSTRUCTIONS
Based on your analysis, generate 5 search queries that:
1. Are ALL DIRECTLY RELATED to the REFINED query - do NOT add new unrelated topics
2. MUST maintain the core information need and topic from the refined query
3. Focus on different facets of EXACTLY THE SAME TOPIC in the refined query
4. All queries should feel like slight variations or elaborations on the refined query
5. Keep all named entities (companies, products, people, etc.) from the refined query
6. AVOID adding topics or products that weren't mentioned in the refined query

# CRITICAL
EVERY query MUST be about the EXACT SAME TOPIC as the refined query. The queries should NOT diverge into related but different topics - they should be variations on the SAME specific information need.

# EXAMPLES

GOOD EXPANSION (stays on topic):
Refined query: "new updates from Nvidia"
Expanded queries:
["latest Nvidia software updates and releases",
"recent Nvidia driver updates and their improvements",
"Nvidia's newest product announcements and releases",
"latest Nvidia GPU driver updates for gaming performance",
"Nvidia's recent technology updates and advancements"]

BAD EXPANSION (strays from topic):
Refined query: "new updates from Nvidia"
Expanded queries:
["new updates from Nvidia",
"Nvidia RTX 4090 specifications",
"AMD vs Nvidia comparison",
"Nvidia stock price prediction 2025",
"historical timeline of Nvidia's GPU innovations"]

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
    llm = init_query_expansion_llm(temperature=0.7)  # Slightly lower temperature for more focused diversity

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

    # Post-process queries to ensure relevance to the refined query
    processed_queries = []

    # Always include the refined query as the first query
    processed_queries.append(refined_query)

    # Extract key terms from refined query for relevance checking
    refined_terms = set(refined_query.lower().split())
    refined_entities = extract_entities(refined_query)

    # Add other queries, ensuring they're diverse but related to the refined query
    for query in expanded_queries:
        # Skip very short queries
        if len(query.split()) < 3:
            continue

        # Skip exact duplicate of refined query
        if query.lower() == refined_query.lower():
            continue

        # Check for relevance to refined query
        query_terms = set(query.lower().split())
        query_entities = extract_entities(query)

        # Check for entity overlap - at least one entity should match
        entity_overlap = False
        if refined_entities and query_entities:
            entity_overlap = len(set(refined_entities).intersection(set(query_entities))) > 0

        # Calculate term overlap
        term_overlap = len(refined_terms.intersection(query_terms)) / len(refined_terms) if refined_terms else 0

        # A query is relevant if it has significant term overlap OR contains shared entities
        is_relevant = term_overlap >= 0.4 or entity_overlap

        if not is_relevant:
            logger.info(f"Skipping irrelevant query: {query}")
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

    # If we don't have enough relevant queries, generate slight variations of the refined query
    if len(processed_queries) < 3:
        variations = generate_query_variations(refined_query)
        for variation in variations:
            if variation not in processed_queries:
                processed_queries.append(variation)
                if len(processed_queries) >= 5:
                    break

    # Limit to 5 diverse queries
    expanded_queries = processed_queries[:5]

    # Update the state
    state.generated_queries = expanded_queries

    # Log the generated queries
    logger.info(f"Generated {len(expanded_queries)} expanded queries")
    for i, query in enumerate(expanded_queries):
        logger.info(f"  Query {i+1}: {query}")

    return state

def extract_entities(text):
    """Extract potential named entities from text based on capital letters."""
    if not text:
        return []

    # Simple heuristic: words that start with capital letters, not at the beginning of sentences
    words = text.split()
    entities = []

    for i, word in enumerate(words):
        # Skip first word or words after punctuation
        if i == 0 or (i > 0 and words[i-1][-1] in '.?!'):
            continue

        # Check if word starts with capital letter
        if word and word[0].isupper():
            # Remove punctuation
            clean_word = word.strip('.,;:!?()"\'')
            if clean_word:
                entities.append(clean_word.lower())

    # Also extract company/product names that might be lowercase
    common_tech_entities = ['nvidia', 'amd', 'intel', 'apple', 'microsoft', 'google', 'amazon']
    for word in words:
        clean_word = word.strip('.,;:!?()"\'').lower()
        if clean_word in common_tech_entities and clean_word not in entities:
            entities.append(clean_word)

    return entities

def generate_query_variations(query):
    """Generate simple variations of a query."""
    variations = []

    words = query.split()
    if not words:
        return variations

    # Add "latest" variation
    if not any(w.lower() in ['latest', 'recent', 'new', 'newest'] for w in words):
        variations.append(f"latest {query}")

    # Add "recent" variation
    if not any(w.lower() in ['recent', 'latest', 'new', 'newest'] for w in words):
        variations.append(f"recent {query}")

    # Add "comprehensive" variation
    if not any(w.lower() in ['comprehensive', 'complete', 'detailed'] for w in words):
        variations.append(f"comprehensive {query}")

    # Add "guide to" variation
    if not any(w.lower() in ['guide', 'tutorial', 'how'] for w in words):
        variations.append(f"guide to {query}")

    return variations