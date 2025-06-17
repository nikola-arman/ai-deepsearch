from typing import List
import os
import json
import logging
import re
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from deepsearch.schemas.agents import SearchState

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger("deepsearch.query_expansion")

# Get the OpenAI-compatible API base URL and API key
openai_api_base = os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1")
openai_api_key = os.environ.get("LLM_API_KEY", "not-needed")

# Define the prompt template for query expansion
QUERY_EXPANSION_TEMPLATE = """You are a query expansion expert. Your task is to understand the user's information needs and generate diverse search queries that will help find comprehensive answers.

Original query: {original_query}

# ANALYSIS PROCESS
First, analyze the query carefully:
1. What is the core information need behind this query?
2. What are the key entities and concepts in this query?
3. What are 5-7 DIFFERENT ASPECTS or angles of this topic that would be valuable to explore?
4. What related concepts would provide useful context for a complete answer?

# QUERY GENERATION INSTRUCTIONS
Based on your analysis, generate 5 search queries that:
1. EXPAND on the query with more specific details or broader context
2. Explore DIFFERENT FACETS of the same general topic
3. Include key entities from the query
4. Add relevant modifiers, related concepts, or specific aspects
5. Vary in scope (some narrower/focused, some broader/comprehensive)
6. Use different phrasing and vocabulary while maintaining meaning

# BALANCING FOCUS AND EXPANSION
- Each query should be clearly connected to the original topic
- Queries should be MEANINGFULLY DIFFERENT from each other
- Add relevant context, qualifiers, timeframes, or specificity
- Include both technical and practical perspectives when appropriate
- Queries should feel like they're exploring different angles of the same topic

# EXAMPLES

GOOD EXPANSION (diverse but related):
Original query: "new updates from Nvidia"
Expanded queries:
["latest Nvidia driver updates and performance improvements",
"Nvidia's recent hardware announcements and upcoming product releases",
"DLSS and ray tracing updates in Nvidia's newest software",
"Nvidia AI and machine learning framework updates 2025",
"professional graphics and workstation updates from Nvidia"]

BAD EXPANSION (too similar):
Original query: "new updates from Nvidia"
Expanded queries:
["new updates from Nvidia",
"recent updates from Nvidia",
"latest updates from Nvidia",
"current Nvidia updates",
"Nvidia's new updates"]

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
        state: The current search state with original query

    Returns:
        Updated state with multiple generated queries
    """
    # Initialize the LLM with higher temperature for more diversity
    llm = init_query_expansion_llm(temperature=0.8)  # Increase temperature for more creative expansions

    # Create the prompt
    query_expansion_prompt = PromptTemplate(
        input_variables=["original_query"],
        template=QUERY_EXPANSION_TEMPLATE
    )

    # Use the newer approach to avoid deprecation warnings
    chain = query_expansion_prompt | llm

    # Generate the expanded queries
    response = chain.invoke({
        "original_query": state.original_query
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
            expanded_queries = [state.original_query]
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
                    expanded_queries = [state.original_query]
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
                    # Fall back to using just the original query
                    expanded_queries = [state.original_query]
                    logger.warning("Could not extract any queries, using original query as fallback")

    # Post-process queries to ensure balance of diversity and relevance
    processed_queries = []

    # Always include the original query in the mix
    processed_queries.append(state.original_query)

    # Extract key terms from original query for basic relevance checking
    original_terms = set(state.original_query.lower().split())
    original_entities = extract_entities(state.original_query)

    # Score queries by combining relevance and diversity metrics
    scored_queries = []
    for query in expanded_queries:
        # Skip very short queries or duplicates
        if len(query.split()) < 3 or query.lower() == state.original_query.lower():
            continue

        # Check for core relevance through entity or term overlap
        query_terms = set(query.lower().split())
        query_entities = extract_entities(query)

        # Calculate entity overlap
        entity_overlap = 0
        if original_entities and query_entities:
            entity_overlap = len(set(original_entities).intersection(set(query_entities))) / len(original_entities) if original_entities else 0

        # Calculate term overlap
        term_overlap = len(original_terms.intersection(query_terms)) / len(original_terms) if original_terms else 0

        # Calculate a base relevance score
        relevance_score = max(entity_overlap, term_overlap * 0.8)

        # Skip completely irrelevant queries
        if relevance_score < 0.2 and not any(e in query.lower() for e in original_entities):
            logger.info(f"Skipping irrelevant query: {query}")
            continue

        # Calculate diversity score against existing processed queries
        diversity_score = 1.0  # Start with maximum diversity
        for existing in processed_queries:
            # Calculate Jaccard similarity
            query_words = set(query.lower().split())
            existing_words = set(existing.lower().split())
            overlap = len(query_words.intersection(existing_words))
            union = len(query_words.union(existing_words))
            similarity = overlap / union if union > 0 else 0

            # Reduce diversity score based on similarity to existing queries
            diversity_score = min(diversity_score, 1 - similarity)

        # Combine scores - balance relevance and diversity
        # We want queries that are somewhat relevant but also diverse
        combined_score = (relevance_score * 0.6) + (diversity_score * 0.4)

        scored_queries.append((query, combined_score))

    # Sort by combined score (descending)
    scored_queries.sort(key=lambda x: x[1], reverse=True)

    # Add the top queries to our processed list
    for query, score in scored_queries:
        if query not in processed_queries and len(processed_queries) < 5:
            processed_queries.append(query)

    # If we don't have enough diverse queries, add some variations
    if len(processed_queries) < 5:
        # Generate more creative variations than before
        variations = generate_creative_variations(state.original_query, original_entities)
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
    common_tech_entities = ['nvidia', 'amd', 'intel', 'apple', 'microsoft', 'google', 'amazon',
                           'gpu', 'cpu', 'ai', 'driver', 'software', 'hardware', 'update']
    for word in words:
        clean_word = word.strip('.,;:!?()"\'').lower()
        if clean_word in common_tech_entities and clean_word not in entities:
            entities.append(clean_word)

    return entities

def generate_creative_variations(query, entities):
    """Generate more creative variations of a query based on common patterns and entities."""
    variations = []

    # Tech-specific context additions based on identified entities
    contexts = {
        'nvidia': ['gaming', 'graphics cards', 'DLSS', 'ray tracing', 'AI', 'professional graphics',
                  'machine learning', 'data center', 'GPU', 'drivers', 'GeForce', 'performance'],
        'apple': ['iOS', 'macOS', 'devices', 'M-series chips', 'App Store', 'software', 'hardware'],
        'microsoft': ['Windows', 'Office', 'Azure', 'Cloud', 'Teams', 'enterprise', 'Surface'],
        'google': ['Android', 'Chrome', 'Search', 'Cloud', 'AI', 'Workspace', 'Pixel'],
        'ai': ['machine learning', 'neural networks', 'deep learning', 'frameworks', 'tools'],
        'update': ['changelog', 'new features', 'improvements', 'releases', 'versions']
    }

    # Base query without any additional qualifiers
    base_words = query.lower().split()

    # Generate entity-specific expansions
    for entity in entities:
        if entity in contexts:
            for context in contexts[entity][:3]:  # Limit to prevent too many variations
                if context.lower() not in base_words:
                    variations.append(f"{query} for {context}")
                    variations.append(f"{entity} {context} {query.replace(entity, '').strip()}")

    # Add time-based variations
    time_frames = ['recent', 'latest', '2025', 'upcoming', 'new']
    for time in time_frames:
        if time not in base_words:
            variations.append(f"{time} {query}")

    # Add perspective variations
    perspectives = ['comprehensive guide to', 'technical details of', 'benefits of',
                    'how to use', 'comparison of', 'analysis of']
    for perspective in perspectives:
        if not any(w in perspective for w in base_words):
            variations.append(f"{perspective} {query}")

    # Shuffle and return unique variations
    import random
    random.shuffle(variations)
    return list(dict.fromkeys(variations))  # Remove duplicates while preserving order