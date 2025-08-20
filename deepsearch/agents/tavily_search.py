from typing import Tuple, List, Dict, Any, Generator
import os
import logging
from dotenv import load_dotenv
from tavily import TavilyClient

from deepsearch.schemas.agents import SearchState, SearchResult

# Set up logging
logger = logging.getLogger("deepsearch.tavily")

# Load environment variables
load_dotenv()

# Get Tavily API key
tavily_api_key = os.environ.get("TAVILY_API_KEY")


def init_tavily_client():
    """Initialize the Tavily client."""
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")

    return TavilyClient(api_key=tavily_api_key)


def convert_tavily_results(tavily_results: List[Dict[str, Any]]) -> List[SearchResult]:
    """Convert Tavily search results to our SearchResult model."""
    search_results = []

    for i, result in enumerate(tavily_results):
        # Extract content with validation
        content = result.get("content", "")

        # Ensure content is a string
        if not isinstance(content, str):
            logger.warning(f"Tavily result {i} has non-string content of type {type(content)}")

            # Try to convert to string properly
            if content is None:
                content = ""
            elif isinstance(content, (list, tuple)) and len(content) > 0:
                # Log details of problematic list content
                logger.warning(f"List/tuple content detected: {content[:10] if len(content) > 10 else content}")
                # Join list elements into a string if they're strings, otherwise convert each element to string
                content = " ".join(str(item) for item in content)
            else:
                # Convert other types to string
                content = str(content)

        # Create the search result
        search_results.append(
            SearchResult(
                title=str(result.get("title", "Untitled")),  # Ensure title is a string
                url=str(result.get("url", "")),              # Ensure URL is a string
                content=content,                             # Now always a string
                score=None  # Tavily doesn't provide scores by default
            )
        )

    return search_results

def tavily_search_agent(state: SearchState) -> SearchState:
    """
    Fetches real-time web search results using the Tavily API.

    Args:
        state: The current search state with the refined query

    Returns:
        Updated state with search results
    """
    try:
        # Use the original query
        query = state.original_query

        logger.debug(f"Searching Tavily for: {query}")

        # Initialize Tavily client
        client = init_tavily_client()

        # Perform the search
        search_response = client.search(
            query=query,
            search_depth="advanced",  # Get comprehensive results
            max_results=10            # Fetch enough results for good indexing
        )

        # Convert the results to our model
        tavily_results = convert_tavily_results(search_response.get("results", []))

        logger.debug(f"Found {len(tavily_results)} results from Tavily")

        # Update the state
        state.tavily_results = tavily_results
    except Exception as e:
        logger.error(f"Error in Tavily search: {str(e)}", exc_info=True)
        # Set empty results on error
        state.tavily_results = []

    return state


def search_tavily(query: str) -> List[SearchResult]:
    """Search Tavily for a query."""
    client = init_tavily_client()

    try:
        search_response = client.search(
            query=query,
            search_depth="advanced",
            max_results=10
        )
        return convert_tavily_results(search_response.get("results", []))
    except Exception as e:
        logger.error(f"Error in Tavily search: {str(e)}", exc_info=True)
        return []
