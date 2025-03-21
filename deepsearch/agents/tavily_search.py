from typing import Tuple, List, Dict, Any
import os
import logging
from dotenv import load_dotenv
from tavily import TavilyClient

from deepsearch.models import SearchState, SearchResult

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

    for result in tavily_results:
        search_results.append(
            SearchResult(
                title=result.get("title", "Untitled"),
                url=result.get("url", ""),
                content=result.get("content", ""),
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
        # Use refined query if available, otherwise use original query
        query = state.refined_query if state.refined_query else state.original_query

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