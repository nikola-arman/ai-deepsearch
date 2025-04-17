from typing import Generator, List
import os
import logging
import requests
from deepsearch.models import SearchState, SearchResult
from deepsearch.utils import to_chunk_data, wrap_thought

# Set up logging
logger = logging.getLogger("deepsearch.brave")

# Get Brave API key from environment
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "no-need")

def brave_search(query: str, max_results: int = 10) -> List[SearchResult]:
    """
    Perform a web search using Brave Search API.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List of SearchResult objects
    """
    if not query:
        return []

    try:
        # Prepare the API request
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": BRAVE_API_KEY
        }
        
        params = {
            "q": query,
            "count": max_results,
            "text_decorations": False,
            "safesearch": "moderate"
        }

        # Make the API request
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response
        data = response.json()
        
        # Extract results
        results = []
        if "web" in data and "results" in data["web"]:
            for item in data["web"]["results"]:
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("description", ""),
                    score=None  # Brave doesn't provide relevance scores
                )
                results.append(result)
        
        return results

    except Exception as e:
        logger.error(f"Error in Brave search: {str(e)}", exc_info=True)
        return []

def brave_search_agent(state: SearchState) -> Generator[bytes, None, SearchState]:
    """
    Uses Brave Search API to perform web searches.

    Args:
        state: The current search state

    Returns:
        Updated state with Brave search results
    """
    # Initialize results as empty list
    state.brave_results = []

    # Check if we have a query to search
    if not state.original_query:
        yield to_chunk_data(
            wrap_thought(
                "Brave search agent: No query",
                "No query provided for Brave search"
            )
        )
        return state

    try:
        yield to_chunk_data(
            wrap_thought(
                "Brave search agent: Starting search",
                f"Searching for: {state.original_query}"
            )
        )

        # Perform the search
        results = brave_search(state.original_query)

        # Update the state
        state.brave_results = results
        yield to_chunk_data(
            wrap_thought(
                "Brave search agent: Complete",
                f"Found {len(results)} results"
            )
        )

    except Exception as e:
        logger.error(f"Error in Brave search agent: {str(e)}", exc_info=True)
        yield to_chunk_data(
            wrap_thought(
                "Brave search agent: Error",
                f"Error occurred during Brave search: {str(e)}"
            )
        )

    return state 