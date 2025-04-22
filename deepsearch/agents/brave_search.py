import json
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

def brave_search(query: str, max_results: int = 10, use_ai_snippets: bool = False) -> List[SearchResult]:
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

        print("Brave search results:")
        print(json.dumps(data, indent=2))
        
        # Extract results
        results = []
        if "web" in data and "results" in data["web"]:
            for item in data["web"]["results"]:
                content_list = []
                if item.get("description", "") != "":
                    content_list.append(item.get("description"))
                if use_ai_snippets:
                    if item.get("extra_snippets", []) != []:
                        content_list.extend(item.get("extra_snippets"))
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content="\n".join(content_list),
                    score=None  # Brave doesn't provide relevance scores
                )
                results.append(result)
        
        return results

    except Exception as e:
        logger.error(f"Error in Brave search: {str(e)}", exc_info=True)
        return []

def brave_search_agent(state: SearchState, max_results: int = 10, use_ai_snippets: bool = False) -> SearchState:
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
        return state

    try:
        # Perform the search
        results = brave_search(
            state.original_query,
            max_results=max_results,
            use_ai_snippets=use_ai_snippets
        )

        # Update the state
        state.brave_results = results

    except Exception as e:
        logger.error(f"Error in Brave search agent: {str(e)}", exc_info=True)

    return state 