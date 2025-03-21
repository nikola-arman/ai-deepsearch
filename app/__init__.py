from . import eternalai_mcp_middleware # do not remove this

import os
os.environ['TAVILY_API_KEY'] = 'no-need'
os.environ['OPENAI_BASE_URL'] = os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL"))
os.environ['OPENAI_API_KEY'] = os.getenv("LLM_API_KEY", 'no-need')

from typing import Dict, Any
from deepsearch.models import SearchState
from deepsearch.agents import (
    query_refinement_agent,
    tavily_search_agent,
    faiss_indexing_agent,
    bm25_search_agent,
    llama_reasoning_agent
)

import logging
logger = logging.getLogger(__name__)



def run_simple_pipeline(query: str, disable_refinement: bool = False) -> Dict[str, Any]:
    """Run a simple pipeline without graph complexity."""
    try:
        # Initialize state
        state = SearchState(original_query=query)

        # Step 1: Query Refinement (if not disabled)
        if not disable_refinement:
            logger.info("Step 1: Refining query...")
            try:
                state = query_refinement_agent(state)
                logger.info(f"  Refined query: {state.refined_query}")
            except Exception as e:
                logger.error(f"  Error in query refinement: {str(e)}", exc_info=True)
                # If refinement fails, use original query
                state.refined_query = state.original_query
        else:
            logger.info("Step 1: Query refinement disabled, using original query")
            state.refined_query = state.original_query

        # Step 2: Tavily Search
        logger.info("Step 2: Performing web search...")
        try:
            state = tavily_search_agent(state)
            logger.info(f"  Found {len(state.tavily_results)} results")
        except Exception as e:
            logger.error(f"  Error in web search: {str(e)}", exc_info=True)
            state.tavily_results = []

        # Step 3: FAISS Indexing (semantic search)
        logger.info("Step 3: Performing semantic search...")
        try:
            state = faiss_indexing_agent(state)
            logger.info(f"  Found {len(state.faiss_results)} semantically relevant results")
        except Exception as e:
            logger.error(f"  Error in semantic search: {str(e)}", exc_info=True)
            state.faiss_results = []

        # Step 4: BM25 Search (keyword search)
        logger.info("Step 4: Performing keyword search...")
        try:
            state = bm25_search_agent(state)
            logger.info(f"  Found {len(state.bm25_results)} keyword relevant results")
            logger.info(f"  Combined {len(state.combined_results)} total relevant results")
        except Exception as e:
            logger.error(f"  Error in keyword search: {str(e)}", exc_info=True)
            state.bm25_results = []
            # Ensure we have combined results even if BM25 fails
            if not state.combined_results:
                state.combined_results = state.faiss_results + state.tavily_results

        # Step 5: LLM Reasoning
        logger.info("Step 5: Generating answer...")
        try:
            state = llama_reasoning_agent(state)
            logger.info(f"  Confidence: {state.confidence_score}")
        except Exception as e:
            logger.error(f"  Error in reasoning: {str(e)}", exc_info=True)
            # Provide a fallback answer
            state.final_answer = "I'm sorry, but I couldn't generate a proper answer for your query due to a technical issue. Please try again with a different query."
            state.confidence_score = 0.1

        # Return the results
        sources = []
        if state.combined_results:
            for res in state.combined_results:
                sources.append({
                    "title": res.title,
                    "url": res.url
                })

        return {
            "original_query": state.original_query,
            "refined_query": state.refined_query,
            "answer": state.final_answer,
            "confidence": state.confidence_score,
            "sources": sources[:5],  # Limit to 5 sources,
            "has_error": False
        }

    except Exception as e:
        # Handle any unexpected errors
        logger.error(f"Unexpected error in pipeline: {str(e)}", exc_info=True)
        return {
            "original_query": query,
            "refined_query": None,
            "answer": "An unexpected error occurred while processing your query. Please try again later.",
            "confidence": 0.0,
            "sources": [],
            "has_error": True
        }

def prompt(messages: list[dict[str, str]], **kwargs) -> str:
    assert len(messages) > 0, "received empty messages"
    query = messages[-1]['content']
    
    res: Dict = run_simple_pipeline(query)
    
    sep = "-" * 30
    final_resp = res["answer"]

    if len(res["sources"]) > 0:
        final_resp += "\n## References:\n" 

        for item in res["sources"]:
            final_resp += "- [{title}]({url})\n".format(**item)

    if not res["has_error"]:
        if final_resp.strip()[-1] != '\n':
            final_resp += '\n'

        final_resp += "{sep}\nConfidence score: {confidence}".format(
            confidence=res["confidence"],
            sep=sep
        )

    return final_resp
