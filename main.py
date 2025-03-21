#!/usr/bin/env python3
"""
Main entry point for the DeepSearch application.
"""

import argparse
import logging
from dotenv import load_dotenv
import os
from typing import Dict, Any
from deepsearch.models import SearchState
from deepsearch.agents import (
    query_refinement_agent,
    tavily_search_agent,
    faiss_indexing_agent,
    bm25_search_agent,
    llama_reasoning_agent
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("deepsearch")

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
            "sources": sources[:5]  # Limit to 5 sources
        }
    except Exception as e:
        # Handle any unexpected errors
        logger.error(f"Unexpected error in pipeline: {str(e)}", exc_info=True)
        return {
            "original_query": query,
            "refined_query": None,
            "answer": "An unexpected error occurred while processing your query. Please try again later.",
            "confidence": 0.0,
            "sources": []
        }

def main():
    """Main entry point for the application."""
    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DeepSearch: A multi-agent deep search system")
    parser.add_argument("query", type=str, help="The query to search for")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--show-confidence", "-c", action="store_true", help="Show confidence score")
    parser.add_argument("--disable-refinement", "-d", action="store_true", help="Disable query refinement")

    args = parser.parse_args()

    # Run the workflow
    print(f"Searching for: {args.query}\n")
    print("Processing... (this may take a while depending on your hardware)\n")

    result = run_simple_pipeline(args.query, args.disable_refinement)

    # Print the results
    print("\n" + "="*80)
    print("SEARCH RESULTS")
    print("="*80)

    if result['refined_query']:
        print(f"\nOriginal query: {result['original_query']}")
        print(f"Refined query: {result['refined_query']}")
    else:
        print(f"\nQuery: {result['original_query']}")

    print("\n" + result['answer'])

    # Only show confidence if explicitly requested
    if args.show_confidence:
        print(f"\nConfidence: {result['confidence']:.2f}")

    if args.verbose:
        print("\nSources:")
        for i, source in enumerate(result['sources']):
            print(f"  {i+1}. {source['title']} - {source['url']}")


if __name__ == "__main__":
    main()