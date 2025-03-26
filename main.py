#!/usr/bin/env python3
"""
Main entry point for the DeepSearch application.
"""

import argparse
import logging
from dotenv import load_dotenv
import os
from typing import Dict, Any
from deepsearch.models import SearchState, SearchResult
from deepsearch.agents import (
    tavily_search_agent,
    faiss_indexing_agent,
    bm25_search_agent,
    query_expansion_agent,
    deep_reasoning_agent
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("deepsearch")

def run_deep_search_pipeline(query: str, disable_refinement: bool = False, max_iterations: int = 3) -> Dict[str, Any]:
    """Run the multi-query, iterative deep search pipeline with reasoning agent."""
    try:
        # Initialize state
        state = SearchState(original_query=query)

        # Use original query as refined query (query refinement removed)
        logger.info("Step 1: Using original query (query refinement removed)")
        state.refined_query = state.original_query

        # Step 2: Query Expansion - generate multiple queries
        logger.info("Step 2: Expanding query into multiple search queries...")
        try:
            state = query_expansion_agent(state)
            logger.info(f"  Generated {len(state.generated_queries)} search queries")
        except Exception as e:
            logger.error(f"  Error in query expansion: {str(e)}", exc_info=True)
            # If expansion fails, use just the refined query
            state.generated_queries = [state.refined_query]

        # Iterative search loop
        while not state.search_complete and state.current_iteration < max_iterations:
            iteration = state.current_iteration + 1
            logger.info(f"Beginning search iteration {iteration}...")

            # Reset results for this iteration but keep accumulated results
            previous_results = state.combined_results.copy() if state.combined_results else []
            state.tavily_results = []
            state.faiss_results = []
            state.bm25_results = []
            state.combined_results = []

            # Process each query in this iteration
            for i, query in enumerate(state.generated_queries):
                logger.info(f"  Processing query {i+1}/{len(state.generated_queries)}: {query}")

                # Create a temporary state for this query
                temp_state = SearchState(
                    original_query=state.original_query,
                    refined_query=query  # Use the current query as the refined query
                )

                # Step 3a: Tavily Search for this query
                logger.info(f"    Performing web search...")
                try:
                    temp_state = tavily_search_agent(temp_state)
                    # Tag results with the query that produced them
                    for result in temp_state.tavily_results:
                        result.query = query
                    logger.info(f"    Found {len(temp_state.tavily_results)} web results")
                except Exception as e:
                    logger.error(f"    Error in web search: {str(e)}", exc_info=True)

                # Step 3b: FAISS Indexing (semantic search) for this query
                logger.info(f"    Performing semantic search...")
                try:
                    temp_state = faiss_indexing_agent(temp_state)
                    # Tag results with the query that produced them
                    for result in temp_state.faiss_results:
                        result.query = query
                    logger.info(f"    Found {len(temp_state.faiss_results)} semantic results")
                except Exception as e:
                    logger.error(f"    Error in semantic search: {str(e)}", exc_info=True)

                # Step 3c: BM25 Search (keyword search) for this query
                logger.info(f"    Performing keyword search...")
                try:
                    temp_state = bm25_search_agent(temp_state)
                    # Tag results with the query that produced them
                    for result in temp_state.bm25_results:
                        result.query = query
                    logger.info(f"    Found {len(temp_state.bm25_results)} keyword results")
                except Exception as e:
                    logger.error(f"    Error in keyword search: {str(e)}", exc_info=True)

                # Collect results from this query
                state.tavily_results.extend(temp_state.tavily_results)
                state.faiss_results.extend(temp_state.faiss_results)
                state.bm25_results.extend(temp_state.bm25_results)
                if temp_state.combined_results:
                    state.combined_results.extend(temp_state.combined_results)

            # Add back previous results to ensure continuity
            if state.combined_results:
                # If we have combined results from this iteration, merge with previous
                seen_urls = {result.url for result in state.combined_results}
                for result in previous_results:
                    if result.url not in seen_urls:
                        state.combined_results.append(result)
            else:
                # No new combined results, use previous ones
                state.combined_results = previous_results

            # Make sure combined_results is populated even if BM25 didn't run
            if not state.combined_results:
                # Combine FAISS and Tavily results
                state.combined_results = state.faiss_results + state.tavily_results

            # Deduplicate combined results by URL
            if state.combined_results:
                unique_results = {}
                for result in state.combined_results:
                    # Keep the highest scoring result for each URL
                    if result.url not in unique_results or (result.score is not None and
                        (unique_results[result.url].score is None or result.score > unique_results[result.url].score)):
                        unique_results[result.url] = result

                state.combined_results = list(unique_results.values())
                logger.info(f"  Deduplicated to {len(state.combined_results)} unique results")

            # Step 4: Deep Reasoning - analyze results and decide whether to continue
            logger.info(f"  Analyzing search results and determining next steps...")
            try:
                state = deep_reasoning_agent(state, max_iterations)
                logger.info(f"  Search complete: {state.search_complete}")
                if not state.search_complete:
                    logger.info(f"  Knowledge gaps identified: {len(state.knowledge_gaps)}")
                    logger.info(f"  New queries generated: {len(state.generated_queries)}")
            except Exception as e:
                logger.error(f"  Error in deep reasoning: {str(e)}", exc_info=True)
                # If reasoning fails, stop the search to avoid infinite loops
                state.search_complete = True
                state.final_answer = "I'm sorry, but I couldn't properly analyze the search results due to a technical issue. Please try again with a different query."
                state.confidence_score = 0.1

        # Prepare the response
        if not state.search_complete:
            # If we exited the loop due to max iterations, generate the final answer
            logger.info("Maximum iterations reached, generating final answer...")
            try:
                from deepsearch.agents.deep_reasoning import generate_final_answer
                state = generate_final_answer(state)
            except Exception as e:
                logger.error(f"Error generating final answer: {str(e)}", exc_info=True)
                state.final_answer = "I reached the maximum number of search iterations but couldn't generate a comprehensive answer. Here's what I found: " + "\n".join([f"- {point}" for point in state.key_points])
                state.confidence_score = 0.5

        # Return the results
        sources = []
        if state.combined_results:
            for res in state.combined_results[:5]:  # Limit to 5 sources
                sources.append({
                    "title": res.title,
                    "url": res.url
                })

        # Extract components from the final answer if available
        answer = state.final_answer
        key_points = state.key_points if state.key_points else []
        detailed_notes = state.detailed_notes if state.detailed_notes else None

        return {
            "original_query": state.original_query,
            "refined_query": state.refined_query,
            "generated_queries": state.generated_queries,
            "iterations": state.current_iteration,
            "answer": answer,
            "key_points": key_points,
            "detailed_notes": detailed_notes,
            "confidence": state.confidence_score,
            "sources": sources
        }
    except Exception as e:
        # Handle any unexpected errors
        logger.error(f"Unexpected error in deep search pipeline: {str(e)}", exc_info=True)
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
    parser.add_argument("--max-iterations", "-i", type=int, default=3, help="Maximum number of search iterations")

    args = parser.parse_args()

    # Run the workflow
    print(f"Searching for: {args.query}\n")
    print("Processing... (this may take a while depending on your hardware)\n")

    result = run_deep_search_pipeline(args.query, args.disable_refinement, args.max_iterations)

    # Print the results
    print("\n" + "="*80)
    print("SEARCH RESULTS")
    print("="*80)

    if result['refined_query']:
        print(f"\nOriginal query: {result['original_query']}")
        print(f"Refined query: {result['refined_query']}")
    else:
        print(f"\nQuery: {result['original_query']}")

    if args.verbose:
        print(f"\nSearch iterations: {result['iterations']}")
        print("\nGenerated queries:")
        for i, query in enumerate(result['generated_queries']):
            if query != result['refined_query'] and query != result['original_query']:
                print(f"  - {query}")

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