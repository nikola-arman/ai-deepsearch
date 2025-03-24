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
    llama_reasoning_agent,
    query_expansion_agent,
    deep_reasoning_agent
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

def run_deep_search_pipeline(query: str, disable_refinement: bool = False, max_iterations: int = 3) -> Dict[str, Any]:
    """Run the multi-query, iterative deep search pipeline with reasoning agent."""
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


def prompt(messages: list[dict[str, str]], **kwargs) -> str:
    assert len(messages) > 0, "received empty messages"
    query = messages[-1]['content']
    
    res: Dict = run_deep_search_pipeline(query)
    
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
