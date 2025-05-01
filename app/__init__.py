from enum import Enum
import uuid
import eai_http_middleware # do not remove this

import os

from deepsearch.magic import retry

os.environ['TAVILY_API_KEY'] = 'no-need'
os.environ['OPENAI_BASE_URL'] = os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL"))
os.environ['OPENAI_API_KEY'] = os.getenv("LLM_API_KEY", 'no-need')
os.environ["EXA_API_KEY"] = "no-need"

from typing import Annotated, Dict, Any, Generator, List
from deepsearch.models import SearchState
from deepsearch.agents import (
    tavily_search_agent,
    faiss_indexing_agent,
    bm25_search_agent,
    llama_reasoning_agent,
    deep_reasoning_agent,
    brave_search_agent,
    information_extraction_agent,
    fact_checking_agent,
    exa_search_agent,
)
from deepsearch.agents.deep_reasoning import init_reasoning_llm
from app.utils import detect_query_complexity, detect_research_intent, get_conversation_summary, reply_conversation

from json_repair import repair_json
import json
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

from langchain.prompts import PromptTemplate
from deepsearch.utils import to_chunk_data, wrap_step_start, wrap_step_finish, wrap_thought


class Retriever(Enum):
    TAVILY = "tavily"
    BRAVE = "brave"
    EXA = "exa"


def write_state_to_file(state: SearchState):
    os.makedirs("output", exist_ok=True)
    with open("output/final_state.json", "w") as f:
        f.write(state.model_dump_json(indent=2))


def get_retriever_from_env() -> List[Retriever]:
    retriever_str = os.getenv("RETRIEVER", "tavily")
    retriever_list = retriever_str.split(",")
    retrievers = []
    for retriever in retriever_list:
        if retriever == Retriever.TAVILY.value:
            retrievers.append(Retriever.TAVILY)
        elif retriever == Retriever.BRAVE.value:
            retrievers.append(Retriever.BRAVE)
        elif retriever == Retriever.EXA.value:
            retrievers.append(Retriever.EXA)
        else:
            raise ValueError(f"Invalid retriever: {retriever}")
    return retrievers


def extract_urls_from_report(report: str) -> set[str]:
    """
    Extract all URLs from a markdown report that appear in markdown links.

    Args:
        report: The markdown report text

    Returns:
        Set of URLs found in the report
    """
    import re
    # Match markdown links with nested square brackets: [text with [nested] brackets](url)
    url_pattern = r'\[(?:[^\[\]]|\[[^\[\]]*\])*\]\((https?://[^)]+)\)'
    urls = set()

    for match in re.finditer(url_pattern, report):
        url = match.group(1)
        urls.add(url)

    return urls


def run_simple_pipeline(query: str) -> Generator[bytes, None, Dict[str, Any]]:
    """Run a simple pipeline without graph complexity."""
    try:
        # Initialize state
        state = SearchState(original_query=query)

        logger.info(f"Using query: {state.original_query}")

        # Step 0: Generate search query
        logger.info("Step 0: Generating search query...")
        generate_search_query_uuid = str(uuid.uuid4())
        yield to_chunk_data(wrap_step_start(generate_search_query_uuid, "Generating search query"))
        try:
            llm = init_reasoning_llm()
            # Use a simple prompt to rewrite query in search engine format
            prompt = PromptTemplate.from_template("""
Rewrite this query into a search engine friendly format and return as JSON with format:
{{
    "search_query": "<rewritten query>"
}}

Query: {query}
JSON response:
""")
            chain = prompt | llm
            response = chain.invoke({"query": query})
            data = repair_json(response.content, return_objects=True)
            search_query = data["search_query"]
            logger.info(f"  Generated search query: {search_query}")
            yield to_chunk_data(wrap_step_finish(generate_search_query_uuid, "Finished"))
        except Exception as e:
            logger.error(f"  Error generating search query: {str(e)}", exc_info=True)
            yield to_chunk_data(wrap_step_finish(generate_search_query_uuid, f"An error occurred", str(e), is_error=True))
            search_query = query

        retrievers = get_retriever_from_env()

        web_search_uuid = str(uuid.uuid4())

        # Step 1: Tavily Search
        if Retriever.TAVILY in retrievers:
            logger.info("Step 1: Performing Tavily web search...")
            yield to_chunk_data(wrap_step_start(web_search_uuid, "Performing web search"))
            temp_state = SearchState(
                original_query=search_query  # Use the current query as the original query for this temp state
            )
            try:
                temp_state = tavily_search_agent(temp_state)
                logger.info(f"  Found {len(temp_state.tavily_results)} results")
                yield to_chunk_data(wrap_step_finish(web_search_uuid, f"Found {len(temp_state.tavily_results)} results from Tavily search"))
            except Exception as e:
                logger.error(f"  Error in Tavily search: {str(e)}", exc_info=True)
                temp_state.tavily_results = []
                yield to_chunk_data(wrap_step_finish(web_search_uuid, f"Tavily search failed", str(e), is_error=True))
            state.tavily_results = temp_state.tavily_results
        else:
            logger.info("Step 1: Tavily search skipped")

        # Step 2: Brave Search
        if Retriever.BRAVE in retrievers:
            logger.info("Step 2: Performing Brave web search...")

            temp_state = SearchState(
                original_query=search_query  # Use the current query as the original query for this temp state
            )
            try:
                temp_state = brave_search_agent(temp_state, max_results=5, use_ai_snippets=False)
                logger.info(f"  Found {len(temp_state.brave_results)} results")
                yield to_chunk_data(wrap_step_finish(web_search_uuid, f"Found {len(temp_state.brave_results)} results from Brave search"))
            except Exception as e:
                logger.error(f"  Error in Brave search: {str(e)}", exc_info=True)
                temp_state.brave_results = []
                yield to_chunk_data(wrap_step_finish(web_search_uuid, f"Brave search failed", str(e), is_error=True))
            state.brave_results = temp_state.brave_results
        else:
            logger.info("Step 2: Brave search skipped")

        # Step 3: Exa search
        if Retriever.EXA in retrievers:
            logger.info("Step 3: Performing Exa search...")
            # Use the current query as the original query for this temp state)
            temp_state = SearchState(
                original_query=search_query,
            )
            try:
                temp_state = exa_search_agent(temp_state)
                logger.info(f"  Found {len(temp_state.exa_results)} results")
                yield to_chunk_data(
                    wrap_step_finish(
                        web_search_uuid,
                        (
                            f"Found {len(temp_state.exa_results)} results "
                            "from Exa search"
                        ),
                    ),
                )
            except Exception as e:
                logger.error(f"Error in Exa search: {str(e)}", exc_info=True)
                temp_state.exa_results = []
                yield to_chunk_data(
                    wrap_step_finish(
                        web_search_uuid,
                        "Exa search failed",
                        str(e),
                        is_error=True,
                    ),
                )
            state.exa_results = temp_state.exa_results
        else:
            logger.info("Step 3: Exa search skipped")
            state.exa_results = []

        # Combine search results
        state.search_results = (
            state.tavily_results
            + state.brave_results
            + state.exa_results
        )
        logger.info(
            f"Combined {len(state.search_results)} total search results",
        )

        # Step 4: FAISS Indexing (semantic search)
        logger.info("Step 4: Performing semantic search...")
        retrieving_search_result_uuid = str(uuid.uuid4())
        yield to_chunk_data(wrap_step_start(retrieving_search_result_uuid, "Retrieving relevant web search results"))
        try:
            state = faiss_indexing_agent(state)
            logger.info(f"  Found {len(state.faiss_results)} semantically relevant results")
            yield to_chunk_data(wrap_step_finish(retrieving_search_result_uuid, f"Found {len(state.faiss_results)} semantically relevant results"))
        except Exception as e:
            logger.error(f"  Error in semantic search: {str(e)}", exc_info=True)
            state.faiss_results = []
            yield to_chunk_data(wrap_step_finish(retrieving_search_result_uuid, f"Retrieving semantically relevant results failed", str(e), is_error=True))

        # Step 5: BM25 Search (keyword search)
        logger.info("Step 5: Performing keyword search...")

        try:
            state = bm25_search_agent(state)
            logger.info(f"  Found {len(state.bm25_results)} keyword relevant results")
            logger.info(f"  Combined {len(state.combined_results)} total relevant results")
            yield to_chunk_data(wrap_step_finish(retrieving_search_result_uuid, f"Found {len(state.bm25_results)} keyword relevant results"))
        except Exception as e:
            logger.error(f"  Error in keyword search: {str(e)}", exc_info=True)
            state.bm25_results = []
            # Ensure we have combined results even if BM25 fails
            if not state.combined_results:
                state.combined_results = state.faiss_results + state.search_results
            yield to_chunk_data(wrap_step_finish(retrieving_search_result_uuid, f"Retrieving keyword relevant results failed", str(e), is_error=True))

        # Step 5: Information Extraction
        logger.info("Step 5: Extracting information from results...")
        information_extraction_uuid = str(uuid.uuid4())
        yield to_chunk_data(wrap_step_start(information_extraction_uuid, "Extracting information from results"))
        try:
            state = yield from information_extraction_agent(state)
            logger.info(f"  Extracted information from {len(state.combined_results)} results")
            yield to_chunk_data(wrap_step_finish(information_extraction_uuid, f"Extracted information from {len(state.combined_results)} results"))
        except Exception as e:
            logger.error(f"  Error in information extraction: {str(e)}", exc_info=True)
            yield to_chunk_data(wrap_step_finish(information_extraction_uuid, f"An error occurred", str(e), is_error=True))

        # Step 6: Fact Checking
        logger.info("Step 6: Verifying extracted information...")
        fact_checking_uuid = str(uuid.uuid4())
        yield to_chunk_data(wrap_step_start(fact_checking_uuid, "Verifying extracted information"))
        try:
            state = yield from fact_checking_agent(state)
            logger.info(f"  Verified {len(state.verified_information['verified'])} statements")
            yield to_chunk_data(wrap_step_finish(fact_checking_uuid, 
                f"Verified {len(state.verified_information['verified'])} statements, "
                f"found {len(state.verified_information['contradicted'])} contradictions, "
                f"and {len(state.verified_information['unverified'])} unverified statements"
            ))
        except Exception as e:
            logger.error(f"  Error in fact checking: {str(e)}", exc_info=True)
            yield to_chunk_data(wrap_step_finish(fact_checking_uuid, f"An error occurred", str(e), is_error=True))

        # Step 7: LLM Reasoning
        logger.info("Step 7: Generating answer...")
        reasoning_uuid = str(uuid.uuid4())
        yield to_chunk_data(wrap_step_start(reasoning_uuid, "Generating answer"))
        try:
            state = llama_reasoning_agent(state)
            logger.info(f"  Confidence: {state.confidence_score}")
            yield to_chunk_data(wrap_step_finish(reasoning_uuid, f"Generated answer"))
        except Exception as e:
            logger.error(f"  Error in reasoning: {str(e)}", exc_info=True)
            # Provide a fallback answer
            state.final_answer = "I'm sorry, but I couldn't generate a proper answer for your query due to a technical issue. Please try again with a different query."
            state.confidence_score = 0.1
            yield to_chunk_data(wrap_step_finish(reasoning_uuid, f"An error occurred", str(e), is_error=True))

        # Return the results
        sources = []
        if state.combined_results:
            # Only include sources whose URLs appear in the report
            for res in state.combined_results:
                sources.append({
                    "title": res.title,
                    "url": res.url
                })

        write_state_to_file(state)

        return {
            "original_query": state.original_query,
            "answer": state.final_answer,
            "confidence": state.confidence_score,
            "sources": sources,
            "has_error": False
        }

    except Exception as e:
        write_state_to_file(state)
        # Handle any unexpected errors
        logger.error(f"Unexpected error in pipeline: {str(e)}", exc_info=True)
        yield to_chunk_data(wrap_thought("Pipeline error", f"Unexpected error: {str(e)}"))
        return {
            "original_query": query,
            "answer": "An unexpected error occurred while processing your query. Please try again later.",
            "confidence": 0.0,
            "sources": [],
            "has_error": True
        }


def run_deep_search_pipeline(
    query: str,
    twitter_search_needed: bool = False,
    max_iterations: int = 3,
) -> Generator[bytes, None, dict[str, Any]]:
    """Run the multi-query, iterative deep search pipeline with reasoning agent."""
    try:
        # Initialize state
        state = SearchState(original_query=query)

        # Instead of using query_expansion_agent, let the deep_reasoning_agent handle initial query generation
        logger.info("Step 1: Initial reasoning to analyze query and generate search queries...")
        try:
            generate_initial_queries_uuid = str(uuid.uuid4())
            yield to_chunk_data(wrap_step_start(generate_initial_queries_uuid, "Generating initial search queries"))
            # Initial call to deep_reasoning_agent will generate the queries
            state = yield from deep_reasoning_agent(state, max_iterations)

            logger.info(f"  Generated {len(state.generated_queries)} initial search queries")
            queries_markdown = "\n".join([f"- {query}" for query in state.generated_queries])
            yield to_chunk_data(wrap_step_finish(generate_initial_queries_uuid, f"Generated {len(state.generated_queries)} initial search queries", queries_markdown))
        except Exception as e:
            logger.error(f"  Error in initial reasoning: {str(e)}", exc_info=True)
            yield to_chunk_data(wrap_step_finish(generate_initial_queries_uuid, f"An error occurred", str(e), is_error=True))
            # If reasoning fails, use just the original query
            state.generated_queries = [state.original_query]
            state.current_iteration = 1  # Ensure we don't skip the first iteration

        # Iterative search loop
        while not state.search_complete and state.current_iteration < max_iterations:
            iteration = state.current_iteration
            logger.info(f"Beginning search iteration {iteration}...")

            yield to_chunk_data(wrap_thought(f"Beginning search iteration {iteration}..."))

            # Reset results for this iteration but keep accumulated results
            previous_results = state.combined_results.copy() if state.combined_results else []
            state.tavily_results = []
            state.brave_results = []
            state.search_results = []
            state.faiss_results = []
            state.bm25_results = []
            state.combined_results = []

            # Process each query in this iteration
            for i, query in enumerate(state.generated_queries):
                logger.info(f"  Processing query {i+1}/{len(state.generated_queries)}: {query}")

                yield to_chunk_data(wrap_thought(f"Processing query {i+1}/{len(state.generated_queries)}: {query}"))

                # Create a temporary state for this query
                temp_state = SearchState(
                    original_query=query  # Use the current query as the original query for this temp state
                )

                # Step 2: Tavily Search for this query

                web_search_uuid = str(uuid.uuid4())
                yield to_chunk_data(wrap_step_start(web_search_uuid, "Performing web search"))

                retrievers = get_retriever_from_env()
                if Retriever.TAVILY in retrievers:
                    logger.info(f"    Performing Tavily web search...")
                    try:
                        temp_state = tavily_search_agent(temp_state)
                        # Tag results with the query that produced them
                        for result in temp_state.tavily_results:
                            result.query = query
                        logger.info(f"    Found {len(temp_state.tavily_results)} web results")
                        yield to_chunk_data(wrap_step_finish(web_search_uuid, f"Found {len(temp_state.tavily_results)} results from Tavily search"))
                    except Exception as e:
                        logger.error(f"    Error in Tavily search: {str(e)}", exc_info=True)
                        yield to_chunk_data(wrap_step_finish(web_search_uuid, f"Tavily search failed", str(e), is_error=True))
                else:
                    logger.info(f"    Tavily web search skipped")

                # Step 3: Brave Search for this query
                if Retriever.BRAVE in retrievers:
                    logger.info("Performing Brave web search...")
                    try:
                        temp_state = brave_search_agent(temp_state, use_ai_snippets=True)
                        # Tag results with the query that produced them
                        for result in temp_state.brave_results:
                            result.query = query
                        logger.info(f"Found {len(temp_state.brave_results)} web results")
                        yield to_chunk_data(wrap_step_finish(web_search_uuid, f"Found {len(temp_state.brave_results)} results from Brave search"))
                    except Exception as e:
                        logger.error(f"    Error in Brave search: {str(e)}", exc_info=True)
                        yield to_chunk_data(wrap_step_finish(web_search_uuid, f"Brave search failed", str(e), is_error=True))
                else:
                    logger.info(f"Brave web search skipped")

                # Step 4: Exa search for this query
                if Retriever.EXA in retrievers:
                    logger.info("Performing Exa search...")
                    try:
                        temp_state = exa_search_agent(temp_state)
                        # Tag results with the query that produced them
                        for result in temp_state.exa_results:
                            result.query = query
                        logger.info(
                            f"Found {len(temp_state.exa_results)} web results",
                        )
                        yield to_chunk_data(
                            wrap_step_finish(
                                web_search_uuid,
                                (
                                    f"Found {len(temp_state.exa_results)} "
                                    "results from Exa search"
                                ),
                            ),
                        )
                    except Exception as e:
                        logger.error(
                            f"Error in Exa search: {str(e)}",
                            exc_info=True,
                        )
                        yield to_chunk_data(
                            wrap_step_finish(
                                web_search_uuid,
                                "Exa search failed",
                                str(e),
                                is_error=True,
                            ),
                        )
                else:
                    logger.info("Exa web search skipped")
                    temp_state.exa_results = []

                # Step 5: Perform Twitter search if needed
                if twitter_search_needed:
                    logger.info("Performing Twitter search...")
                    try:
                        temp_state = exa_search_agent(
                            temp_state,
                            max_results=20,
                            field_to_update="exa_twitter_results",
                            domains=["x.com", "twitter.com"],
                        )
                        # Tag results with the query that produced them
                        for result in temp_state.exa_twitter_results:
                            result.query = query
                        logger.info(
                            f"Found {len(temp_state.exa_twitter_results)} Twitter results",  # noqa
                        )
                        yield to_chunk_data(
                            wrap_step_finish(
                                web_search_uuid,
                                f"Found {len(temp_state.exa_twitter_results)} from Twitter search",  # noqa
                            ),
                        )
                    except Exception as e:
                        logger.error(
                            f"Error in Twitter search: {str(e)}",
                            exc_info=True,
                        )
                        yield to_chunk_data(
                            wrap_step_finish(
                                web_search_uuid,
                                "Twitter search failed",
                                str(e),
                                is_error=True,
                            ),
                        )
                else:
                    logger.info("Twitter web search skipped")
                    temp_state.exa_results = []

                # Combine search results
                temp_state.search_results = (
                    temp_state.tavily_results
                    + temp_state.brave_results
                    + temp_state.exa_results
                )
                logger.info(f"Combined {len(temp_state.search_results)} total search results")

                retrieving_search_result_uuid = str(uuid.uuid4())
                yield to_chunk_data(wrap_step_start(retrieving_search_result_uuid, "Retrieving relevant web search results"))

                # Step 5: FAISS Indexing (semantic search) for this query
                logger.info(f"    Performing semantic search...")
                try:
                    temp_state = faiss_indexing_agent(temp_state)
                    # Tag results with the query that produced them
                    for result in temp_state.faiss_results:
                        result.query = query
                    logger.info(f"    Found {len(temp_state.faiss_results)} semantic results")
                    yield to_chunk_data(wrap_step_finish(retrieving_search_result_uuid, f"Found {len(temp_state.faiss_results)} semantically relevant results"))
                except Exception as e:
                    logger.error(f"    Error in semantic search: {str(e)}", exc_info=True)
                    yield to_chunk_data(wrap_step_finish(retrieving_search_result_uuid, f"Retrieving semantically relevant results failed", str(e), is_error=True))

                # Step 6: BM25 Search (keyword search) for this query
                logger.info(f"    Performing keyword search...")
                try:
                    temp_state = bm25_search_agent(temp_state)
                    # Tag results with the query that produced them
                    for result in temp_state.bm25_results:
                        result.query = query
                    logger.info(f"    Found {len(temp_state.bm25_results)} keyword results")
                    yield to_chunk_data(wrap_step_finish(retrieving_search_result_uuid, f"Found {len(temp_state.bm25_results)} keyword relevant results"))
                except Exception as e:
                    logger.error(f"    Error in keyword search: {str(e)}", exc_info=True)
                    yield to_chunk_data(wrap_step_finish(retrieving_search_result_uuid, f"Retrieving keyword relevant results failed", str(e), is_error=True))

                # Step 6: Information Extraction for this query
                logger.info(f"    Extracting information from results...")
                try:
                    temp_state = yield from information_extraction_agent(temp_state)
                    logger.info(f"    Extracted information from {len(temp_state.combined_results)} results")
                except Exception as e:
                    logger.error(f"    Error in information extraction: {str(e)}", exc_info=True)

                # Step 7: Fact Checking for this query
                logger.info(f"    Verifying extracted information...")
                try:
                    temp_state = yield from fact_checking_agent(temp_state)
                    logger.info(f"    Verified {len(temp_state.verified_information['verified'])} statements")
                except Exception as e:
                    logger.error(f"    Error in fact checking: {str(e)}", exc_info=True)

                # Collect results from this query
                state.tavily_results.extend(temp_state.tavily_results)
                state.brave_results.extend(temp_state.brave_results)
                state.search_results.extend(temp_state.search_results)
                state.faiss_results.extend(temp_state.faiss_results)
                state.bm25_results.extend(temp_state.bm25_results)
                if temp_state.combined_results:
                    state.combined_results.extend(temp_state.combined_results)
                # Merge verified information
                for category in ["verified", "contradicted", "unverified"]:
                    state.verified_information[category].extend(temp_state.verified_information[category])

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
                # Combine FAISS and search results
                state.combined_results = state.faiss_results + state.search_results

            # Deduplicate combined results by URL
            if state.combined_results:
                unique_results = {}
                for result in state.combined_results:
                    # Keep the highest scoring result for each URL
                    key = result.url + "\n" + result.content
                    if key not in unique_results or (result.score is not None and
                        (unique_results[key].score is None or result.score > unique_results[key].score)):
                        unique_results[key] = result

                state.combined_results = list(unique_results.values())
                logger.info(f"  Deduplicated to {len(state.combined_results)} unique results")

            # Step 8: Deep Reasoning - analyze results and decide whether to continue
            logger.info(f"  Analyzing search results and determining next steps...")
            analyze_results_uuid = str(uuid.uuid4())
            yield to_chunk_data(wrap_step_start(analyze_results_uuid, "Analyzing search results and determining next steps"))
            try:
                state = yield from deep_reasoning_agent(state, max_iterations)
                logger.info(f"  Search complete: {state.search_complete}")
                if not state.search_complete:
                    joined_gaps = '\n'.join([f'- {gap}' for gap in state.knowledge_gaps])
                    logger.info(f"  Knowledge gaps identified:\n{joined_gaps}")
                    logger.info(f"  New queries generated: {len(state.generated_queries)}")
                    yield to_chunk_data(wrap_step_finish(analyze_results_uuid, f"Identified {len(state.knowledge_gaps)} knowledge gaps. Generated {len(state.generated_queries)} new queries"))
                else:
                    yield to_chunk_data(wrap_step_finish(analyze_results_uuid, "No knowledge gaps identified. Search completed."))
            except Exception as e:
                logger.error(f"  Error in deep reasoning: {str(e)}", exc_info=True)
                yield to_chunk_data(wrap_step_finish(analyze_results_uuid, f"An error occurred", str(e), is_error=True))
                # If reasoning fails, stop the search to avoid infinite loops
                state.search_complete = True
                state.final_answer = "I'm sorry, but I couldn't properly analyze the search results due to a technical issue. Please try again with a different query."
                state.confidence_score = 0.1

        # Prepare the response
        if not state.search_complete:
            # If we exited the loop due to max iterations, generate the final answer
            logger.info("Maximum iterations reached, generating final answer...")
            yield to_chunk_data(wrap_thought("Maximum iterations reached. Search completed."))
            try:
                from deepsearch.agents.deep_reasoning import generate_final_answer
                state = yield from generate_final_answer(state)
            except Exception as e:
                logger.error(f"Error generating final answer: {str(e)}", exc_info=True)
                state.final_answer = "I reached the maximum number of search iterations but couldn't generate a comprehensive answer. Here's what I found: " + "\n".join([f"- {point}" for point in state.key_points])
                state.confidence_score = 0.5

        # Return the results
        sources = []
        if state.combined_results:
            # Extract URLs from the final answer
            report_urls = extract_urls_from_report(state.final_answer)

            # Only include sources whose URLs appear in the report
            for res in state.combined_results:
                if res.url in report_urls:
                    sources.append({
                        "title": res.title,
                        "url": res.url
                    })

        # Extract components from the final answer if available
        answer = state.final_answer
        key_points = state.key_points if state.key_points else []
        detailed_notes = state.detailed_notes if state.detailed_notes else None

        write_state_to_file(state)

        return {
            "original_query": state.original_query,
            "generated_queries": state.generated_queries,
            "iterations": state.current_iteration,
            "answer": answer,
            "key_points": key_points,
            "detailed_notes": detailed_notes,
            "confidence": state.confidence_score,
            "sources": sources,
            "has_error": False
        }
    except Exception as e:
        write_state_to_file(state)
        # Handle any unexpected errors
        logger.error(f"Unexpected error in deep search pipeline: {str(e)}", exc_info=True)
        return {
            "original_query": query,
            "answer": "An unexpected error occurred while processing your query. Please try again later.",
            "confidence": 0.0,
            "sources": [],
            "has_error": True
        }

class GeneratorValue:
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        self.value = yield from self.gen
        return self.value


def prompt(messages: list[dict[str, str]], **kwargs) -> Generator[bytes, None, None]:
    assert len(messages) > 0, "received empty messages"

    conversation_summary = retry(get_conversation_summary)(messages)
    research_intent = retry(detect_research_intent)(conversation_summary, messages[-1]["content"])

    logger.info(f"Research intent: {research_intent.model_dump_json()}")

    if not research_intent.is_research_request:
        yield retry(reply_conversation)(conversation_summary, messages[-1]["content"])
        return

    query = research_intent.research_query
    twitter_search_needed = True

    logger.info("Always using deep search pipeline")
    res = yield from run_deep_search_pipeline(
        query,
        twitter_search_needed=twitter_search_needed,
    )

    final_resp = res["answer"]

    if len(res["sources"]) > 0:
        unique_results = {}
        for result in res["sources"]:
            key = result["url"]
            if key not in unique_results:
                unique_results[key] = result

        res["sources"] = list(unique_results.values())
        logger.info(f"  Deduplicated to {len(res['sources'])} unique sources")

        final_resp += "\n## References:\n"

        for item in res["sources"]:
            final_resp += "- [{title}]({url})\n".format(**item)

    yield final_resp
