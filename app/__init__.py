from enum import Enum
import eai_http_middleware # do not remove this

import os

os.environ['TAVILY_API_KEY'] = 'no-need'
os.environ['OPENAI_BASE_URL'] = os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL"))
os.environ['OPENAI_API_KEY'] = os.getenv("LLM_API_KEY", 'no-need')

from typing import Dict, Any, Generator, List
from deepsearch.models import SearchState
from deepsearch.agents import (
    tavily_search_agent,
    faiss_indexing_agent,
    bm25_search_agent,
    llama_reasoning_agent,
    query_expansion_agent,
    deep_reasoning_agent,
    brave_search_agent
)
from deepsearch.agents.deep_reasoning import init_reasoning_llm

from json_repair import repair_json
import json
import logging
logger = logging.getLogger(__name__)

from langchain.prompts import PromptTemplate
from deepsearch.utils import to_chunk_data, wrap_thought

class Retriever(Enum):
    TAVILY = "tavily"
    BRAVE = "brave"


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
        else:
            raise ValueError(f"Invalid retriever: {retriever}")
    return retrievers


def run_simple_pipeline(query: str) -> Generator[bytes, None, Dict[str, Any]]:
    """Run a simple pipeline without graph complexity."""
    try:
        # Initialize state
        state = SearchState(original_query=query)

        logger.info(f"Using query: {state.original_query}")
        yield to_chunk_data(wrap_thought("Initializing simple search pipeline"))

        # Step 0: Generate search query for Brave
        logger.info("Step 0: Generating Brave search query...")
        yield to_chunk_data(wrap_thought("Generating Brave search query..."))
        try:
            llm = init_reasoning_llm()
            # Use a simple prompt to rewrite query in search engine format
            prompt = PromptTemplate.from_template(
                "Rewrite this query into a search engine friendly format"
                "Query: {query}\n"
                "Rewritten query:"
            )
            chain = prompt | llm
            response = chain.invoke({"query": query})
            search_query = response.content
            logger.info(f"  Generated search query: {search_query}")
            yield to_chunk_data(wrap_thought("Search query generation complete", f"Generated query: {search_query}"))
        except Exception as e:
            logger.error(f"  Error generating search query: {str(e)}", exc_info=True)
            search_query = query
            yield to_chunk_data(wrap_thought("Search query generation error", f"Using original query: {query}"))

        retrievers = get_retriever_from_env()

        # Step 1: Tavily Search
        if Retriever.TAVILY in retrievers:
            logger.info("Step 1: Performing Tavily web search...")
            yield to_chunk_data(wrap_thought("Performing Tavily web search..."))
            temp_state = SearchState(
                original_query=search_query  # Use the current query as the original query for this temp state
            )
            try:
                temp_state = yield from tavily_search_agent(temp_state)
                logger.info(f"  Found {len(temp_state.tavily_results)} results")
                yield to_chunk_data(wrap_thought("Tavily search complete", f"Found {len(temp_state.tavily_results)} results"))
            except Exception as e:
                logger.error(f"  Error in Tavily search: {str(e)}", exc_info=True)
                temp_state.tavily_results = []
                yield to_chunk_data(wrap_thought("Tavily search error", str(e)))
            state.tavily_results = temp_state.tavily_results
        else:
            logger.info("Step 1: Tavily search skipped")
            yield to_chunk_data(wrap_thought("Tavily search skipped"))

        # Step 2: Brave Search
        if Retriever.BRAVE in retrievers:
            logger.info("Step 2: Performing Brave web search...")
            yield to_chunk_data(wrap_thought("Performing Brave web search..."))
            temp_state = SearchState(
                original_query=search_query  # Use the current query as the original query for this temp state
            )
            try:
                temp_state = yield from brave_search_agent(temp_state, max_results=5, use_ai_snippets=False)
                logger.info(f"  Found {len(temp_state.brave_results)} results")
                yield to_chunk_data(wrap_thought("Brave search complete", f"Found {len(temp_state.brave_results)} results"))
            except Exception as e:
                logger.error(f"  Error in Brave search: {str(e)}", exc_info=True)
                temp_state.brave_results = []
                yield to_chunk_data(wrap_thought("Brave search error", str(e)))
            state.brave_results = temp_state.brave_results
        else:
            logger.info("Step 2: Brave search skipped")
            yield to_chunk_data(wrap_thought("Brave search skipped"))

        # Combine search results
        state.search_results = state.tavily_results + state.brave_results
        logger.info(f"  Combined {len(state.search_results)} total search results")
        yield to_chunk_data(wrap_thought("Search results combined", f"Combined {len(state.search_results)} total search results"))

        # Step 3: FAISS Indexing (semantic search)
        logger.info("Step 3: Performing semantic search...")
        yield to_chunk_data(wrap_thought("Performing semantic search..."))
        try:
            state = yield from faiss_indexing_agent(state)
            logger.info(f"  Found {len(state.faiss_results)} semantically relevant results")
            yield to_chunk_data(wrap_thought("Semantic search complete", f"Found {len(state.faiss_results)} semantically relevant results"))
        except Exception as e:
            logger.error(f"  Error in semantic search: {str(e)}", exc_info=True)
            state.faiss_results = []
            yield to_chunk_data(wrap_thought("Semantic search error", str(e)))

        # Step 4: BM25 Search (keyword search)
        logger.info("Step 4: Performing keyword search...")
        yield to_chunk_data(wrap_thought("Performing keyword search..."))
        try:
            state = yield from bm25_search_agent(state)
            logger.info(f"  Found {len(state.bm25_results)} keyword relevant results")
            logger.info(f"  Combined {len(state.combined_results)} total relevant results")
            yield to_chunk_data(wrap_thought("Keyword search complete", f"Found {len(state.bm25_results)} keyword relevant results and {len(state.combined_results)} total results"))
        except Exception as e:
            logger.error(f"  Error in keyword search: {str(e)}", exc_info=True)
            state.bm25_results = []
            # Ensure we have combined results even if BM25 fails
            if not state.combined_results:
                state.combined_results = state.faiss_results + state.search_results
            yield to_chunk_data(wrap_thought("Keyword search error", str(e)))

        # Step 5: LLM Reasoning
        logger.info("Step 5: Generating answer...")
        yield to_chunk_data(wrap_thought("Generating answer..."))
        try:
            state = yield from llama_reasoning_agent(state)
            logger.info(f"  Confidence: {state.confidence_score}")
            yield to_chunk_data(wrap_thought("Answer generation complete", f"Generated answer with confidence: {state.confidence_score}"))
        except Exception as e:
            logger.error(f"  Error in reasoning: {str(e)}", exc_info=True)
            # Provide a fallback answer
            state.final_answer = "I'm sorry, but I couldn't generate a proper answer for your query due to a technical issue. Please try again with a different query."
            state.confidence_score = 0.1
            yield to_chunk_data(wrap_thought("Reasoning error", str(e)))

        # Return the results
        sources = []
        if state.combined_results:
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
            "sources": sources[:5],  # Limit to 5 sources,
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

def run_deep_search_pipeline(query: str, max_iterations: int = 3) -> Generator[bytes, None, Dict[str, Any]]:
    """Run the multi-query, iterative deep search pipeline with reasoning agent."""
    try:
        # Initialize state
        state = SearchState(original_query=query)

        # Instead of using query_expansion_agent, let the deep_reasoning_agent handle initial query generation
        logger.info("Step 1: Initial reasoning to analyze query and generate search queries...")
        try:
            # Initial call to deep_reasoning_agent will generate the queries
            state = yield from deep_reasoning_agent(state, max_iterations)
            logger.info(f"  Generated {len(state.generated_queries)} initial search queries")
        except Exception as e:
            logger.error(f"  Error in initial reasoning: {str(e)}", exc_info=True)
            # If reasoning fails, use just the original query
            state.generated_queries = [state.original_query]
            state.current_iteration = 1  # Ensure we don't skip the first iteration

        # Iterative search loop
        while not state.search_complete and state.current_iteration < max_iterations:
            iteration = state.current_iteration
            logger.info(f"Beginning search iteration {iteration}...")

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

                # Create a temporary state for this query
                temp_state = SearchState(
                    original_query=query  # Use the current query as the original query for this temp state
                )

                # Step 2: Tavily Search for this query
                retrievers = get_retriever_from_env()
                if Retriever.TAVILY in retrievers:
                    logger.info(f"    Performing Tavily web search...")
                    try:
                        temp_state = yield from tavily_search_agent(temp_state)
                        # Tag results with the query that produced them
                        for result in temp_state.tavily_results:
                            result.query = query
                        logger.info(f"    Found {len(temp_state.tavily_results)} web results")
                    except Exception as e:
                        logger.error(f"    Error in Tavily search: {str(e)}", exc_info=True)
                else:
                    logger.info(f"    Tavily web search skipped")

                # Step 3: Brave Search for this query
                if Retriever.BRAVE in retrievers:
                    logger.info(f"    Performing Brave web search...")
                    try:
                        temp_state = yield from brave_search_agent(temp_state, use_ai_snippets=True)
                        # Tag results with the query that produced them
                        for result in temp_state.brave_results:
                            result.query = query
                        logger.info(f"    Found {len(temp_state.brave_results)} web results")
                    except Exception as e:
                        logger.error(f"    Error in Brave search: {str(e)}", exc_info=True)
                else:
                    logger.info(f"    Brave web search skipped")

                # Combine search results
                temp_state.search_results = temp_state.tavily_results + temp_state.brave_results
                logger.info(f"    Combined {len(temp_state.search_results)} total search results")

                # Step 4: FAISS Indexing (semantic search) for this query
                logger.info(f"    Performing semantic search...")
                try:
                    temp_state = yield from faiss_indexing_agent(temp_state)
                    # Tag results with the query that produced them
                    for result in temp_state.faiss_results:
                        result.query = query
                    logger.info(f"    Found {len(temp_state.faiss_results)} semantic results")
                except Exception as e:
                    logger.error(f"    Error in semantic search: {str(e)}", exc_info=True)

                # Step 5: BM25 Search (keyword search) for this query
                logger.info(f"    Performing keyword search...")
                try:
                    temp_state = yield from bm25_search_agent(temp_state)
                    # Tag results with the query that produced them
                    for result in temp_state.bm25_results:
                        result.query = query
                    logger.info(f"    Found {len(temp_state.bm25_results)} keyword results")
                except Exception as e:
                    logger.error(f"    Error in keyword search: {str(e)}", exc_info=True)

                # Collect results from this query
                state.tavily_results.extend(temp_state.tavily_results)
                state.brave_results.extend(temp_state.brave_results)
                state.search_results.extend(temp_state.search_results)
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

            # Step 6: Deep Reasoning - analyze results and decide whether to continue
            logger.info(f"  Analyzing search results and determining next steps...")
            try:
                state = yield from deep_reasoning_agent(state, max_iterations)
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
                state = yield from generate_final_answer(state)
            except Exception as e:
                logger.error(f"Error generating final answer: {str(e)}", exc_info=True)
                state.final_answer = "I reached the maximum number of search iterations but couldn't generate a comprehensive answer. Here's what I found: " + "\n".join([f"- {point}" for point in state.key_points])
                state.confidence_score = 0.5

        # Return the results
        sources = []
        if state.combined_results:
            for res in state.combined_results:  # Limit to 5 sources
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
    
def detect_query_complexity(query: str) -> bool:
    """
    Analyze the query to determine if it requires a simple or complex search pipeline.
    
    Args:
        query: The user's query string
        
    Returns:
        bool: True if the query is complex and requires deep search, False if it's simple
    """
    # Initialize LLM for complexity analysis
    llm = init_reasoning_llm()

    # Create prompt for complexity analysis
    complexity_prompt = """Analyze the following query and determine if it requires a simple or complex search approach.

QUERY: {query}

Consider the following factors:
1. Does the query ask for a simple fact or definition that can be answered in a few sentences?
2. Does the query require gathering and synthesizing information from multiple sources?
3. Is the query open-ended or exploratory in nature?
4. Does the query require comparing different perspectives or analyzing trends?
5. Would answering the query benefit from multiple search iterations?
6. Does the query involve temporal aspects or need recent/current information?
7. Does it require domain expertise or technical knowledge?
8. Are there multiple sub-questions within the main query?

Respond with a JSON object in this format:
{{
    "complexity": "simple" or "complex",
    "reasoning": ["reason1", "reason2", ...],
    "confidence": 0.0 to 1.0
}}
"""

    # Create the prompt template
    prompt_template = PromptTemplate(
        input_variables=["query"],
        template=complexity_prompt
    )

    # Create the chain
    chain = prompt_template | llm

    # Get the response
    response = chain.invoke({"query": query})
    
    # Extract the content if it's a message object
    response_text = response.content if hasattr(response, 'content') else response
    
    try:
        analysis = json.loads(repair_json(response_text))
        
        logger.info(f"Query complexity analysis: {analysis}")
        
        # Return False for simple queries, True for complex ones
        return analysis["complexity"].strip().lower() == "complex"
        
    except Exception as e:
        logger.error(f"Error parsing complexity analysis: {str(e)}")
        # Default to treating as complex if parsing fails
        return True

def prompt(messages: list[dict[str, str]], **kwargs) -> str:
    assert len(messages) > 0, "received empty messages"
    query = messages[-1]['content']

    # Detect query complexity
    logger.info("Analyzing query complexity...")
    is_complex = detect_query_complexity(query)
    logger.info(f"Query complexity: {'complex' if is_complex else 'simple'}")

    # Choose appropriate pipeline based on complexity
    if is_complex:
        logger.info("Using deep search pipeline for complex query")
        gen = GeneratorValue(run_deep_search_pipeline(query))
    else:
        logger.info("Using simple pipeline for straightforward query")
        gen = GeneratorValue(run_simple_pipeline(query))

    # Process the pipeline
    for chunk in gen:
        print(chunk)
    res: Dict = gen.value

    sep = "-" * 30
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

    return final_resp
