from typing import Generator, List, Dict, Any
import logging
from deepsearch.models import SearchState, SearchResult
from deepsearch.utils import to_chunk_data, wrap_thought
from deepsearch.agents.deep_reasoning import init_reasoning_llm
from langchain.prompts import PromptTemplate
from json_repair import repair_json

# Set up logging
logger = logging.getLogger("deepsearch.extraction")

def extract_information_from_result(result: SearchResult, query: str) -> List[str]:
    """
    Extract structured information from a single search result.
    
    Args:
        result: The search result to analyze
        
    Returns:
        Dictionary containing extracted information
    """
    try:
        # Initialize LLM for information extraction
        llm = init_reasoning_llm()

        # Create prompt for information extraction
        extraction_prompt = """Analyze the following search result and extract key pieces of information.

TITLE: {title}
URL: {url}
CONTENT: {content}

Extract the following information that is related to the query: {query}
1. Key facts and figures
2. Any statistics or numerical data
3. Main arguments or points made
4. Any quoted statements

Instructions:
- Phrase each item as a complete, standalone sentence that clearly conveys the information without requiring additional context. Use complete phrases when referring to specific objects or concepts.
- Do not include duplicate or overly similar statements.
- Only include information that appears explicitly in the content and is relevant to the query. Do not include information about the source itself.
- Quotes must be word-for-word and properly attributed.
- If no relevant information is found, return an empty list.

Format the response as a JSON object with these fields:
{{    
    "information": ["information1", "information2", ...],
}}
"""

        # Create the prompt template
        prompt_template = PromptTemplate(
            input_variables=["title", "url", "content", "query"],
            template=extraction_prompt
        )

        # Create the chain
        chain = prompt_template | llm

        # Get the response
        response = chain.invoke({
            "title": result.title,
            "url": result.url,
            "content": result.content,
            "query": query
        })
        
        # Extract the content if it's a message object
        response_text = response.content if hasattr(response, 'content') else response

        print("response_text:", response_text)
        
        # Parse the response as JSON
        extracted_info = repair_json(response_text, return_objects=True)  # Using eval since we trust the LLM output

        extracted_all_info = []
        for key, value in extracted_info.items():
            extracted_all_info.extend(value)
        
        return extracted_all_info

    except Exception as e:
        logger.error(f"Error extracting information from result: {str(e)}", exc_info=True)
        return []

def information_extraction_agent(state: SearchState) -> Generator[bytes, None, SearchState]:
    """
    Extracts structured information from all search results.

    Args:
        state: The current search state with search results

    Returns:
        Updated state with extracted information
    """
    # Initialize extracted information dictionary

    # Check if we have results to analyze
    if not state.combined_results and not state.search_results:
        yield to_chunk_data(
            wrap_thought(
                "Information extraction agent: No results",
                "No search results available for information extraction"
            )
        )
        return state

    # Use combined results if available, otherwise use search results
    results_to_analyze = state.combined_results if state.combined_results else state.search_results

    yield to_chunk_data(
        wrap_thought(
            "Information extraction agent: Starting extraction",
            f"Analyzing {len(results_to_analyze)} search results"
        )
    )

    # Process each result
    for i, result in enumerate(results_to_analyze):
        try:
            yield to_chunk_data(
                wrap_thought(
                    "Information extraction agent: Processing result",
                    f"Extracting information from result {i+1}/{len(results_to_analyze)}"
                )
            )

            print("result:", result)

            # Extract information from the result
            extracted_info = extract_information_from_result(result, state.original_query)

            print("extracted_info:", extracted_info)
            result.extracted_information = extracted_info
        except Exception as e:
            logger.error(f"Error in information extraction agent: {str(e)}", exc_info=True)
            result.extracted_information = []
            yield to_chunk_data(
                wrap_thought(
                    "Information extraction agent: Error",
                    f"Error occurred during information extraction: {str(e)}"
                )
            )

    return state
