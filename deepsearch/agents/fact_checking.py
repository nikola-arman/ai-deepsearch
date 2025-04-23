from typing import Generator, List, Dict, Any, Set
import logging
from deepsearch.models import SearchState, SearchResult
from deepsearch.utils import to_chunk_data, wrap_thought
from deepsearch.agents.deep_reasoning import init_reasoning_llm
from langchain.prompts import PromptTemplate
from collections import defaultdict
from json_repair import repair_json
# Set up logging
logger = logging.getLogger("deepsearch.factcheck")

def group_similar_statements(statements: List[str]) -> Dict[str, List[str]]:
    """
    Group similar statements together based on their semantic meaning.
    
    Args:
        statements: List of statements to group
        
    Returns:
        Dictionary mapping representative statements to lists of similar statements
    """
    try:
        # Initialize LLM for semantic grouping
        llm = init_reasoning_llm()

        # Create prompt for semantic grouping
        grouping_prompt = """Group the following statements based on their semantic meaning.
Statements that convey the same or very similar information should be grouped together.

STATEMENTS:
{statements}

For each group, provide:
1. A representative statement that best captures the meaning of the group
2. All statements that belong to this group

Format the response as a JSON object where:
- Keys are the representative statements
- Values are lists of all statements in that group

Example format:
{{
    "representative statement 1": ["statement1", "statement2", ...],
    "representative statement 2": ["statement3", "statement4", ...]
}}
"""

        # Create the prompt template
        prompt_template = PromptTemplate(
            input_variables=["statements"],
            template=grouping_prompt
        )

        # Create the chain
        chain = prompt_template | llm

        print("grouping_prompt:", prompt_template.invoke({
            "statements": "\n".join(f"- {s}" for s in statements)
        }))

        # Get the response
        response = chain.invoke({"statements": "\n".join(f"- {s}" for s in statements)})
        
        # Extract the content if it's a message object
        response_text = response.content if hasattr(response, 'content') else response
        
        # Parse the response as JSON
        groups = repair_json(response_text, return_objects=True)  # Using eval since we trust the LLM output
        
        return groups

    except Exception as e:
        logger.error(f"Error grouping statements: {str(e)}", exc_info=True)
        # Return each statement as its own group if grouping fails
        return {s: [s] for s in statements}

def verify_statement_group(representative: str, statements: List[str], sources: List[SearchResult]) -> Dict[str, Any]:
    """
    Verify a group of similar statements against their sources.
    
    Args:
        representative: The representative statement for the group
        statements: All statements in the group
        sources: The search results that contain these statements
        
    Returns:
        Dictionary containing verification results
    """
    try:
        # Initialize LLM for verification
        llm = init_reasoning_llm()

        # Create prompt for verification
        verification_prompt = """Verify the following statement against multiple sources.

STATEMENT: {representative}

SOURCES:
{sources}

For each source, analyze:
1. Whether the source supports the statement
2. The level of confidence in the support (high/medium/low)
3. Any contradictions or conflicting information
4. The source's credibility

Format the response as a JSON object with these fields:
{{
    "verification_status": "verified" or "contradicted" or "unverified",
    "confidence": "high" or "medium" or "low",
    "supporting_sources": ["url1", "url2", ...],
    "contradicting_sources": ["url3", "url4", ...],
    "notes": "Additional observations about the verification"
}}
"""

        # Format sources for the prompt
        formatted_sources = []
        for source in sources:
            formatted_sources.append(f"URL: {source.url}")
            formatted_sources.append(f"Content: {source.content}")
            formatted_sources.append("---")

        # Create the prompt template
        prompt_template = PromptTemplate(
            input_variables=["representative", "sources"],
            template=verification_prompt
        )

        print("verification_prompt:", prompt_template.invoke({
            "representative": representative,
            "sources": "\n".join(formatted_sources)    
        }))

        # Create the chain
        chain = prompt_template | llm

        # Get the response
        response = chain.invoke({
            "representative": representative,
            "sources": "\n".join(formatted_sources)
        })
        
        # Extract the content if it's a message object
        response_text = response.content if hasattr(response, 'content') else response
        
        # Parse the response as JSON
        verification = repair_json(response_text, return_objects=True)  # Using eval since we trust the LLM output
        
        return verification

    except Exception as e:
        logger.error(f"Error verifying statement: {str(e)}", exc_info=True)
        return {
            "verification_status": "error",
            "confidence": "low",
            "supporting_sources": [],
            "contradicting_sources": [],
            "notes": f"Error during verification: {str(e)}"
        }

def fact_checking_agent(state: SearchState) -> Generator[bytes, None, SearchState]:
    """
    Cross-validates extracted information across different sources.

    Args:
        state: The current search state with extracted information

    Returns:
        Updated state with verification results
    """
    # Check if we have results to analyze
    if not state.combined_results:
        return state

    try:
        # Collect all extracted information
        all_statements = []
        for result in state.combined_results:
            if result.extracted_information:
                all_statements.extend(result.extracted_information)

        if not all_statements:
            return state

        # Group similar statements
        statement_groups = group_similar_statements(all_statements)
        logger.info(f"Grouped statements into {len(statement_groups)} groups")

        # Initialize verification results
        state.verified_information = {
            "verified": [],
            "contradicted": [],
            "unverified": []
        }

        print("statement_groups:", statement_groups)

        # Verify each group
        for i, (representative, statements) in enumerate(statement_groups.items()):
            yield to_chunk_data(
                wrap_thought(
                    "Fact checking agent: Verifying group",
                    f"Verifying group {i+1}/{len(statement_groups)}: {representative[:100]}..."
                )
            )

            # Find sources that contain these statements
            relevant_sources = []
            for result in state.combined_results:
                if result.extracted_information and any(s in result.extracted_information for s in statements):
                    relevant_sources.append(result)

            # Verify the statement group
            verification = verify_statement_group(representative, statements, relevant_sources)

            print("verification:", verification)

            # Store verification results
            if verification["verification_status"] == "verified":
                state.verified_information["verified"].append({
                    "statement": representative,
                    "confidence": verification["confidence"],
                    "sources": verification["supporting_sources"],
                    "notes": verification["notes"]
                })
            elif verification["verification_status"] == "contradicted":
                state.verified_information["contradicted"].append({
                    "statement": representative,
                    "confidence": verification["confidence"],
                    "supporting_sources": verification["supporting_sources"],
                    "contradicting_sources": verification["contradicting_sources"],
                    "notes": verification["notes"]
                })
            else:
                state.verified_information["unverified"].append({
                    "statement": representative,
                    "confidence": verification["confidence"],
                    "sources": verification["supporting_sources"],
                    "notes": verification["notes"]
                })

        yield to_chunk_data(
            wrap_thought(
                "Fact checking agent: Complete",
                f"Verified {len(state.verified_information['verified'])} statements, "
                f"found {len(state.verified_information['contradicted'])} contradictions, "
                f"and {len(state.verified_information['unverified'])} unverified statements"
            )
        )

    except Exception as e:
        logger.error(f"Error in fact checking agent: {str(e)}", exc_info=True)
        yield to_chunk_data(
            wrap_thought(
                "Fact checking agent: Error",
                f"Error occurred during fact checking: {str(e)}"
            )
        )
        state.verified_information = {
            "verified": [],
            "contradicted": [],
            "unverified": []
        }

    return state 