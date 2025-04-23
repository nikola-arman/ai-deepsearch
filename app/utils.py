import os

os.environ['TAVILY_API_KEY'] = 'no-need'
os.environ['OPENAI_BASE_URL'] = os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL"))
os.environ['OPENAI_API_KEY'] = os.getenv("LLM_API_KEY", 'no-need')

from deepsearch.agents.deep_reasoning import init_reasoning_llm

from json_repair import repair_json
import json
import logging
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


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

        print(f"Query complexity analysis: {analysis}")

        # Return False for simple queries, True for complex ones
        return analysis["complexity"].strip().lower() == "complex"

    except Exception as e:
        logger.error(f"Error parsing complexity analysis: {str(e)}")
        # Default to treating as complex if parsing fails
        return True


if __name__ == "__main__":
    query = "singer Ado"
    res = detect_query_complexity(query)
    print(res)
