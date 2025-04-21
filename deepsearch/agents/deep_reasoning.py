from typing import Generator, List, Dict, Any, Tuple
import os
import json
import logging
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import re
import datetime  # Add import for datetime module

from deepsearch.models import SearchState, SearchResult
from deepsearch.utils import to_chunk_data, wrap_thought

# Set up logging
logger = logging.getLogger("deepsearch.deep_reasoning")

# Load environment variables
load_dotenv()

# Get the OpenAI-compatible API base URL and API key
openai_api_base = os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1")
openai_api_key = os.environ.get("LLM_API_KEY", "not-needed")

# Define the prompt template for analysis and reasoning
REASONING_TEMPLATE = """You are an expert research analyst and reasoning agent. Your task is to analyze search results,
identify relevant information, and determine if further searches are needed.

ORIGINAL QUERY: {original_query}

CURRENT SEARCH ITERATION: {iteration}

CURRENT DATE: {current_date} (ISO 8601 format: YYYY-MM-DD)

SEARCH RESULTS:
{search_results}

PREVIOUSLY IDENTIFIED KNOWLEDGE GAPS:
{previous_knowledge_gaps}

INSTRUCTIONS:
1. Analyze the search results carefully to extract key information related to the original query.
2. Identify any NEW knowledge gaps that require further searches. Do NOT repeat previously identified knowledge gaps.
3. Decide if the search process should continue or if we have sufficient information to answer the query.
4. If further searches are needed, generate specific new search queries to fill the NEW knowledge gaps.
5. IMPORTANT: Include the current year or time period in queries when relevant (especially for financial, news, trend, or stock queries) to ensure recency and up-to-date information.
6. When referring to time periods, use clear universal formats (e.g., "Q1 2024", "May 2024", "2024", etc.)
7. Format your response as a JSON object with the following structure:

{{
  "key_points": ["point 1", "point 2", "..."],
  "knowledge_gaps": ["gap 1", "gap 2", "..."],
  "new_queries": ["query 1", "query 2", "..."],
  "search_complete": true/false,
  "reasoning": "Your explanation of why the search is complete or needs to continue"
}}

CRITICAL: Your entire response MUST be a valid, parseable JSON object and nothing else. Do not include any text before or after the JSON object. Do not include any explanation, markdown formatting, or code blocks around the JSON. The response must start with '{{' and end with '}}' and contain only valid JSON.

If there are no knowledge gaps or the search should stop, return an empty array for "knowledge_gaps" and "new_queries"
and set "search_complete" to true.

IMPORTANT: If this is already iteration {max_iterations} or higher, set "search_complete" to true regardless of knowledge gaps.
"""

# Define the prompt template for final answer formulation
ANSWER_TEMPLATE = """You are an expert research analyst and outline creator. Your task is to create a well-structured outline for answering a query based on search results.

ORIGINAL QUERY: {original_query}

SEARCH CONTEXT:
{search_context}

SEARCH DETAILS:
{search_details}

INSTRUCTIONS:
Your task is to formulate an OUTLINE ONLY for a complete answer with three distinct sections:

1. KEY POINTS: List 5-7 bullet points that would be the most important findings and facts
2. DIRECT ANSWER: Provide a brief description of what should be covered in the direct answer section (2-3 paragraphs)
3. DETAILED NOTES: Create a comprehensive outline with:
   a. Main section headings (3-5 sections)
   b. For each section, provide 2-3 sub-points that should be covered
   c. Note any specific technical details, examples, or comparisons that should be included
   d. Suggest logical flow for presenting the information

IMPORTANT RULES:
1. ONLY include information that is directly supported by the search context
2. DO NOT make up or infer information not present in the search results
3. If information is missing or unclear, note it as a limitation rather than making assumptions
4. Clearly indicate which search results support each point using markdown hyperlinks
5. Use direct quotes from search results when appropriate
6. Maintain academic rigor and avoid speculation

Format your outline using proper markdown sections. THIS IS ONLY AN OUTLINE - do not write the full content.
Make the outline detailed enough that a content writer can easily expand it into a complete, informative answer.

The outline should follow this structure:
```
# OUTLINE: [Query Title]

## 1. KEY POINTS
- [Key point 1] ([Source Title](source_url))
- [Key point 2] ([Source Title](source_url))
...

## 2. DIRECT ANSWER
[Brief description of what the direct answer should cover, with source references]

## 3. DETAILED NOTES
### [Section Heading 1]
- [Subpoint 1] ([Source Title](source_url))
- [Subpoint 2] ([Source Title](source_url))
...

### [Section Heading 2]
- [Subpoint 1] ([Source Title](source_url))
- [Subpoint 2] ([Source Title](source_url))
...
```
"""

# Define the template for the writer agent that will expand the outline into full content
WRITER_TEMPLATE = """You are an expert content writer. Your task is to expand an outline into a comprehensive, detailed answer.

ORIGINAL QUERY: {original_query}

SEARCH CONTEXT:
{search_context}

OUTLINE:
{outline}

SEARCH DETAILS:
{search_details}

INSTRUCTIONS:
Transform the provided outline into a comprehensive, detailed answer that follows the exact structure of the outline.
For each section:
1. Expand bullet points into detailed paragraphs with rich information
2. Maintain the hierarchical structure from the outline
3. Include technical details, examples, and comparisons suggested in the outline
4. Ensure smooth transitions between sections
5. Use an authoritative, clear writing style

IMPORTANT RULES:
1. ONLY include information that is directly supported by the search context
2. DO NOT make up or infer information not present in the search results
3. If information is missing or unclear, note it as a limitation rather than making assumptions
4. Clearly cite sources for each piece of information using markdown hyperlinks: ([Source Title](source_url))
5. Use direct quotes from search results when appropriate
6. Maintain academic rigor and avoid speculation
7. If the search context is insufficient to answer a point, clearly state this limitation
8. Do not use phrases like "based on the search results" or "according to the information provided" - instead cite specific sources

Your expanded answer should be thorough, informative, and directly address the original query,
while carefully following the outline structure and maintaining strict adherence to the search context.
"""

# Define the prompt template for generating key points
KEY_POINTS_TEMPLATE = """You are an expert research analyst. Your task is to extract the most important key points from search results.

ORIGINAL QUERY: {original_query}

SEARCH CONTEXT:
{search_context}

SEARCH DETAILS:
{search_details}

KEY POINTS IDENTIFIED DURING SEARCH:
{key_points}

INSTRUCTIONS:
Create a concise list of 5-7 bullet points that represent the most important findings and facts related to the query.
Each point should be clear, specific, and directly relevant to answering the original query.

IMPORTANT RULES:
1. ONLY include information that is directly supported by the search context
2. DO NOT make up or infer information not present in the search results
3. If information is missing or unclear, note it as a limitation rather than making assumptions
4. Clearly cite sources for each point using markdown hyperlinks: ([Source Title](source_url))
5. Use direct quotes from search results when appropriate
6. Maintain academic rigor and avoid speculation
7. If there are different results, carefully consider all search results and provide a final answer that reflects the most accurate information.
8. If the search results are contradictory, acknowledge the uncertainty and provide a balanced view.
9. If the search results are not relevant to the query, state that you cannot provide an answer based on the search results.
10. If the search results are too vague or unclear, state that you cannot provide a definitive answer.


Format your response as a markdown list of bullet points ONLY:
- Key point 1 ([Source Title](source_url))
- Key point 2 ([Source Title](source_url))
...

Do not include any introduction, explanation, or conclusion outside of the bullet points.
"""

# Define the prompt template for direct answer generation
DIRECT_ANSWER_TEMPLATE = """You are an expert content writer. Your task is to formulate a direct answer to the original query.

ORIGINAL QUERY: {original_query}

SEARCH CONTEXT:
{search_context}

KEY POINTS:
{key_points}

SEARCH DETAILS:
{search_details}

INSTRUCTIONS:
Create a well-rounded, complete direct answer to the original query. The answer should:
1. Be comprehensive but concise (use as many paragraphs as needed to cover the topic thoroughly)
2. Address the core question directly without tangents
3. Synthesize the key points into a coherent response
4. Use an authoritative, clear writing style
5. Use line breaks between paragraphs for better readability
6. Use **bold** for important terms and concepts
7. Use *italics* for emphasis when appropriate

IMPORTANT RULES:
1. ONLY include information that is directly supported by the search context
2. DO NOT make up or infer information not present in the search results
3. If information is missing or unclear, note it as a limitation rather than making assumptions
4. Clearly cite sources for each piece of information using markdown hyperlinks: ([Source Title](source_url))
5. Use direct quotes from search results when appropriate
6. Maintain academic rigor and avoid speculation
7. If the search context is insufficient to answer a point, clearly state this limitation
8. If there are different results, carefully consider all search results and provide a final answer that reflects the most accurate information.

Your direct answer should be self-contained and provide a complete response to the original query.
Do not include any headings, bullet points, or section markers.
"""

# Define the template for detailed notes generation
DETAILED_NOTES_TEMPLATE = """You are an expert content writer. Your task is to provide an outline of detailed sections for expanding on the direct answer.

ORIGINAL QUERY: {original_query}

SEARCH CONTEXT:
{search_context}

KEY POINTS:
{key_points}

DIRECT ANSWER:
{direct_answer}

SEARCH DETAILS:
{search_details}

INSTRUCTIONS:
Create an outline for detailed, structured notes that expand on the direct answer with more in-depth information. Your outline should:
1. Include logical sections with clear headings (up to 5 sections for thorough coverage)
2. Focus on clear, descriptive section titles that reflect the key aspects of the topic
3. Keep the outline simple - just the section headings in markdown format

IMPORTANT RULES:
1. ONLY include sections that can be fully supported by the search context
2. DO NOT create sections that would require information not present in the search results
3. Clearly indicate which search results support each section using markdown hyperlinks
4. If certain aspects cannot be covered due to limited search context, note this limitation

Format your response as a numbered list of section headings in markdown format, like this:
1. ## Section Heading 1 ([Source Title](source_url))
2. ## Section Heading 2 ([Source Title](source_url))
3. ## Section Heading 3 ([Source Title](source_url))

DO NOT include any content under these headings - just provide the section headings.
Each section will be expanded in a separate step. Do not include an introduction or conclusion.
"""

# Define the template for generating content for a single section
SECTION_CONTENT_TEMPLATE = """You are an expert content writer. Your task is to write detailed content for a specific section of a comprehensive report.

ORIGINAL QUERY: {original_query}

SEARCH CONTEXT:
{search_context}

KEY POINTS:
{key_points}

DIRECT ANSWER:
{direct_answer}

SEARCH DETAILS:
{search_details}

SECTION TO EXPAND: {section_heading}

INSTRUCTIONS:
Create rich, detailed content for the section "{section_heading}". Your content should:
1. Be thorough and comprehensive (at least 2-4 paragraphs plus additional elements as needed)
2. Include technical details, examples, and comparisons where relevant
3. Elaborate on all important aspects related to this specific section
4. Use proper markdown formatting for subsections and formatting
5. Where relevant, include:
   - Tables for comparing options or features
   - Bulleted lists for steps or features
   - Numbered lists for sequential processes
   - Mathematical formulas if applicable
6. Use **bold** for important terms and concepts
7. Use *italics* for emphasis when appropriate
8. Create subsections with ### heading level when needed to organize complex information

IMPORTANT RULES:
1. ONLY include information that is directly supported by the search context
2. DO NOT make up or infer information not present in the search results
3. If information is missing or unclear, note it as a limitation rather than making assumptions
4. Clearly cite sources for each piece of information using markdown hyperlinks: ([Source Title](source_url))
5. Use direct quotes from search results when appropriate
6. Maintain academic rigor and avoid speculation
7. If the search context is insufficient to cover a point, clearly state this limitation
8. Focus ONLY on this section without repeating information from other sections
9. If there are different results, carefully consider all search results and provide a final answer that reflects the most accurate information.

IMPORTANT: DO NOT include the main section heading ("{section_heading}") in your response - I will add it separately.
Start directly with the content. If you need subsections, use ### level headings, not ## level headings.

Provide in-depth, authoritative content with specific facts, figures, and examples where possible,
while strictly adhering to the information available in the search context.
"""

# Define the template for initial query generation
QUERY_GENERATOR_TEMPLATE = """You are an expert research strategist. Your task is to break down a complex research query into multiple focused search queries.

ORIGINAL QUERY: {original_query}

CURRENT DATE: {current_date} (ISO 8601 format: YYYY-MM-DD)

INSTRUCTIONS:
Analyze the original query and break it down into 5 distinct, focused search queries that collectively cover all important aspects of the original question. Each query should:

1. Target a specific aspect of the original question
2. Be clear, concise, and searchable (under 100 characters if possible)
3. Use natural language that would work well with search engines
4. Avoid overlapping too much with other queries
5. Focus on factual information rather than opinions
6. IMPORTANT: Include the current year or time period in queries when relevant (especially for financial, news, trend, or stock queries) to ensure recency
7. When referring to time periods, use clear universal formats (e.g., "Q1 2024", "May 2024", "2024", etc.)

Format your response as a JSON array of 5 strings representing the search queries:
["query1", "query2", "query3", "query4", "query5"]

CRITICAL: Your entire response MUST be a valid, parseable JSON array and nothing else. Do not include any text before or after the JSON array. Do not include any explanation, markdown formatting, or code blocks around the JSON. The response must start with '[' and end with ']' and contain only valid JSON.
"""

def init_reasoning_llm(temperature: float = 0.3):
    """Initialize the language model for reasoning using OpenAI-compatible API."""
    # Use OpenAI-compatible server
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "no-need"),
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base if not openai_api_key or openai_api_key == "not-needed" else None,
        temperature=temperature,
        max_tokens=1024,
        streaming=True
    )
    return llm

def generate_initial_queries(original_query: str) -> List[str]:
    """
    Generate multiple focused search queries from the original complex query.

    Args:
        original_query: The original user query, which might be complex or very long

    Returns:
        A list of 5 focused search queries
    """
    # Initialize the LLM with low temperature for consistent output
    llm = init_reasoning_llm(temperature=0.2)

    # Get current date information in ISO 8601 format (YYYY-MM-DD)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Create the query generator prompt
    query_generator_prompt = PromptTemplate(
        input_variables=["original_query", "current_date"],
        template=QUERY_GENERATOR_TEMPLATE
    )

    # Create the chain
    chain = query_generator_prompt | llm

    # Generate the search queries
    response = chain.invoke({
        "original_query": original_query,
        "current_date": current_date
    })

    # Extract the content if it's a message object
    query_text = response.content if hasattr(response, 'content') else response

    # Clean up the response text
    query_text = query_text.strip()
    # Remove any markdown code block markers
    query_text = re.sub(r'^```json\s*', '', query_text)
    query_text = re.sub(r'\s*```$', '', query_text)

    try:
        # Parse the JSON response
        queries = json.loads(query_text)

        # Ensure we have a list of strings
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            logger.info(f"Generated {len(queries)} initial search queries")
            for i, query in enumerate(queries):
                logger.info(f"  Initial query {i+1}: {query}")
            return queries
        else:
            logger.warning("Query generator did not return a proper list of strings")
            # Fall back to using the original query
            return [original_query]

    except json.JSONDecodeError:
        logger.error(f"Failed to parse query generator JSON output: {query_text[:100]}...")
        # Fall back to using the original query
        return [original_query]

def format_search_results(state: SearchState) -> str:
    """Format the search results for the prompt."""
    results_text = ""

    # Use combined results if available
    results = state.combined_results if state.combined_results else []

    # If we don't have combined results, try individual result types
    if not results:
        if state.faiss_results:
            results.extend(state.faiss_results)
        if state.bm25_results:
            results.extend(state.bm25_results)
        if state.tavily_results:
            results.extend(state.tavily_results)

    # Format each result with the query that produced it (if available)
    for i, result in enumerate(results):
        results_text += f"RESULT {i+1}:\n"
        results_text += f"Title: {result.title}\n"
        results_text += f"URL: {result.url}\n"
        if result.query:
            results_text += f"Query: {result.query}\n"
        results_text += f"Content: {result.content}\n"
        if result.score is not None:
            results_text += f"Relevance Score: {result.score:.4f}\n"
        results_text += "\n"

    return results_text

def format_search_details(state: SearchState) -> str:
    """Format the search details for the answer generation."""
    details = f"Total search iterations: {state.current_iteration}\n\n"
    details += f"Queries used:\n"

    # Add the original query
    details += f"- Original query: {state.original_query}\n"

    # Add all the generated queries
    for i, query in enumerate(state.generated_queries):
        if query != state.original_query:
            details += f"- {query}\n"

    # Add knowledge gaps that were identified
    if state.knowledge_gaps:
        details += f"\nKnowledge gaps identified during search:\n"
        for gap in state.knowledge_gaps:
            details += f"- {gap}\n"

    return details

def deep_reasoning_agent(state: SearchState, max_iterations: int = 5) -> Generator[bytes, None, SearchState]:
    """
    Uses deep reasoning to analyze results, identify knowledge gaps, and decide if further search is needed.

    Args:
        state: The current search state with combined search results
        max_iterations: Maximum number of search iterations allowed

    Returns:
        Updated state with analysis and potentially new search queries
    """
    # If this is the first iteration, generate initial search queries
    if state.current_iteration == 0:
        # Generate initial focused queries from the original query
        yield to_chunk_data(
            wrap_thought(
                "Deep reasoning agent: Generating initial search queries",
                f"Generating initial search queries from the original query: {state.original_query}"
            )
        )
        initial_queries = generate_initial_queries(state.original_query)

        # Store the generated queries
        state.generated_queries = initial_queries

        # Don't analyze results yet since we need to perform searches first
        state.current_iteration += 1
        return state

    # Check if we have any search results to work with
    if (not state.combined_results and
        not state.faiss_results and
        not state.bm25_results and
        not state.tavily_results):
        yield to_chunk_data(
            wrap_thought(
                "Deep reasoning agent: No results found.",
                "No relevant search results were found. Ending search."
            )
        )
        # No results, set search as complete with appropriate message
        state.search_complete = True
        state.key_points = ["No relevant information found for the query."]
        state.final_answer = "I couldn't find relevant information to answer your query."
        state.confidence_score = 0.1
        return state

    yield to_chunk_data(
        wrap_thought(
            "Deep reasoning agent: Analyzing search results",
            f"Analyzing results from iteration {state.current_iteration}"
        )
    )

    # Initialize the LLM with a very low temperature for structured output
    llm = init_reasoning_llm(temperature=0.1)

    # Get current date information in ISO 8601 format (YYYY-MM-DD)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Create the reasoning prompt
    reasoning_prompt = PromptTemplate(
        input_variables=["original_query", "iteration", "search_results", "previous_knowledge_gaps", "max_iterations", "current_date"],
        template=REASONING_TEMPLATE
    )

    # Use the newer approach to avoid deprecation warnings
    chain = reasoning_prompt | llm

    # Format the search results
    formatted_results = format_search_results(state)

    # Format the previous knowledge gaps
    formatted_previous_gaps = "None identified yet." if not state.historical_knowledge_gaps else "\n".join([f"- {gap}" for gap in state.historical_knowledge_gaps])

    # Generate the analysis and reasoning
    response = chain.invoke({
        "original_query": state.original_query,
        "iteration": state.current_iteration,
        "search_results": formatted_results,
        "previous_knowledge_gaps": formatted_previous_gaps,
        "max_iterations": max_iterations,
        "current_date": current_date
    })

    # Extract the content if it's a message object
    if hasattr(response, 'content'):
        analysis_text = response.content
    else:
        analysis_text = response

    # Clean up the response text to improve JSON parsing chances
    analysis_text = analysis_text.strip()
    # Remove any markdown code block markers
    analysis_text = re.sub(r'^```json\s*', '', analysis_text)
    analysis_text = re.sub(r'\s*```$', '', analysis_text)
    # Remove any stray markdown characters
    analysis_text = re.sub(r'^#+\s*', '', analysis_text)

    # Parse the JSON response
    try:
        analysis = json.loads(analysis_text)

        # Update the state with the analysis results
        state.key_points = analysis.get("key_points", [])
        new_knowledge_gaps = analysis.get("knowledge_gaps", [])

        # Filter out any knowledge gaps that have been identified before
        filtered_knowledge_gaps = [gap for gap in new_knowledge_gaps
                                  if gap not in state.historical_knowledge_gaps]

        # Update current knowledge gaps (for this iteration)
        state.knowledge_gaps = filtered_knowledge_gaps

        # Add new knowledge gaps to the historical list
        state.historical_knowledge_gaps.extend(filtered_knowledge_gaps)

        state.search_complete = analysis.get("search_complete", False)

        # If we need to continue searching, add new queries
        if not state.search_complete and "new_queries" in analysis and analysis["new_queries"]:
            yield to_chunk_data(
                wrap_thought(
                    "Deep reasoning agent: Identified knowledge gaps",
                    f"Found {len(filtered_knowledge_gaps)} new knowledge gaps. Generating new search queries."
                )
            )
            state.generated_queries = analysis["new_queries"]
            logger.info(f"Generated {len(state.generated_queries)} new queries based on knowledge gaps")
            for i, query in enumerate(state.generated_queries):
                logger.info(f"  New query {i+1}: {query}")

        # Log the reasoning
        if "reasoning" in analysis:
            logger.info(f"Reasoning: {analysis['reasoning']}")

    except json.JSONDecodeError:
        yield to_chunk_data(
            wrap_thought(
                "Deep reasoning agent: Error parsing analysis",
                "Encountered error parsing analysis output. Attempting recovery."
            )
        )
        logger.error(f"Failed to parse deep reasoning JSON output: {analysis_text[:100]}...")
        # Try to extract JSON-like content using regex

        try:
            # First attempt: Look for complete JSON object in the text
            json_match = re.search(r'(\{.*\})', analysis_text, re.DOTALL)
            if json_match:
                json_string = json_match.group(1)
                # Try to parse it again
                analysis = json.loads(json_string)
                logger.info("Successfully extracted JSON using regex pattern 1")
            else:
                # Second attempt: Try to fix common JSON formatting issues
                fixed_text = analysis_text
                # Replace single quotes with double quotes if they're being used for JSON
                if "'" in fixed_text and '"' not in fixed_text:
                    fixed_text = fixed_text.replace("'", '"')

                # Try to add missing quotes around property names
                fixed_text = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed_text)

                # Fix boolean values that might be capitalized
                fixed_text = re.sub(r':\s*True\b', r':true', fixed_text)
                fixed_text = re.sub(r':\s*False\b', r':false', fixed_text)

                try:
                    analysis = json.loads(fixed_text)
                    logger.info("Successfully parsed JSON after fixing formatting")
                except json.JSONDecodeError:
                    # If we still can't parse JSON, try to build a minimal valid structure
                    logger.warning("Attempting to build a minimal valid structure from response")

                    # Try to extract key points from the response
                    key_points = []
                    knowledge_gaps = []
                    new_queries = []
                    search_complete = True  # Default to true to avoid infinite loops

                    # Extract any content that looks like lists
                    key_points_match = re.search(r'"key_points"\s*:\s*\[(.*?)\]', fixed_text, re.DOTALL)
                    if key_points_match:
                        # Split by commas, but only those followed by a quote to avoid splitting content with commas
                        points_text = key_points_match.group(1)
                        points = re.findall(r'"([^"]*)"', points_text)
                        if points:
                            key_points = points

                    # Extract knowledge gaps similarly
                    gaps_match = re.search(r'"knowledge_gaps"\s*:\s*\[(.*?)\]', fixed_text, re.DOTALL)
                    if gaps_match:
                        gaps_text = gaps_match.group(1)
                        gaps = re.findall(r'"([^"]*)"', gaps_text)
                        if gaps:
                            knowledge_gaps = gaps
                            search_complete = False

                    # Extract new queries
                    queries_match = re.search(r'"new_queries"\s*:\s*\[(.*?)\]', fixed_text, re.DOTALL)
                    if queries_match:
                        queries_text = queries_match.group(1)
                        queries = re.findall(r'"([^"]*)"', queries_text)
                        if queries:
                            new_queries = queries

                    # Check for search_complete value
                    complete_match = re.search(r'"search_complete"\s*:\s*(true|false)', fixed_text, re.IGNORECASE)
                    if complete_match:
                        search_complete = complete_match.group(1).lower() == 'true'

                    # Build our analysis dictionary
                    analysis = {
                        "key_points": key_points,
                        "knowledge_gaps": knowledge_gaps,
                        "new_queries": new_queries,
                        "search_complete": search_complete
                    }

                    logger.info("Successfully built analysis from extracted components")

                    # If we couldn't extract anything useful, try to get key points from bullet points
                    if not key_points:
                        lines = analysis_text.split('\n')
                        for line in lines:
                            line = line.strip()
                            # Look for bullet points or numbered lists that might be key points
                            if re.match(r'^[\*\-\d\.]\s+', line) and len(line) > 5:
                                # Remove the bullet or number
                                point = re.sub(r'^[\*\-\d\.]+\s+', '', line)
                                key_points.append(point)

                        if key_points:
                            analysis["key_points"] = key_points
                            logger.info(f"Extracted {len(key_points)} key points from bullet points")

            # Update the state with the analysis results
            state.key_points = analysis.get("key_points", [])
            new_knowledge_gaps = analysis.get("knowledge_gaps", [])
            filtered_knowledge_gaps = [gap for gap in new_knowledge_gaps if gap not in state.historical_knowledge_gaps]
            state.knowledge_gaps = filtered_knowledge_gaps
            state.historical_knowledge_gaps.extend(filtered_knowledge_gaps)
            state.search_complete = analysis.get("search_complete", False)

            # If we need to continue searching, add new queries
            if not state.search_complete and "new_queries" in analysis and analysis["new_queries"]:
                yield to_chunk_data(
                    wrap_thought(
                        "Recovered analysis",
                        f"Successfully recovered analysis. Generated {len(analysis['new_queries'])} new queries."
                    )
                )
                state.generated_queries = analysis["new_queries"]
                logger.info(f"Generated {len(state.generated_queries)} new queries based on knowledge gaps")
                for i, query in enumerate(state.generated_queries):
                    logger.info(f"  New query {i+1}: {query}")

        except Exception as e:
            yield to_chunk_data(
                wrap_thought(
                    "Deep reasoning agent: Analysis failed",
                    f"Failed to analyze search results: {str(e)}"
                )
            )
            logger.error(f"Failed to extract JSON data: {str(e)}")
            state.search_complete = True
            state.key_points = ["Error in analyzing search results."]

    # If we've reached the maximum iterations, force completion
    if state.current_iteration >= max_iterations:
        yield to_chunk_data(
            wrap_thought(
                "Deep reasoning agent: Maximum iterations reached",
                f"Reached maximum iterations ({max_iterations}). Concluding search."
            )
        )
        state.search_complete = True
        logger.info(f"Reached maximum iterations ({max_iterations}), forcing search completion")

    # Increment the iteration counter
    state.current_iteration += 1

    # If search is complete, generate the final answer
    if state.search_complete:
        yield to_chunk_data(
            wrap_thought(
                "Deep reasoning agent: Generating final answer",
                "Search complete. Generating comprehensive final answer."
            )
        )
        state = yield from generate_final_answer(state)

    return state

def generate_final_answer(state: SearchState) -> Generator[bytes, None, SearchState]:
    """
    Generates the final, structured answer in a multi-stage process:
    1. Generate concise key points
    2. Create a direct answer based on key points
    3. Generate an outline for detailed notes sections
    4. Expand each section with dedicated LLM calls

    Args:
        state: The current search state with key points and other information

    Returns:
        Updated state with the final structured answer
    """
    # Format the search details
    search_details = format_search_details(state)

    # Format search context from combined results
    search_context = ""
    if state.combined_results:
        search_context = "\n\n".join([
            f"Source: {result.title}\nURL: {result.url}\nContent: {result.content}"
            for result in state.combined_results
        ])

    # Format the key points from the deep reasoning
    initial_key_points = "\n".join([f"- {point}" for point in state.key_points])

    # Stage 1: Generate refined key points
    yield to_chunk_data(
        wrap_thought(
            "Deep reasoning agent: Refining key points",
            "Generating refined key points from search results"
        )
    )
    key_points_llm = init_reasoning_llm(temperature=0.2)
    key_points_prompt = PromptTemplate(
        input_variables=["original_query", "search_details", "key_points", "search_context"],
        template=KEY_POINTS_TEMPLATE
    )
    key_points_chain = key_points_prompt | key_points_llm

    key_points_response = key_points_chain.invoke({
        "original_query": state.original_query,
        "search_details": search_details,
        "key_points": initial_key_points,
        "search_context": search_context
    })

    # Extract the content if it's a message object
    key_points = key_points_response.content if hasattr(key_points_response, 'content') else key_points_response
    logger.info("Generated key points for final answer")

    # Stage 2: Generate direct answer
    yield to_chunk_data(
        wrap_thought(
            "Generating final answer: Creating direct answer",
            "Generating concise direct answer from key points"
        )
    )
    direct_answer_llm = init_reasoning_llm(temperature=0.3)
    direct_answer_prompt = PromptTemplate(
        input_variables=["original_query", "key_points", "search_details", "search_context"],
        template=DIRECT_ANSWER_TEMPLATE
    )
    direct_answer_chain = direct_answer_prompt | direct_answer_llm

    direct_answer_response = direct_answer_chain.invoke({
        "original_query": state.original_query,
        "key_points": key_points,
        "search_details": search_details,
        "search_context": search_context
    })

    # Extract the content if it's a message object
    direct_answer = direct_answer_response.content if hasattr(direct_answer_response, 'content') else direct_answer_response
    logger.info("Generated direct answer")

    # Stage 3: Generate detailed notes outline (section headings only)
    yield to_chunk_data(
        wrap_thought(
            "Generating final answer: Planning detailed notes",
            "Creating outline for detailed explanation"
        )
    )
    outline_llm = init_reasoning_llm(temperature=0.3)
    outline_prompt = PromptTemplate(
        input_variables=["original_query", "key_points", "direct_answer", "search_details", "search_context"],
        template=DETAILED_NOTES_TEMPLATE
    )
    outline_chain = outline_prompt | outline_llm

    outline_response = outline_chain.invoke({
        "original_query": state.original_query,
        "key_points": key_points,
        "direct_answer": direct_answer,
        "search_details": search_details,
        "search_context": search_context
    })

    # Extract the content if it's a message object
    section_outline = outline_response.content if hasattr(outline_response, 'content') else outline_response
    logger.info("Generated section outline for detailed notes")

    print("Section outline:")
    print(section_outline)

    # Parse the section headings from the outline
    section_headings = []
    for line in section_outline.strip().split('\n'):
        # Match lines that contain section headings (## Something)
        if '##' in line:
            # Extract just the heading text, removing numbers, source citations and other artifacts
            heading = line.split('##')[1].strip()
            if '(' in heading:  # Remove source citation if present
                heading = heading.split('(')[0].strip()
            if heading:  # Skip empty headings
                section_headings.append(heading)

    logger.info(f"Identified {len(section_headings)} sections to expand")

    # Stage 4: Generate detailed content for each section
    yield to_chunk_data(
        wrap_thought(
            "Deep reasoning agent: Expanding sections",
            f"Generating detailed content for {len(section_headings)} sections"
        )
    )
    section_llm = init_reasoning_llm(temperature=0.4)
    section_prompt = PromptTemplate(
        input_variables=["original_query", "key_points", "direct_answer", "search_details", "section_heading", "search_context"],
        template=SECTION_CONTENT_TEMPLATE
    )
    section_chain = section_prompt | section_llm

    # Generate content for each section
    detailed_notes = "## Detailed Notes\n\n"
    for heading in section_headings:
        logger.info(f"Generating content for section: {heading}")

        section_response = section_chain.invoke({
            "original_query": state.original_query,
            "key_points": key_points,
            "direct_answer": direct_answer,
            "search_details": search_details,
            "section_heading": heading,
            "search_context": search_context
        })

        # Extract the content if it's a message object
        section_content = section_response.content if hasattr(section_response, 'content') else section_response

        # Process the content to remove any headings that match the current heading
        # This prevents duplication of the heading we're about to add
        content_lines = section_content.split('\n')
        cleaned_lines = []
        skip_next_line = False

        for line in content_lines:
            # Skip lines that contain the section heading with ## prefix
            if f"## {heading}" in line or f"##  {heading}" in line or heading in line and line.startswith('##'):
                skip_next_line = True
                continue

            # Skip empty line after a heading to maintain proper spacing
            if skip_next_line and not line.strip():
                skip_next_line = False
                continue

            # Keep all other lines
            cleaned_lines.append(line)
            skip_next_line = False

        cleaned_content = '\n'.join(cleaned_lines)

        # Add the heading and cleaned content to the detailed notes
        detailed_notes += f"## {heading}\n\n{cleaned_content.strip()}\n\n"

    logger.info("Generated all section content for detailed notes")

    # Combine all sections into the final answer with enhanced markdown
    final_answer = f"""## Key Points

{key_points}

## Direct Answer

{direct_answer.strip()}

{detailed_notes.replace('# DETAILED NOTES', '## Detailed Notes')}
"""

    # Update the state with the final answer
    state.final_answer = final_answer.strip()

    # Set a confidence score based on search context availability
    state.confidence_score = 0.8 if state.combined_results and state.key_points else 0.5

    return state