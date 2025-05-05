import os

from pydantic import BaseModel

os.environ['TAVILY_API_KEY'] = 'no-need'
os.environ['OPENAI_BASE_URL'] = os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL"))
os.environ['OPENAI_API_KEY'] = os.getenv("LLM_API_KEY", 'no-need')

from deepsearch.agents.deep_reasoning import init_reasoning_llm

from json_repair import repair_json
import json
import logging
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class ResearchIntent(BaseModel):
    is_research_request: bool
    research_query: str | None


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
2. Is the query open-ended or exploratory in nature?
3. Does the query require comparing different perspectives or analyzing trends?
4. Are there multiple sub-questions within the main query?

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


def get_conversation_summary_prompt(conversation: list[dict[str, str]]) -> str:
    conversation_str = "\n".join([f"{message['role']}: {message['content']}" for message in conversation])
    
    return f"""
    You are an expert in summarizing conversations.
    You are given a conversation history.
    You need to summarize the conversation in a few sentences.

    Here is the conversation history:
    {conversation_str}
    """


def get_conversation_summary(conversation: list[dict[str, str]]) -> str:
    prompt = get_conversation_summary_prompt(conversation)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    llm = init_reasoning_llm()

    response = llm.invoke(messages)
    
    return response.content


DETECT_RESEARCH_INTENT_PROMPT = """
You are an expert at detecting whether a user message is a research request or just casual conversation.

A research request is when the user asks for information or analysis about a topic.
A casual conversation includes greetings, small talk, or questions about you as an AI.

Given the conversation so far and the user's last message, determine:
1. Is this a research request? (true/false)
2. If yes, what is the main research question being asked? (null if not a research request)

Return your response in this JSON format:
{{
    "is_research_request": <true or false>,
    "research_query": <string or null>
}}

Return only the JSON object without any other text or comments.

Examples:

Input:
- conversation_summary: Discussing transportation history and automobiles
- user_last_message: "What were the social effects of cars in the 1900s?"

Output:
{{
    "is_research_request": true,
    "research_query": "Social effects of car ownership in the early 20th century"
}}

Input:
- conversation_summary: Just started chatting
- user_last_message: "Hi! What can you do?"

Output:
{{
    "is_research_request": false,
    "research_query": null
}}

Now analyze this input:
- conversation_summary: {conversation_summary}
- user_last_message: "{user_last_message}"

Output:
"""


def get_detect_research_intent_prompt(conversation_summary: str, user_last_message: str) -> str:
    return DETECT_RESEARCH_INTENT_PROMPT.format(conversation_summary=conversation_summary, user_last_message=user_last_message)


def detect_research_intent(conversation_summary: str, user_last_message: str) -> ResearchIntent:            
    prompt = get_detect_research_intent_prompt(conversation_summary=conversation_summary, user_last_message=user_last_message)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": prompt,
        },
    ]

    llm = init_reasoning_llm()

    response = llm.invoke(messages)

    print(f"Detect research intent response: {response.content}")

    data = curly_brackets_repair_json(response.content)

    return ResearchIntent.model_validate(data)


def get_conversation_reply_prompt(conversation_summary: str, user_last_message: str) -> str:
    return f"""
You are Vibe Deepsearch, an helpful and friendly AI assistant that can perform thorough research and write detailed report that explores any topic in depth.

Engage in a casual conversation with the user by replying to the following conversation summary and user's last message.
    
Conversation Summary:
{conversation_summary}

User's Last Message:
{user_last_message}"""


def reply_conversation(conversation_summary: str, user_last_message: str) -> str:
    prompt = get_conversation_reply_prompt(conversation_summary=conversation_summary, user_last_message=user_last_message)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    
    llm = init_reasoning_llm()

    response = llm.invoke(messages)

    return response.content


def curly_brackets_repair_json(text: str) -> dict:
    """
    Repair JSON from a string, only take the content between curly brackets.
    """

    # Find the first and last curly brackets
    start = text.find('{')
    end = text.rfind('}') + 1

    # Extract the content between the curly brackets
    content = text[start:end]

    return repair_json(content, return_objects=True)
