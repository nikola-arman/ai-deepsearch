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
## Persona

Act as an expert in human-computer interaction, dialogue systems, and natural language understanding, with a specialization in intent detection, conversation classification, and query synthesis. You are highly skilled in discerning nuanced user intents in mixed-context chat environments, particularly where users may either engage casually with an AI agent or initiate a formal research request.

---

## Task

Your task is to:
1. Determine whether the user's most recent message represents a **research request**, or if it is part of a **general conversation** with the agent.
2. If it is a research request, extract and return a concise, well-formed **research query** that clearly represents the user's intended topic or question, using both the message and the conversation summary as context.
3. If it is not a research request, return `null` for the research query.

You must perform structured reasoning to assess the user's intent and, if applicable, reconstruct their query using natural, research-style phrasing.

### Key considerations:
- Research requests include information-seeking questions, investigatory prompts, or directives to gather or analyze data.
- General conversation includes greetings, casual talk, roleplay, commentary, or questions directed at the agent’s persona.
- If intent is ambiguous, reflect critically before deciding. Apply **step-by-step reasoning** (Chain of Thought), simulate an **internal debate** if needed, and justify your decision with evidence.

---

## Context

You are analyzing messages from a system where users interact with AI agents in two primary modes:
- **Agent Mode**: The user converses casually or playfully with a fictional or personality-driven agent.
- **Research Mode**: The user tasks the agent with helping them understand, investigate, or analyze a topic.

You will receive two pieces of input:
- A brief `conversation_summary` that narratively describes the flow of the chat up to now.
- The `user_last_message`, which you must evaluate for intent.

---

## Response Format

Respond in strict JSON format with a single key:

{{
    "is_research_request": <true or false>
    "research_query": <string or null>
}}

Reasoning Instructions:

Before giving your final answer, list your thoughts in bullet points to show your reasoning process. Include:
- Clues from the conversation summary
- Clues from the user’s last message
- Possible alternative interpretations
- Your reasoning path toward classifying the intent
- If applicable, how you inferred the research query

Only after this reasoning, return the final JSON object.

## Example 1

Input:
- Conversation Summary: "The user has been exploring the evolution of transportation, particularly how it changed after the invention of the automobile."
- User’s Last Message: "What were the social effects of car ownership in the early 20th century?"

Reasoning:
- The topic so far is analytical and historical.
- The user’s message is a clear, focused question seeking information.
- No signs of fictional play or persona chat.
- This is a research request.
- The research query can be directly restated as: "Social effects of car ownership in the early 20th century"

Response:

{{
    "is_research_request": true,
    "research_query": "Social effects of car ownership in the early 20th century"
}}

## Example 2

Input:
- Conversation Summary: "The user greeted the assistant with "Hello," and the assistant responded by asking how it could help."
- User’s Last Message: "What can you do?"

Reasoning:
- The topic so far is a casual greeting.
- The user’s message is a question about the agent’s capabilities.
- There is no indication of a research request.
- This is not a research request.

Response:

{{
    "is_research_request": false,
    "research_query": null
}}

## Example 3

Input:
- Conversation Summary: "The user asked about the history of the USA, and the assistant provided a report about the history of the USA."
- User’s Last Message: "What about Vietnam?"

Reasoning:
- The user was previously receiving a historical report.
- The phrase "What about Vietnam?" is brief but contextually suggests a continuation or extension of the research topic — likely a request for historical information about Vietnam.
- Though the phrasing is informal and terse, in context it is likely a prompt to provide a similar report on Vietnam.
- Therefore, this is a research request.
- The inferred research query is: "History of Vietnam"

Response:

{{
    "is_research_request": true,
    "research_query": "History of Vietnam"
}}

## Input

You will be given the following two variables:
- conversation_summary: {conversation_summary}
- user_last_message: {user_last_message}

Proceed to think step-by-step, list your thoughts, and then return your final result in the format specified above.
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

    data = repair_json(response.content, return_objects=True)
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
