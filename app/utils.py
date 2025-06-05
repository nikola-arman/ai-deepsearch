import os

from pydantic import BaseModel
import datetime
import base64
import re

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

async def preserve_upload_file(file_data_uri: str, file_name: str, preserve_attachments: bool = False) -> str:
    os.makedirs(os.path.join(os.getcwd(), 'uploads'), exist_ok=True)

    file_data_base64 = file_data_uri.split(',')[-1]
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    try:
        file_path = os.path.join(os.getcwd(), 'uploads', f"{timestamp}_{file_name}")

        if not preserve_attachments:
            return file_path

        file_data = base64.b64decode(file_data_base64)

        with open(file_path, 'wb') as f:
            f.write(file_data)

        return file_path
    except Exception as e:
        logger.error(f"Failed to preserve upload file: {e}")
        return None

def get_attachments(content: list[dict[str, str]]) -> list[str]:
    attachments = []

    if isinstance(content, str):
        return []

    for item in content:
        logger.info(f"ITEM: {item.keys()}")

        if item.get('type', 'undefined') == 'file':
            file = item.get('file')
            logger.info(f"FILE: {file.keys()}")
            data = file.get('file_data')
            filename = file.get('filename')

            if data and filename:
                attachments.append((data, filename))

        elif item.get('type', 'undefined') == 'image_url':
            image_url = item.get('image_url')
            logger.info(f"IMAGE URL: {image_url.keys()}")
            name = image_url.get('name')
            url = image_url.get('url')

            if url and name:
                attachments.append((url, name))

    return attachments

def refine_chat_history(messages: list[dict[str, str]], system_prompt: str, preserve_attachments: bool = False) -> list[dict[str, str]]:
    refined_messages = []

    has_system_prompt = False

    for message in messages:
        message: dict[str, str]

        if isinstance(message, dict) and message.get('role', 'undefined') == 'system':
            message['content'] += f'\n{system_prompt}'
            has_system_prompt = True
            refined_messages.append(message)
            continue

        if isinstance(message, dict) \
            and message.get('role', 'undefined') == 'user' \
            and isinstance(message.get('content'), list):

            content = message['content']
            text_input = ''
            attachments = []

            for item in content:
                if item.get('type', 'undefined') == 'text':
                    text_input += item.get('text') or ''

                elif item.get('type', 'undefined') == 'file':
                    file_item = item.get('file', {})
                    if 'file_data' in file_item and 'filename' in file_item:


                        file_path = preserve_upload_file(
                            file_item.get('file_data', ''),
                            file_item.get('filename', ''),
                            preserve_attachments
                        )

                        if file_path:
                            attachments.append(file_path)

                elif item.get('type', 'undefined') == 'image_url':
                    file_item = item.get('image_url', {})

                    if 'url' in file_item:
                        file_path = preserve_upload_file(
                            file_item.get('url', ''),
                            file_item.get('name', f'image_{len(attachments)}.jpg'),
                            preserve_attachments
                        )

                        if file_path:
                            attachments.append(file_path)

            if len(attachments) > 0:
                text_input += f'\nAttachments:\n'
                for attachment in attachments:
                    text_input += f'- {attachment}\n'

            refined_messages.append({
                "role": "user",
                "content": text_input
            })

        else:
            refined_messages.append(message)

    if not has_system_prompt and system_prompt != "":
        refined_messages.insert(0, {
            "role": "system",
            "content": system_prompt
        })

    if isinstance(refined_messages[-1], str):
        refined_messages[-1] = {
            "role": "user",
            "content": refined_messages[-1] + '\n/no_think'
        }

    return refined_messages


def strip_thinking_content(content: str) -> str:
    pat = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
    return pat.sub("", content).lstrip()

def refine_assistant_message(
    assistant_message: dict[str, str]
) -> dict[str, str]:

    if 'content' in assistant_message:
        assistant_message['content'] = strip_thinking_content(assistant_message['content'] or "")

    return assistant_message
