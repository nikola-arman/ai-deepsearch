from dotenv import load_dotenv; load_dotenv()

from app import prompt
import asyncio
import json
from enum import Enum
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

class bcolors(str, Enum):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def print_colored(text, color: bcolors):
    """
    Print text in a specific color.
    
    Args:
        text (str): The text to print.
        color (bcolors): The color to use for printing.
    """

    print(f"{color}{text}{bcolors.ENDC}")

from io import StringIO
import sys

class STDOUTCapture:
    def __init__(self, buffer: StringIO):
        self._original_stdout = None
        self._captured_output = buffer
        
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = self._captured_output
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout

        if exc_type is not None:
            return False

        return True

async def main():
    messages = []

    print("Welcome to the chat! Ctrl+C to exit.")
    
    while True:
        user_input = input("You: ")
        messages.append({"role": "user", "content": user_input})
        assistant_message = ''

        for chunk in prompt(messages):

            if not chunk:
                continue
            
            chunk = chunk.strip()[6:]

            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")

            if chunk == '[DONE]':
                break

            try:
                json_chunk = json.loads(chunk)
                choice = json_chunk['choices'][0]

                role = choice['delta'].get('role')
                content = choice['delta'].get('content')
                reasoning_content = choice['delta'].get('reasoning_content')

                if reasoning_content or role != 'assistant':
                    pass

                else:
                    assistant_message += content
            except json.JSONDecodeError:
                assistant_message += chunk
                logging.error(f"JSON decode error: {chunk}")
                continue

        print("\nAssistant: ", assistant_message)
        messages.append({"role": "assistant", "content": assistant_message})

if __name__ == "__main__":
    asyncio.run(main())
