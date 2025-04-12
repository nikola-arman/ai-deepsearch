from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.DEBUG)

load_dotenv()

from app import prompt
import asyncio
print(asyncio.run(prompt([{"role": "user", "content": "research about lung cancer"}])))