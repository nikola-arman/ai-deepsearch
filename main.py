from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.DEBUG)

load_dotenv()

from app import prompt
import asyncio
res, raw = asyncio.run(prompt([{"role": "user", "content": "research about lung cancer"}]))


with open("res.md", "w") as f:
    f.write(res)

with open("raw.md", "w") as f:
    f.write(str(raw))
