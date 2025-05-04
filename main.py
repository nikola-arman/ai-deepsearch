import logging
# logging.basicConfig(level=logging.DEBUG)

from dotenv import load_dotenv

load_dotenv()

from app import prompt 
import asyncio

import base64

async def main():
    with open('/home/tndo/projects/medical-ai-deepsearch/0d7923bfdd60e8bcc7f7d239b6afeedd.jpg', 'rb') as fp:
        image_data = fp.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        image_uri = f'data:image/jpeg;base64,{base64_image}'
    

    async for b in prompt([
        {"role": "user", "content": "Surgical Findings: Anteverted, anteflexed, mobile uterus upon hysteroscopy. Findings noted to have an irregular-shaped lesion at the posterior aspect of the mid-section of the intrauterine cavity. In a 71-year-old, is this a reason for a hysterectomy?"}]):
        print(b)

asyncio.run(main())
