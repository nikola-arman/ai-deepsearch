import asyncio
import aiohttp
import json
from typing import List

queries = [
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?",
    "What is the capital of Spain?",
    "What is the capital of Portugal?",
    "What is the capital of Greece?",
    "What is the capital of Turkey?",
    "What is the capital of Egypt?",
    "What is the capital of South Africa?",
    "What is the capital of China?",
    "What is the capital of Japan?",
    "What is the capital of Korea?",
    "What is the capital of India?",
    "What is the capital of Pakistan?",
    "What is the capital of Bangladesh?",
    "What is the capital of Nepal?",
]

async def process_query(session: aiohttp.ClientSession, query: str) -> dict:
    async with session.post(
        "http://localhost:7000/prompt",
        json={"messages": [
            {
                "role": "user",
                "content": query
            }
        ]},
        timeout=aiohttp.ClientTimeout(total=None)  # No timeout for streaming
    ) as response:
        print(f"Processing query: {query}")
        full_response = ""
        async for chunk in response.content.iter_chunked(1024):
            if chunk:
                chunk_text = chunk.decode('utf-8')
                full_response += chunk_text
                print(f"Chunk for '{query}': {chunk_text}", end='', flush=True)
        print(f"\nFinished streaming for query: {query}")
        return {"query": query, "response": full_response}

async def process_all_queries(queries: List[str]):
    async with aiohttp.ClientSession() as session:
        tasks = [process_query(session, query) for query in queries]
        results = await asyncio.gather(*tasks)
        return results

async def main():
    results = await process_all_queries(queries)
    print("\nAll queries completed!")
    for result in results:
        print(f"Query: {result['query']}")
        print(f"Full Response: {result['response']}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())


