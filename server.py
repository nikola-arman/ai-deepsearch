import json
from app import prompt

async def main():
    user_messages = [
        'I want to build strength. I train boxing and S&C. after each train, I can do sauna and cold plunge, please do search and advise me to maximize my recovery and performance.',
        'yes'
    ]

    user_messages_it = iter(user_messages)
    messages = []
    
    while True:
        message = next(user_messages_it, None)
        
        if message is None:
            break
        
        print("\n" * 2)
        print("User: ", message, end="", flush=True)

        messages.append({"role": "user", "content": message})
        response = prompt(messages)
        
        with open("messages.json", "w") as f:
            json.dump(messages, f, indent=2)

        print("\n" * 2)
        print("Assistant: ", end="", flush=True)
        
        assistant_message = ''

        async for chunk in response:
            chunk = chunk.strip()
            if chunk and chunk.startswith(b"data: "):
                chunk = chunk[6:]
        
                if chunk == b"[DONE]":
                    break
        
                decoded_chunk = chunk.decode("utf-8")
                json_chunk = json.loads(decoded_chunk)

                content = json_chunk["choices"][0]["delta"]["content"]
                reasoning_content = json_chunk["choices"][0]["delta"].get("reasoning_content") or ""

                role = json_chunk["choices"][0]["delta"]["role"]

                print(reasoning_content, end="", flush=True)
                print(content, end="", flush=True)

                if role in [None, "assistant"] and content:
                    assistant_message += content

        messages.append({"role": "assistant", "content": assistant_message})
        print("\n" * 2)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())