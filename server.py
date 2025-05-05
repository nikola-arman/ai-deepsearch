from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import uuid
import asyncio
import json
from typing import Dict, Optional, List
from app import prompt
from pathlib import Path
import openai

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
public_dir = os.path.join(current_dir, "public")
cache_dir = os.path.join(current_dir, "cache")

# Create cache directory if it doesn't exist
Path(cache_dir).mkdir(parents=True, exist_ok=True)


def get_task_file_path(task_id: str) -> str:
    """Get the file path for a task's cache file."""
    return os.path.join(cache_dir, f"{task_id}.json")

def save_task_to_disk(task_id: str, task_data: Dict):
    """Save task data to disk."""
    file_path = get_task_file_path(task_id)
    with open(file_path, 'w') as f:
        json.dump(task_data, f)

def load_task_from_disk(task_id: str) -> Optional[Dict]:
    """Load task data from disk."""
    file_path = get_task_file_path(task_id)
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        return json.load(f)

def delete_task_from_disk(task_id: str):
    """Delete task data from disk."""
    file_path = get_task_file_path(task_id)
    if os.path.exists(file_path):
        os.remove(file_path)
        

from copy import deepcopy
async def chat_completion_loop(messages: list[dict[str, str]], **additional_kwargs):
    original_message = deepcopy(messages)

    toolcalls = [
        {
            "type": "function",
            "function": {
                "name": "do_specialized",
                "description": f"Start deep diving into the problem",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    ]

    client = openai.AsyncClient(
        api_key=os.getenv("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", 'local-model')),
        base_url=os.getenv("LLM_BASE_URL", 'https://api.openai.com/v1'),
    )

    name = "Bio-Medical Deep Search"
    description = "a user-friendly assistant, deepdive into biomedical research, walkarround PubMed and answer any question related to biomedical ressearch and advancements"

    messages[-1]['content'] = f'''
{messages[-1]['content']}
{"-" * 30}
Use the basic information below to quickly complete simple tasks like introducing, greeting, or answering follow-up questions, etc. Also, follow the conversation and keep it natural and concise:
Information: You are {name}, {description}.'''

    completion = await client.chat.completions.create(
        model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
        messages=messages,
        tools=toolcalls,
        tool_choice="auto",
        max_tokens=1024
    )

    # check if the model requests toolcall or not 
    if (
        completion.choices[0].message.tool_calls is not None \
        and len(completion.choices[0].message.tool_calls) > 0
    ):
        res = await prompt(
            original_message,
            **additional_kwargs
        )
        return res

    return completion.choices[0].message.content

async def run_research_task(task_id: str, messages: List[Dict[str, str]]):
    """Run research in background and store results."""
    try:
        # Update task status to running
        task_data = {
            "status": "running",
            "messages": messages,
            "results": None,
            "error": None
        }
        save_task_to_disk(task_id, task_data)
        
        # Run the research pipeline
        results = await chat_completion_loop(messages)
        
        # Update task with results
        task_data["status"] = "completed"
        task_data["results"] = results
        save_task_to_disk(task_id, task_data)
        print(f"task {task_id} completed")
        
    except Exception as e:
        # Handle any errors
        task_data["status"] = "failed"
        task_data["error"] = str(e)
        save_task_to_disk(task_id, task_data)

@app.post('/v1/chat/completions')
async def chat_completions(request: Request):
    """Start a research task and return a task ID."""
    try:
        data = await request.json()
        messages = data.get("messages", [])
        
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
            
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        
        # Initialize task status
        task_data = {
            "status": "running",
            "messages": messages,
            "results": None,
            "error": None
        }
        save_task_to_disk(task_id, task_data)
        
        # Start research in background
        asyncio.create_task(run_research_task(task_id, messages))
        
        # Return task ID
        return JSONResponse({
            "task_id": task_id,
            "status": "running",
            "message": "Research started. Use the polling endpoint to check results."
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/v1/research/{task_id}')
async def get_research_results(task_id: str):
    """Poll endpoint to check research status and get results."""
    task_data = load_task_from_disk(task_id)
    if task_data is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task_data["status"] == "failed":
        raise HTTPException(status_code=500, detail=task_data["error"])
        
    if task_data["status"] == "completed":
        results = task_data["results"]
        return JSONResponse({
            "status": "completed",
            "results": results
        })
        
    return JSONResponse({
        "status": "running",
        "message": "Research in progress"
    })

@app.delete('/v1/research/{task_id}')
async def delete_research_results(task_id: str):
    """Delete research results from cache."""
    if not os.path.exists(get_task_file_path(task_id)):
        raise HTTPException(status_code=404, detail="Task not found")
    
    delete_task_from_disk(task_id)
    return JSONResponse({
        "status": "success",
        "message": "Research results deleted"
    })

# For SPA-like behavior, serve index.html for 404 errors
@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return FileResponse(os.path.join(public_dir, "index.html"))

# Mount static files from the 'public' directory
app.mount("/", StaticFiles(directory=public_dir, html=True), name="static")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 2013))
    uvicorn.run(app, host="0.0.0.0", port=port)
