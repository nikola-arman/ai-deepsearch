#!/usr/bin/env python3
"""
FastAPI application for the DeepSearch system.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from main import run_deep_search_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("deepsearch-api")

# Load environment variables
load_dotenv()

# Verify environment variables
required_env_vars = ["TAVILY_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Check if OPENAI_API_BASE or OPENAI_API_KEY is set
if not os.environ.get("OPENAI_API_BASE") and not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Either OPENAI_API_BASE or OPENAI_API_KEY must be set")

# Create FastAPI app
app = FastAPI(
    title="DeepSearch API",
    description="A multi-agent deep search system using Tavily, Llama.cpp, FAISS, and BM25",
    version="0.2.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class Source(BaseModel):
    """Source model for search results."""
    title: str
    url: str


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., description="The query to search for")
    include_sources: bool = Field(True, description="Include sources in the response")
    include_confidence: bool = Field(False, description="Include confidence score in the response")
    max_iterations: int = Field(3, description="Maximum number of search iterations")


class KeyPoint(BaseModel):
    """Key point model for structured answers."""
    text: str


class SearchResponse(BaseModel):
    """Search response model."""
    original_query: str
    answer: str
    key_points: Optional[List[KeyPoint]] = None
    detailed_notes: Optional[str] = None
    generated_queries: Optional[List[str]] = None
    iterations: Optional[int] = None
    confidence: Optional[float] = None
    sources: Optional[List[Source]] = None


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for the API."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> Dict[str, Any]:
    """
    Perform a deep search with the given query.

    Args:
        request: The search request

    Returns:
        The search response
    """
    try:
        logger.info(f"Processing search request for query: {request.query}")

        # Run the deep search pipeline
        result = run_deep_search_pipeline(
            request.query,
            request.max_iterations
        )

        # Build the response
        response = {
            "original_query": result["original_query"],
            "answer": result["answer"],
            "generated_queries": result["generated_queries"],
            "iterations": result["iterations"]
        }

        # Include confidence if requested
        if request.include_confidence:
            response["confidence"] = result["confidence"]

        # Include sources if requested
        if request.include_sources:
            response["sources"] = result["sources"]

        # Include structured answer components if available
        if "key_points" in result and result["key_points"]:
            response["key_points"] = [{"text": point} for point in result["key_points"]]

        if "detailed_notes" in result and result["detailed_notes"]:
            response["detailed_notes"] = result["detailed_notes"]

        logger.info(f"Successfully processed query: {request.query}")
        return response

    except Exception as e:
        # Log the error
        logger.error(f"Error processing search request: {str(e)}", exc_info=True)
        # Raise an HTTP exception
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/healthcheck")
async def healthcheck() -> Dict[str, str]:
    """
    Health check endpoint.

    Returns:
        A status message
    """
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)