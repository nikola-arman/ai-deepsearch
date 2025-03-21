# Multi-Agent Deep Search System

A highly dynamic and adaptive deep search system that efficiently combines Tavily API for real-time web search, Llama reasoning for analysis and synthesis, FAISS for dense vector search, and BM25 for keyword-based retrieval.

## Features

- No preloaded knowledge base: FAISS and BM25 index freshly fetched search results dynamically
- Dynamic index creation: Generates embeddings and keyword indices on-the-fly
- Multi-agent architecture: Each component is specialized for a specific search task
- Adaptive reasoning: Refines queries, combines insights, and handles ambiguous requests

## Requirements

- Tavily API key for web search capabilities
- OpenAI-compatible API server (like LM Studio, llama.cpp server, Ollama, etc.) OR OpenAI API key

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
```

3. Edit `.env` to add your Tavily API key and either:
   - Set the OpenAI-compatible API server URL (default: http://localhost:8080/v1), OR
   - Add your OpenAI API key

4. Run the application:
```bash
python main.py "your search query here"
```

Options:
- `--verbose` or `-v`: Display source information
- `--show-confidence` or `-c`: Show confidence scores
- `--disable-refinement` or `-d`: Disable query refinement

Or run the API server:
```bash
uvicorn app:app --reload
```

## API Usage

The system exposes a REST API endpoint at `/search` that accepts:
- `query`: The search query string
- `include_sources`: Whether to include source links (default: true)
- `include_confidence`: Whether to include confidence score (default: false)
- `disable_refinement`: Whether to skip query refinement (default: false)

## Architecture

The system uses the following agents:
1. Query Refinement Agent - Improves the original query for better search results
2. Tavily Search Agent - Fetches real-time web search results
3. Dynamic FAISS Indexing Agent - Creates vector embeddings for semantic search
4. Dynamic BM25 Search Agent - Performs keyword-based retrieval
5. Reasoning Agent - Synthesizes information into a coherent answer