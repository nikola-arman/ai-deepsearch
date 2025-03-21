# DeepSearch System - Solution Design

## Overview

DeepSearch is a multi-agent deep search system that combines real-time web search with different retrieval methods and reasoning capabilities. The system is designed to provide high-quality answers to user queries by leveraging multiple specialized agents, each optimized for a specific search-related task.

## Key Features

- **No preloaded knowledge base**: The system creates indexes on-the-fly from freshly fetched search results.
- **Dynamic indexing**: Both vector embeddings (FAISS) and keyword indices (BM25) are generated at runtime.
- **Multi-agent architecture**: Each component is specialized for a specific search task.
- **Adaptive reasoning**: The system refines queries, combines insights, and handles ambiguous requests.
- **Hybrid search approach**: Combination of semantic (vector-based) and keyword-based search for improved results.

## System Architecture

### High-Level Architecture

The DeepSearch system follows a pipeline architecture with specialized agents that process a query sequentially:

1. User submits a query via CLI or REST API
2. Query goes through a sequence of specialized agents
3. Each agent enhances the search state
4. Final answer is synthesized and returned to the user

```
                  ┌──────────────────┐
                  │   User Interface │
                  │   (CLI or API)   │
                  └────────┬─────────┘
                           │
                           ▼
          ┌─────────────────────────────────┐
          │        Search Pipeline          │
          │                                 │
┌─────────┴───────────┐   ┌────────────────┴─────────────┐
│  Query Refinement   │──►│        Web Search            │
│       Agent         │   │         Agent                │
└─────────────────────┘   └────────────────┬─────────────┘
                                           │
                                           ▼
┌─────────────────────┐   ┌────────────────┴─────────────┐
│  BM25 Keyword-based │◄──┤  FAISS Semantic Indexing     │
│    Search Agent     │   │         Agent                │
└─────────┬───────────┘   └─────────────────────────────┘
          │
          ▼
┌─────────┴───────────┐
│  Reasoning Agent    │
│ (Answer Generation) │
└─────────────────────┘
```

### Core Components

#### 1. Search State

The system maintains a `SearchState` object that passes through each agent in the pipeline. This object contains:

- Original and refined queries
- Results from each search method (Tavily, FAISS, BM25)
- Combined results from all search methods
- Final answer and confidence score

#### 2. Query Refinement Agent

- Improves the original query to make it more effective for search
- Preserves named entities and critical terms
- Uses an LLM to generate a refined version of the query
- Falls back to the original query if refinement isn't successful

#### 3. Tavily Search Agent

- Connects to the Tavily API for real-time web search
- Retrieves relevant web pages based on the refined query
- Processes and normalizes the search results
- Provides the foundation data for the dynamic indexing

#### 4. FAISS Indexing Agent

- Creates vector embeddings for search results on-the-fly
- Performs semantic search to find relevant passages
- Ranks results based on semantic similarity to the query
- Adds semantic search results to the search state

#### 5. BM25 Search Agent

- Implements keyword-based search using the BM25 algorithm
- Builds an inverted index from search results dynamically
- Retrieves passages based on term frequency and document length
- Combines results with the semantic search results

#### 6. Reasoning Agent

- Uses an LLM to analyze and synthesize information from all sources
- Generates a comprehensive, coherent answer
- Evaluates confidence in the answer
- Provides attribution to sources when requested

## Implementation Details

### Technology Stack

- **Python**: Core programming language
- **FastAPI**: Web framework for REST API
- **LangChain**: Framework for LLM application development
- **FAISS**: Library for efficient similarity search and clustering of vectors
- **Rank-BM25**: Implementation of the BM25 algorithm for keyword search
- **Tavily API**: External service for web search
- **OpenAI-compatible API**: For LLM reasoning (supports local LLMs like Llama)

### Integration Points

- **LLM Integration**: The system can use either a local OpenAI-compatible API server (like LM Studio, Llama.cpp, or Ollama) or the OpenAI API directly.
- **Tavily API**: External integration for web search capabilities.
- **Docker Support**: The system can be containerized for easy deployment.

### Deployment Options

1. **CLI Application**: Run `main.py` with a query for a one-off search.
2. **REST API**: Run `app.py` to start a FastAPI server with search endpoints.
3. **Docker Container**: Use the included Dockerfile and docker-compose.yml for containerized deployment.

## Data Flow

1. User submits a query through CLI or API
2. Query refinement agent improves the query
3. Tavily search agent fetches web results
4. FAISS indexing agent creates vector embeddings and finds semantically similar passages
5. BM25 search agent builds a keyword index and finds relevant passages
6. Results are combined and ranked
7. Reasoning agent synthesizes information into a coherent answer
8. Answer is returned to the user with optional sources and confidence score

## Extension Points

The system is designed to be extensible in several ways:

1. **Additional Search Methods**: New search agents can be added to the pipeline
2. **Alternative LLMs**: Different language models can be used for refinement and reasoning
3. **Custom Ranking**: The ranking algorithm for combining results can be customized
4. **UI Integration**: The API can be integrated with various frontend applications

## Configuration and Environment

The system uses environment variables for configuration:

- `TAVILY_API_KEY`: Required for web search functionality
- `OPENAI_API_BASE`: URL for OpenAI-compatible API server (default: http://localhost:8080/v1)
- `OPENAI_API_KEY`: OpenAI API key (alternative to OPENAI_API_BASE)

## Conclusion

The DeepSearch system provides a flexible, powerful search solution that combines the best of traditional search techniques with modern LLM capabilities. By using a multi-agent approach, the system can leverage the strengths of different search methodologies while mitigating their individual weaknesses.