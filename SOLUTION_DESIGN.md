# DeepSearch System - Solution Design

## Overview

DeepSearch is a multi-agent deep search system that combines real-time web search with different retrieval methods and reasoning capabilities. The system is designed to provide high-quality answers to user queries by leveraging multiple specialized agents, each optimized for a specific search-related task. The latest version implements a powerful multi-query approach with iterative reasoning for comprehensive and in-depth search results.

## Key Features

- **No preloaded knowledge base**: The system creates indexes on-the-fly from freshly fetched search results.
- **Dynamic indexing**: Both vector embeddings (FAISS) and keyword indices (BM25) are generated at runtime.
- **Multi-agent architecture**: Each component is specialized for a specific search task.
- **Multi-query approach**: Generates multiple diverse queries to explore different aspects of the question.
- **Iterative reasoning**: Analyzes search results, identifies knowledge gaps, and generates new queries when needed.
- **Structured answers**: Provides formatted responses with key points, direct answer, and detailed notes.
- **Hybrid search approach**: Combination of semantic (vector-based) and keyword-based search for improved results.

## System Architecture

### High-Level Architecture

The DeepSearch system follows a pipeline architecture with specialized agents that process a query sequentially:

1. User submits a query via CLI or REST API
2. Query is refined and expanded into multiple search queries
3. Each query is processed by specialized search agents
4. Results are analyzed by a reasoning agent that identifies knowledge gaps
5. The system iteratively searches for more information if needed
6. Final structured answer is synthesized and returned to the user

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
│  Query Refinement   │──►│       Query Expansion        │
│       Agent         │   │          Agent               │
└─────────────────────┘   └────────────────┬─────────────┘
                                           │
                                           ▼
                           ┌─────────────────────────────┐
                           │     Iterative Search Loop   │◄────┐
                           └────────────────┬────────────┘     │
                                            │                   │
                                            ▼                   │
┌─────────────────────┐   ┌────────────────┴─────────────┐     │
│  BM25 Keyword-based │◄──┤  FAISS Semantic Indexing     │     │
│    Search Agent     │   │         Agent                │     │
└─────────┬───────────┘   └─────────────────────────────┘     │
          │                                                    │
          ▼                                                    │
┌─────────┴───────────┐   ┌─────────────────────────────┐     │
│  Deep Reasoning     │──►│      Knowledge Gap           │─────┘
│       Agent         │   │      Detection               │
└─────────┬───────────┘   └─────────────────────────────┘
          │
          ▼
┌─────────┴───────────┐
│  Answer Generation  │
│  (Structured Format)│
└─────────────────────┘
```

### Core Components

#### 1. Search State

The system maintains a `SearchState` object that passes through each agent in the pipeline. The enhanced state now contains:

- Original and refined queries
- Multiple generated queries for different aspects of the question
- Knowledge gaps identified during analysis
- Iteration tracking for recursive search
- Results from each search method (Tavily, FAISS, BM25) with query attribution
- Combined results from all search methods
- Structured answer components (key points, direct answer, detailed notes)
- Final answer and confidence score

#### 2. Query Refinement Agent

- Improves the original query to make it more effective for search
- Preserves named entities and critical terms
- Uses an LLM to generate a refined version of the query
- Falls back to the original query if refinement isn't successful

#### 3. Query Expansion Agent

- Generates multiple diverse queries that approach the question from different angles
- Each query focuses on specific aspects or dimensions of the original question
- Ensures all queries are directly relevant to answering the original question
- Maintains named entities exactly as written in the original query

#### 4. Tavily Search Agent

- Connects to the Tavily API for real-time web search
- Retrieves relevant web pages based on each generated query
- Processes and normalizes the search results
- Provides the foundation data for the dynamic indexing

#### 5. FAISS Indexing Agent

- Creates vector embeddings for search results on-the-fly
- Performs semantic search to find relevant passages
- Ranks results based on semantic similarity to the query
- Tags results with the query that produced them

#### 6. BM25 Search Agent

- Implements keyword-based search using the BM25 algorithm
- Builds an inverted index from search results dynamically
- Retrieves passages based on term frequency and document length
- Combines results with the semantic search results

#### 7. Deep Reasoning Agent

- Analyzes search results to extract key information
- Identifies knowledge gaps that require further searches
- Decides if the search process should continue or if sufficient information has been gathered
- Generates new search queries to fill knowledge gaps when needed
- Implements a two-stage content generation process:
  - **Outline Creation Stage**: Generates a structured outline with key points, direct answer description, and hierarchical notes outline
  - **Content Writing Stage**: Expands the outline into comprehensive content with detailed paragraphs while maintaining the structure
- Formats the final answer in a structured format with key points, direct answer, and detailed notes

## Implementation Details

### Technology Stack

- **Python**: Core programming language
- **FastAPI**: Web framework for REST API
- **LangChain**: Framework for LLM application development
- **FAISS**: Library for efficient similarity search and clustering of vectors
- **Rank-BM25**: Implementation of the BM25 algorithm for keyword search
- **Tavily API**: External service for web search
- **OpenAI-compatible API**: For LLM reasoning (supports local LLMs like Llama)

### Iterative Search Process

The system implements an iterative search process:

1. **Initial Queries**: Generate multiple diverse queries based on the original question
2. **Search Execution**: Execute searches for each query across all search agents
3. **Result Analysis**: Analyze combined results to extract key information
4. **Knowledge Gap Identification**: Identify gaps in the current knowledge
5. **Query Generation**: Generate new, targeted queries to fill knowledge gaps
6. **Iterative Loop**: Continue the process until sufficient information is gathered or maximum iterations reached
7. **Structured Answer**: Format final answer with key points, direct answer, and detailed notes

### Two-Stage Content Generation

The system uses a sophisticated two-stage approach for generating the final content:

1. **Outline Creation Stage**:
   - Generates a structured outline with three main sections
   - Creates 5-7 bullet points for key findings
   - Develops a description of what the direct answer should cover
   - Establishes a hierarchical outline for detailed notes with main sections and sub-points
   - Suggests specific technical details, examples, and comparisons to include

2. **Content Writing Stage**:
   - Uses a dedicated writer agent to expand the outline into comprehensive content
   - Transforms each outline item into detailed paragraphs
   - Maintains the hierarchical structure established in the outline
   - Adds technical details and examples as specified
   - Ensures smooth transitions between sections
   - Uses an authoritative, clear writing style

This approach separates logical structure creation from content generation, resulting in better-organized answers with more detailed content and improved readability.

### Answer Structure

The system produces structured answers with three key sections:

1. **Key Points**: Concise bullet points summarizing the most important findings (5-7 points)
2. **Direct Answer**: A clear, direct response to the original query in a concise paragraph
3. **Detailed Notes**: Comprehensive explanation with supporting evidence and organization

### Integration Points

- **LLM Integration**: The system can use either a local OpenAI-compatible API server (like LM Studio, Llama.cpp, or Ollama) or the OpenAI API directly.
- **Tavily API**: External integration for web search capabilities.
- **Docker Support**: The system can be containerized for easy deployment.

### Deployment Options

1. **CLI Application**: Run `main.py` with a query for a comprehensive search.
2. **REST API**: Run `app.py` to start a FastAPI server with search endpoints.
3. **Docker Container**: Use the included Dockerfile and docker-compose.yml for containerized deployment.

## Data Flow

1. User submits a query through CLI or API
2. Query refinement agent improves the query
3. Query expansion agent generates multiple diverse queries
4. For each query:
   - Tavily search agent fetches web results
   - FAISS indexing agent creates vector embeddings and finds semantically similar passages
   - BM25 search agent builds a keyword index and finds relevant passages
   - Results are tagged with the query that produced them
5. All results are combined, deduplicated, and ranked
6. Deep reasoning agent analyzes results, extracts key information, and identifies knowledge gaps
7. If knowledge gaps exist and iteration limit not reached, new queries are generated and the process repeats
8. When sufficient information is gathered, a structured answer is generated
9. Final answer is returned to the user with key points, direct answer, and detailed notes

## Extension Points

The system is designed to be extensible in several ways:

1. **Additional Search Methods**: New search agents can be added to the pipeline
2. **Alternative LLMs**: Different language models can be used for refinement, expansion, and reasoning
3. **Custom Ranking**: The ranking algorithm for combining results can be customized
4. **UI Integration**: The API can be integrated with various frontend applications
5. **Knowledge Gap Detection**: The reasoning process for identifying gaps can be enhanced

## Configuration and Environment

The system uses environment variables for configuration:

- `TAVILY_API_KEY`: Required for web search functionality
- `OPENAI_API_BASE`: URL for OpenAI-compatible API server (default: http://localhost:8080/v1)
- `OPENAI_API_KEY`: OpenAI API key (alternative to OPENAI_API_BASE)
- `LLM_MODEL_ID`: Model identifier for the LLM (when using OpenAI API)

## Command Line Options

The CLI supports the following options:

- `--verbose, -v`: Enable verbose output including search iterations and generated queries
- `--show-confidence, -c`: Show confidence score for the answer
- `--disable-refinement, -d`: Disable query refinement
- `--max-iterations, -i`: Set maximum number of search iterations (default: 3)

## Conclusion

The DeepSearch system provides a flexible, powerful search solution that combines multi-query approach with iterative reasoning. By generating multiple diverse queries and analyzing results to identify knowledge gaps, the system can provide more comprehensive and accurate answers to complex questions. The structured answer format with key points, direct answer, and detailed notes ensures that users receive information in a clear and organized manner.