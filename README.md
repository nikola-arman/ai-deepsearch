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

## Installing and Running llama.cpp Server

If you prefer to use llama.cpp as your local inference server, follow these steps:

### 1. Clone and Build llama.cpp

```bash
# Clone the repository
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build the project (standard build)
make

# For optimized builds with specific hardware acceleration:
# - For Apple Silicon: make LLAMA_METAL=1
# - For CUDA: make LLAMA_CUBLAS=1
# - For OpenBLAS: make LLAMA_OPENBLAS=1
# - For AMD ROCm: make LLAMA_HIPBLAS=1
```

### 2. Run the Server

You can run the server in two ways:

#### Option A: Automatic Model Download (Recommended)

`llama-server` can automatically download models from Hugging Face:

```bash
# Start server and download model in one command
./llama-server --hf-repo unsloth/gemma-3-4b-it-GGUF --hf-file gemma-3-4b-it-Q8_0.gguf -c 65536

# Additional useful options:
# --no-mmap: Don't use memory mapping for model loading (can improve performance)
# --mlock: Lock model in memory to prevent swapping
# --pooling cls: Use CLS token pooling for embeddings
# -p 8080: Set server port (default is 8080)
# --host 0.0.0.0: Bind to all network interfaces
```

This will automatically download the specified model from Hugging Face and start the server.

#### Option B: Manual Model Download

If you prefer to download models manually:

1. Download a compatible GGUF model:
```bash
# Example: Download Mistral 7B Instruct
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

2. Start the server with your downloaded model:
```bash
# Basic server start
./server -m models/mistral-7b-instruct-v0.2.Q4_K_M.gguf -c 2048
```

### 3. Configure DeepSearch to Use llama.cpp Server

In your `.env` file, ensure you have:

```
OPENAI_API_BASE=http://localhost:8080/v1
OPENAI_API_KEY=not-needed
LLM_MODEL_ID=gpt-3.5-turbo  # This is just a placeholder, llama.cpp will use the loaded model
```

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

## Architecture

The system uses the following agents:
1. Query Refinement Agent - Improves the original query for better search results
2. Tavily Search Agent - Fetches real-time web search results
3. Dynamic FAISS Indexing Agent - Creates vector embeddings for semantic search
4. Dynamic BM25 Search Agent - Performs keyword-based retrieval
5. Reasoning Agent - Synthesizes information into a coherent answer