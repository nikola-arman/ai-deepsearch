docker build . -t ai_deepsearch --pull

docker run --rm -it \
    --network=network-agent-external  \
    -p 7000:80 \
    --add-host=localmodel:host-gateway \
    --volume $(pwd)/output:/workspace/output \
    -e LLM_BASE_URL="$LLM_BASE_URL" \
    -e LLM_API_KEY="$LLM_API_KEY" \
    -e LLM_MODEL_ID="$LLM_MODEL_ID" \
    -e EMBEDDING_MODEL_ID="$EMBEDDING_MODEL_ID" \
    -e EMBEDDING_URL="$EMBEDDING_URL" \
    -e EMBEDDING_API_KEY="$EMBEDDING_API_KEY" \
    -e TWITTER_API_URL="$TWITTER_API_URL" \
    -e TWITTER_API_KEY="$TWITTER_API_KEY" \
    -e DEBUG_MODE="true" \
    -e LOCAL_TEST=1 \
    -e ETERNALAI_MCP_PROXY_URL=http://localmodel:4001/prompt \
    ai_deepsearch 
