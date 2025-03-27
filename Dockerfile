from nikolasigmoid/py-agent-infra:latest

copy app app
copy deepsearch deepsearch
copy requirements.txt requirements.txt

run python -m pip install --no-cache-dir -r requirements.txt

env ETERNALAI_MCP_PROXY_URL="https://agent-service-client.dev.eternalai.org/execution"
env PROXY_SCOPE="*api.tavily.com*"
env RETRIEVER="tavily"