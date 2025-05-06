from nikolasigmoid/py-agent-infra:latest
copy requirements.txt requirements.txt

run python -m pip install --no-cache-dir -r requirements.txt 
run python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
run python -m pip install --no-cache-dir ultralytics transformers
run python -m pip install --no-cache-dir pillow-heif

run apt-get update && apt-get install -y libgl1

copy app app
copy deepsearch deepsearch
copy system_prompt.txt system_prompt.txt

env ETERNALAI_MCP_PROXY_URL="http://84532-proxy/prompt"
env PROXY_SCOPE="*api.tavily.com*"
env RETRIEVER="tavily"
env PUBMED_EMAIL="daniel@bvm.network"
env TAVILY_API_KEY="tvly-hahaha"
env FORWARD_ALL_MESSAGES=1

env VLM_BASE_URL="https://mac1-9090.eternalai.org/v1"
env VLM_API_KEY="d50b6ba5169ea538a71fe7b0685b755823a3746934fa3cc4k"