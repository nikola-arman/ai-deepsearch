from nikolasigmoid/py-agent-infra:latest

copy requirements.txt requirements.txt

run python -m pip install --no-cache-dir -r requirements.txt

copy app app
copy deepsearch deepsearch

env PROXY_SCOPE="*api.tavily.com*,*api.search.brave.com*,*api.exa.ai*"
env RETRIEVER="brave,tavily,exa"
env FORWARD_ALL_MESSAGES=1
