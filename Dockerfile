from nikolasigmoid/py-agent-infra:latest

copy requirements.txt requirements.txt
run --mount=type=cache,target=/root/.cache/pip python -m pip install -r requirements.txt

copy app app
copy deepsearch deepsearch
copy system_prompt.txt system_prompt.txt

env PROXY_SCOPE="*api.tavily.com*,*api.search.brave.com*,*api.exa.ai*"
env RETRIEVER="brave,tavily,exa"
env FORWARD_ALL_MESSAGES=1
