from python:3.12-slim 

run apt-get update && apt-get install -y redis-server git

copy requirements.txt requirements.txt
run python -m pip install -r requirements.txt

workdir /workspace
env APP_ENV=production

copy app app
copy deepsearch deepsearch
copy system_prompt.txt system_prompt.txt
copy server.py server.py

env PROXY_SCOPE="*api.tavily.com*,*api.search.brave.com*,*api.exa.ai*,*imagine-backend.bvm.network*"
env RETRIEVER="brave,tavily,exa,twitter"
env TWITTER_API_URL="https://imagine-backend.bvm.network/api/internal/twitter/"
env FORWARD_ALL_MESSAGES=1
env CACHE_DB_FOLDER="/storage"

cmd ["python", "server.py"]