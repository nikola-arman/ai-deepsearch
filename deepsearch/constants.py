import os

LLM_API_KEY = os.getenv('LLM_API_KEY', 'no-need')
LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'no-need').rstrip('/')
LLM_MODEL_ID = os.getenv('LLM_MODEL_ID', 'no-need')

EMBEDDING_MODEL_ID=os.getenv('EMBEDDING_MODEL_ID', 'no-need')
EMBEDDING_URL=os.getenv('EMBEDDING_URL', 'no-need')
EMBEDDING_API_KEY=os.getenv('EMBEDDING_API_KEY', 'no-need')

TWITTER_API_URL=os.getenv('TWITTER_API_URL', 'https://imagine-backend.bvm.network/api/internal/twitter/')
TWITTER_API_KEY=os.getenv('TWITTER_API_KEY', 'no-need')

CACHE_DB_FOLDER = os.getenv('CACHE_DB_FOLDER', '/storage')
