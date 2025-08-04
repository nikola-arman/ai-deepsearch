import fastapi
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.apis import api_router
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

host: str = os.getenv("HOST", "0.0.0.0")
port: int = os.getenv("PORT", 80)

def main():
    server_app = fastapi.FastAPI()

    server_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    server_app.include_router(api_router)

    @server_app.get("/health")
    async def healthcheck():
        return {"status": "ok", "message": "Yo, I am alive"}

    uvicorn.run(server_app, host=host, port=port)

if __name__ == '__main__':
    main()