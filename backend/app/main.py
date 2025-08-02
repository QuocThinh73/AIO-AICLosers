from fastapi import FastAPI
from app.api.endpoints import search

def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(search.router, prefix="/api/search",tags=["search"])
    return app