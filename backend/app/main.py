from fastapi import FastAPI
from backend.app.api.endpoints import search
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    
    # Running
    yield
    
    # Shutdown


def create_app() -> FastAPI:
    # Initialize app
    app = FastAPI(lifespan=lifespan)

    # Add routers
    app.include_router(search.router, prefix="/api/search", tags=["search"])
    
    return app