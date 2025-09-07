from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup

    # Running
    yield
    
    # Shutdown

def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    from app.api.endpoints import search
    app.include_router(search.router, prefix="/api/search", tags=["search"])

    return app