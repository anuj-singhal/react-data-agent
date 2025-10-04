# api/app.py
"""FastAPI application initialization and configuration."""
from fastapi import FastAPI
from api.lifespan import lifespan
from api.middleware import setup_middleware
from config.settings import settings

# Import routers
from routers import health, query, database, chat

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title=settings.API_TITLE,
        version=settings.API_VERSION,
        description=settings.API_DESCRIPTION,
        lifespan=lifespan
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Include routers
    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(database.router)
    app.include_router(chat.router)  # Added chat router
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint - API information."""
        return {
            "name": settings.API_TITLE,
            "version": settings.API_VERSION,
            "status": "running",
            "endpoints": {
                "health": "/health",
                "query": "/query",
                "chat": "/chat",
                "tables": "/tables",
                "table_info": "/table/{table_name}",
                "docs": "/docs",
                "redoc": "/redoc"
            }
        }
    
    return app