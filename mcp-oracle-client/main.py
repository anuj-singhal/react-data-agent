# main.py
"""Main entry point for the MCP Oracle Client application."""
import sys
import uvicorn
from config.settings import settings
from config.logging_config import setup_logging
from api.app import create_app

# Setup logging
logger = setup_logging()

def main():
    """Main entry point."""
    try:
        # Validate settings
        if not settings.validate():
            logger.warning("Settings validation failed, but continuing...")
        
        logger.info("Starting FastAPI server...")
        logger.info(f"Server will be available at: http://{settings.HOST}:{settings.PORT}")
        logger.info(f"API documentation at: http://{settings.HOST}:{settings.PORT}/docs")
        logger.info(f"Debug tools list at: http://{settings.HOST}:{settings.PORT}/health/tools")
        
        # Create and run the app
        app = create_app()
        
        uvicorn.run(
            app,
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.RELOAD,
            log_level=settings.LOG_LEVEL
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()