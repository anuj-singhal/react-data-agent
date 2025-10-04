# config/settings.py
"""Application configuration and settings."""
import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings loaded from environment variables."""
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    RELOAD: bool = os.getenv("RELOAD", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    
    # MCP Configuration
    #MCP_SERVER_PATH: str = os.getenv("MCP_SERVER_PATH", "oracle_server.py")
    MCP_SERVER_PATH: str = "./mcp-servers/oracle_server/oracle_server.py"
    MCP_PYTHON_COMMAND: str = os.getenv("MCP_PYTHON_COMMAND", "python")
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "500"))
    
    # API Configuration
    API_TITLE: str = "MCP Oracle Client API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Oracle Database Query Interface via MCP"
    
    # CORS Configuration
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list = ["*"]
    CORS_ALLOW_HEADERS: list = ["*"]
    
    @classmethod
    def validate(cls) -> bool:
        """Validate critical settings."""
        if not Path(cls.MCP_SERVER_PATH).exists():
            print(f"Warning: MCP server script not found at: {cls.MCP_SERVER_PATH}")
            return False
        return True
    
    @classmethod
    def is_openai_configured(cls) -> bool:
        """Check if OpenAI is properly configured."""
        return bool(cls.OPENAI_API_KEY)

settings = Settings()