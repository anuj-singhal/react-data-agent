# models/requests.py
"""Request models for API endpoints."""
from pydantic import BaseModel, Field, validator

class QueryRequest(BaseModel):
    """Request model for query execution."""
    query: str = Field(..., min_length=1, description="Natural language or SQL query")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query is not empty."""
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class TableInfoRequest(BaseModel):
    """Request model for table information."""
    table_name: str = Field(..., min_length=1, description="Name of the table")
    
    @validator('table_name')
    def validate_table_name(cls, v):
        """Validate table name."""
        if not v or not v.strip():
            raise ValueError('Table name cannot be empty')
        # Remove any potential SQL injection attempts
        cleaned = v.strip().replace(';', '').replace('--', '')
        return cleaned