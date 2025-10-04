# api/middleware.py
"""Middleware configuration for FastAPI app."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings

def setup_middleware(app: FastAPI):
    """Configure middleware for the application."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )
    
    # Add more middleware here as needed
    # e.g., Authentication, Rate limiting, etc.