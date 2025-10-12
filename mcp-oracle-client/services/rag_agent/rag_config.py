"""
Enhanced configuration with Multi-LLM support for Oracle MCP Client.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

@dataclass
class RAGConfig:
    """Advanced RAG system configuration."""
    persist_directory: str = "./vector_db/chroma_db"
    collection_name: str = "oracle_queries_advanced"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Schema-aware retrieval
    enable_schema_graph: bool = True
    max_graph_depth: int = 3
    min_similarity_threshold: float = 0.7
    
    # Multi-stage retrieval
    enable_hybrid_search: bool = True
    keyword_weight: float = 0.3
    semantic_weight: float = 0.7
    
    # Validation thresholds
    schema_match_threshold: float = 0.9
    semantic_alignment_threshold: float = 0.8
    syntax_validity_threshold: float = 1.0
    
    # Query pattern matching
    enable_pattern_matching: bool = True
    pattern_similarity_threshold: float = 0.85
    
    # Caching
    enable_query_cache: bool = True
    cache_ttl_seconds: int = 3600

# Enhanced Data Dictionary with relationships
DATA_DICTIONARY: Dict[str, Dict[str, Any]] = {
    "banks": {
        "description": "Master table containing bank information",
        "columns": {
            "bank_id": {
                "description": "Unique identifier for each bank",
                "data_type": "NUMBER",
                "constraints": ["PRIMARY KEY", "NOT NULL"]
            },
            "bank_name": {
                "description": "Full legal name of the bank",
                "data_type": "VARCHAR2(100)",
                "constraints": ["NOT NULL"]
            },
            "bank_code": {
                "description": "Standard bank code or ticker symbol",
                "data_type": "VARCHAR2(10)",
                "constraints": ["NOT NULL", "UNIQUE"]
            }
        },
        "relationships": {
            "financial_performance": "ONE_TO_MANY",
            "market_data": "ONE_TO_MANY"
        },
        "common_queries": [
            "List all banks",
            "Find bank by code",
            "Bank performance comparison"
        ]
    },
    
    "financial_performance": {
        "description": "Quarterly financial performance metrics for banks",
        "columns": {
            "performance_id": {
                "description": "Unique identifier for each record",
                "data_type": "NUMBER",
                "constraints": ["PRIMARY KEY", "NOT NULL"]
            },
            "year": {
                "description": "Financial reporting year",
                "data_type": "NUMBER",
                "constraints": ["NOT NULL"]
            },
            "quarter": {
                "description": "Reporting quarter (1-4)",
                "data_type": "NUMBER",
                "constraints": ["NOT NULL", "CHECK (quarter BETWEEN 1 AND 4)"]
            },
            "bank_id": {
                "description": "Foreign key reference to banks table",
                "data_type": "NUMBER",
                "constraints": ["NOT NULL", "FOREIGN KEY REFERENCES banks(bank_id)"]
            },
            "ytd_income": {
                "description": "Year-to-date total operating income",
                "data_type": "NUMBER"
            },
            "ytd_profit": {
                "description": "Year-to-date net profit",
                "data_type": "NUMBER"
            },
            "quarterly_profit": {
                "description": "Net profit for the specific quarter",
                "data_type": "NUMBER"
            },
            "loans_advances": {
                "description": "Total loans and advances portfolio",
                "data_type": "NUMBER"
            },
            "nim": {
                "description": "Net Interest Margin percentage",
                "data_type": "NUMBER"
            },
            "deposits": {
                "description": "Total customer deposits",
                "data_type": "NUMBER"
            },
            "casa": {
                "description": "Current Account and Savings Account deposits",
                "data_type": "NUMBER"
            },
            "cost_income": {
                "description": "Cost-to-Income ratio percentage",
                "data_type": "NUMBER"
            },
            "npl_ratio": {
                "description": "Non-Performing Loans ratio percentage",
                "data_type": "NUMBER"
            },
            "cor": {
                "description": "Cost of Risk percentage",
                "data_type": "NUMBER"
            },
            "stage3_cover": {
                "description": "Stage 3 NPL coverage ratio percentage",
                "data_type": "NUMBER"
            },
            "rote": {
                "description": "Return on Tangible Equity percentage",
                "data_type": "NUMBER"
            }
        },
        "relationships": {
            "banks": "MANY_TO_ONE"
        },
        "indexes": ["year", "quarter", "bank_id"],
        "common_queries": [
            "Quarterly performance trends",
            "Year-over-year comparison",
            "Profitability analysis",
            "Asset quality metrics"
        ]
    },
    
    "market_data": {
        "description": "Market valuation and capital adequacy metrics",
        "columns": {
            "market_id": {
                "description": "Unique identifier for each record",
                "data_type": "NUMBER",
                "constraints": ["PRIMARY KEY", "NOT NULL"]
            },
            "year": {
                "description": "Financial reporting year",
                "data_type": "NUMBER",
                "constraints": ["NOT NULL"]
            },
            "quarter": {
                "description": "Reporting quarter (1-4)",
                "data_type": "NUMBER",
                "constraints": ["NOT NULL", "CHECK (quarter BETWEEN 1 AND 4)"]
            },
            "bank_id": {
                "description": "Foreign key reference to banks table",
                "data_type": "NUMBER",
                "constraints": ["NOT NULL", "FOREIGN KEY REFERENCES banks(bank_id)"]
            },
            "cet1": {
                "description": "Common Equity Tier 1 ratio percentage",
                "data_type": "NUMBER"
            },
            "cet_capital": {
                "description": "Common Equity Tier 1 capital amount",
                "data_type": "NUMBER"
            },
            "rwa": {
                "description": "Risk-Weighted Assets",
                "data_type": "NUMBER"
            },
            "share_price": {
                "description": "Share price at quarter end",
                "data_type": "NUMBER"
            },
            "market_cap_aed_bn": {
                "description": "Market capitalization in AED billions",
                "data_type": "NUMBER"
            },
            "market_cap_usd_bn": {
                "description": "Market capitalization in USD billions",
                "data_type": "NUMBER"
            }
        },
        "relationships": {
            "banks": "MANY_TO_ONE"
        },
        "indexes": ["year", "quarter", "bank_id"],
        "common_queries": [
            "Market capitalization trends",
            "Capital adequacy analysis",
            "Share price performance"
        ]
    }
}

# Query Pattern Templates
QUERY_PATTERNS = {
    "simple_select": {
        "pattern": "SELECT {columns} FROM {table} WHERE {conditions}",
        "description": "Basic selection with filtering"
    },
    "join_query": {
        "pattern": "SELECT {columns} FROM {table1} JOIN {table2} ON {join_condition} WHERE {conditions}",
        "description": "Join between two tables"
    },
    "aggregation": {
        "pattern": "SELECT {group_columns}, {agg_functions} FROM {table} GROUP BY {group_columns}",
        "description": "Aggregation with grouping"
    },
    "window_function": {
        "pattern": "SELECT {columns}, {window_function} OVER (PARTITION BY {partition} ORDER BY {order}) FROM {table}",
        "description": "Window function analysis"
    }
}