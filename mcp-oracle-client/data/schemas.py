# data/schemas.py
"""Database schemas and data dictionaries."""

DATA_DICTIONARY = {
    "uae_banks_financial_data": {
        "description": "Contains quarterly financial performance metrics for UAE banks",
        "columns": {
            "year": "Financial reporting year",
            "quarter": "Quarter 1-4",
            "bank": "Bank name",
            "ytd_income": "Year-to-date total operating income",
            "ytd_profit": "Year-to-date net profit",
            "loans_advances": "Total loans and advances",
            "nim": "Net Interest Margin",
            "deposits": "Total customer deposits",
            "casa": "Current Account and Savings Account deposits",
            "cost_income": "Cost-to-Income ratio",
            "npl_ratio": "Non-Performing Loans ratio",
            "rote": "Return on Tangible Equity",
            "cet1": "Common Equity Tier 1 capital ratio",
            "share_price": "Market share price per share",
        }
    }
}

# SQL Pattern mappings for fallback conversion
SQL_PATTERNS = {
    "show tables": "SELECT table_name FROM user_tables",
    "list tables": "SELECT table_name FROM user_tables",
    "test connection": "SELECT USER FROM dual",
    "show users": "SELECT username FROM all_users",
    "current user": "SELECT USER FROM dual",
    "database version": "SELECT banner FROM v$version WHERE ROWNUM = 1",
}

def get_table_schema(table_name: str) -> dict:
    """Get schema for a specific table."""
    return DATA_DICTIONARY.get(table_name, {})

def get_all_tables() -> list:
    """Get list of all known tables."""
    return list(DATA_DICTIONARY.keys())

def get_columns_for_table(table_name: str) -> dict:
    """Get columns for a specific table."""
    schema = get_table_schema(table_name)
    return schema.get("columns", {})