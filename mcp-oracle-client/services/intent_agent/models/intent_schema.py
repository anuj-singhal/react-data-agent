# services/intent_agent/models/intent_schema.py
"""Intent schema models for structured intent representation."""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

class IntentType(str, Enum):
    """Enumeration of possible intent types."""
    SIMPLE_SELECT = "simple_select"
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    SORTING = "sorting"
    JOINING = "joining"
    COMPARISON = "comparison"
    TIME_SERIES = "time_series"
    FOLLOW_UP = "follow_up"
    MODIFICATION = "modification"
    EXPORT = "export"
    METADATA = "metadata"  # For "show tables", "describe table"
    UNKNOWN = "unknown"

class OperationType(str, Enum):
    """Types of operations that can be performed."""
    SELECT = "select"
    AGGREGATE = "aggregate"
    GROUP_BY = "group_by"
    SORT = "sort"
    LIMIT = "limit"
    FILTER = "filter"
    JOIN = "join"
    DISTINCT = "distinct"
    COUNT = "count"

class AggregateFunction(str, Enum):
    """Aggregate functions."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"

class Entity(BaseModel):
    """Represents a database entity (table, column, etc.)."""
    type: Literal["table", "column", "metric", "dimension"]
    name: str
    original_text: str  # Original text from user query
    confidence: float = Field(ge=0.0, le=1.0)
    table_context: Optional[str] = None  # For columns, which table they belong to

class Operation(BaseModel):
    """Represents a query operation."""
    type: OperationType
    function: Optional[AggregateFunction] = None
    column: Optional[str] = None
    columns: Optional[List[str]] = None
    direction: Optional[Literal["asc", "desc"]] = None
    value: Optional[Any] = None
    alias: Optional[str] = None

class Filter(BaseModel):
    """Represents a filter condition."""
    column: str
    operator: Literal["=", "!=", ">", "<", ">=", "<=", "in", "not in", "like", "between", "is null", "is not null"]
    value: Optional[Any] = None
    values: Optional[List[Any]] = None  # For IN operations
    logical_operator: Optional[Literal["and", "or"]] = None  # For chaining filters

class TimeContext(BaseModel):
    """Represents temporal context in the query."""
    period: Optional[Literal["day", "week", "month", "quarter", "year"]] = None
    reference: Optional[Literal["current", "last", "next", "specific"]] = None
    specific_value: Optional[str] = None  # For specific dates
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    relative_offset: Optional[int] = None  # e.g., "last 3 months" = -3

class IntentSchema(BaseModel):
    """Complete structured intent representation."""
    intent_type: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Entities
    tables: List[Entity] = []
    columns: List[Entity] = []
    metrics: List[Entity] = []  # Columns used for calculations
    dimensions: List[Entity] = []  # Columns used for grouping
    
    # Operations
    operations: List[Operation] = []
    
    # Filters
    filters: List[Filter] = []
    
    # Time context
    time_context: Optional[TimeContext] = None
    
    # Additional metadata
    original_query: str
    requires_clarification: bool = False
    clarification_questions: List[str] = []
    ambiguities: List[str] = []
    
    # For follow-up intents
    references_previous: bool = False
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Ensure confidence is within valid range."""
        return max(0.0, min(1.0, v))
    
    def to_explanation(self) -> str:
        """Generate human-readable explanation of the intent."""
        explanation = f"I understood you want to perform a {self.intent_type.value} operation"
        
        if self.tables:
            table_names = [t.name for t in self.tables]
            explanation += f" on {', '.join(table_names)}"
        
        if self.intent_type == IntentType.AGGREGATION and self.operations:
            agg_ops = [op for op in self.operations if op.type == OperationType.AGGREGATE]
            if agg_ops:
                funcs = [f"{op.function.value}({op.column})" for op in agg_ops if op.function]
                explanation += f" calculating {', '.join(funcs)}"
        
        if self.filters:
            explanation += f" with {len(self.filters)} filter(s)"
        
        if self.time_context:
            explanation += f" for {self.time_context.reference} {self.time_context.period}"
        
        return explanation
    
    def is_valid(self) -> bool:
        """Check if the intent has minimum required information."""
        if self.intent_type == IntentType.UNKNOWN:
            return False
        
        if self.intent_type == IntentType.METADATA:
            return True  # Metadata queries don't need tables/columns
        
        # Most queries need at least a table
        if not self.tables and self.intent_type != IntentType.FOLLOW_UP:
            return False
        
        # Aggregation needs metrics
        if self.intent_type == IntentType.AGGREGATION and not self.metrics:
            return False
        
        return True

class IntentValidation(BaseModel):
    """Validation result for an intent."""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []