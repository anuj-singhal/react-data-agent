# services/intent_agent/extractors/filter_extractor.py
"""Filter extraction for database queries."""
import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from services.intent_agent.models.intent_schema import Filter

logger = logging.getLogger(__name__)

class FilterExtractor:
    """Extracts filter conditions from natural language queries."""
    
    # Operator mappings
    OPERATOR_PATTERNS = {
        "=": ["equals", "equal to", "is", "=", "=="],
        "!=": ["not equals", "not equal to", "is not", "!=", "<>"],
        ">": ["greater than", "more than", "above", "over", ">", "exceeds"],
        "<": ["less than", "below", "under", "<"],
        ">=": ["greater than or equal", "at least", ">=", "minimum"],
        "<=": ["less than or equal", "at most", "<=", "maximum"],
        "like": ["contains", "like", "includes", "matching"],
        "between": ["between", "range", "from.*to"],
        "in": ["in", "among", "one of"],
        "is null": ["is null", "is empty", "has no value"],
        "is not null": ["is not null", "has value", "is not empty"]
    }
    
    def extract_filters(self, query: str, entities: Dict) -> List[Filter]:
        """
        Extract filter conditions from the query.
        
        Args:
            query: Natural language query
            entities: Extracted entities for context
            
        Returns:
            List of filters
        """
        query_lower = query.lower()
        filters = []
        
        # Extract different types of filters
        filters.extend(self._extract_comparison_filters(query_lower, entities))
        filters.extend(self._extract_range_filters(query_lower, entities))
        filters.extend(self._extract_in_filters(query_lower, entities))
        filters.extend(self._extract_null_filters(query_lower, entities))
        filters.extend(self._extract_pattern_filters(query_lower, entities))
        
        # Extract time-based filters
        filters.extend(self._extract_time_filters(query_lower))
        
        # Deduplicate filters
        filters = self._deduplicate_filters(filters)
        
        return filters
    
    def _extract_comparison_filters(self, query: str, entities: Dict) -> List[Filter]:
        """Extract comparison filters (=, !=, >, <, >=, <=)."""
        filters = []
        
        # Patterns for comparison operations
        patterns = [
            r"(\w+)\s+(?:is\s+)?(?:greater than|more than|above|over|>)\s+(\d+(?:\.\d+)?)",
            r"(\w+)\s+(?:is\s+)?(?:less than|below|under|<)\s+(\d+(?:\.\d+)?)",
            r"(\w+)\s+(?:is\s+)?(?:equals?|equal to|is|=)\s+['\"]?(\w+)['\"]?",
            r"(\w+)\s+(?:is\s+)?(?:not equals?|not equal to|is not|!=|<>)\s+['\"]?(\w+)['\"]?",
            r"(\w+)\s*(?:>=?|<=?|=)\s*(\d+(?:\.\d+)?)",
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                column = self._validate_column(match.group(1).lower(), entities)
                if column:
                    operator = self._determine_operator(match.group(0))
                    value = self._parse_value(match.group(2))
                    
                    filters.append(Filter(
                        column=column,
                        operator=operator,
                        value=value
                    ))
        
        return filters
    
    def _extract_range_filters(self, query: str, entities: Dict) -> List[Filter]:
        """Extract BETWEEN filters."""
        filters = []
        
        # Pattern for between operations
        pattern = r"(\w+)\s+(?:is\s+)?between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)"
        matches = re.finditer(pattern, query, re.IGNORECASE)
        
        for match in matches:
            column = self._validate_column(match.group(1).lower(), entities)
            if column:
                filters.append(Filter(
                    column=column,
                    operator="between",
                    values=[
                        self._parse_value(match.group(2)),
                        self._parse_value(match.group(3))
                    ]
                ))
        
        return filters
    
    def _extract_in_filters(self, query: str, entities: Dict) -> List[Filter]:
        """Extract IN filters."""
        filters = []
        
        # Pattern for IN operations
        pattern = r"(\w+)\s+(?:is\s+)?(?:in|among|one of)\s+\(([^)]+)\)"
        matches = re.finditer(pattern, query, re.IGNORECASE)
        
        for match in matches:
            column = self._validate_column(match.group(1).lower(), entities)
            if column:
                # Parse the list of values
                values_str = match.group(2)
                values = [self._parse_value(v.strip().strip("'\"")) 
                         for v in values_str.split(',')]
                
                filters.append(Filter(
                    column=column,
                    operator="in",
                    values=values
                ))
        
        return filters
    
    def _extract_null_filters(self, query: str, entities: Dict) -> List[Filter]:
        """Extract NULL/NOT NULL filters."""
        filters = []
        
        # Patterns for null checks
        null_patterns = [
            (r"(\w+)\s+is\s+(?:null|empty)", "is null"),
            (r"(\w+)\s+is\s+not\s+(?:null|empty)", "is not null"),
            (r"(\w+)\s+has\s+no\s+value", "is null"),
            (r"(\w+)\s+has\s+value", "is not null")
        ]
        
        for pattern, operator in null_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                column = self._validate_column(match.group(1).lower(), entities)
                if column:
                    filters.append(Filter(
                        column=column,
                        operator=operator,
                        value=None
                    ))
        
        return filters
    
    def _extract_pattern_filters(self, query: str, entities: Dict) -> List[Filter]:
        """Extract LIKE pattern filters."""
        filters = []
        
        # Patterns for LIKE operations
        pattern = r"(\w+)\s+(?:contains|includes|like)\s+['\"]?([^'\"]+)['\"]?"
        matches = re.finditer(pattern, query, re.IGNORECASE)
        
        for match in matches:
            column = self._validate_column(match.group(1).lower(), entities)
            if column:
                value = f"%{match.group(2)}%"  # Add wildcards
                filters.append(Filter(
                    column=column,
                    operator="like",
                    value=value
                ))
        
        return filters
    
    def _extract_time_filters(self, query: str) -> List[Filter]:
        """Extract time-based filters."""
        filters = []
        
        # Current date reference
        now = datetime.now()
        current_year = now.year
        current_quarter = (now.month - 1) // 3 + 1
        
        # Time patterns
        time_patterns = {
            "current year": ("year", "=", current_year),
            "this year": ("year", "=", current_year),
            "last year": ("year", "=", current_year - 1),
            "previous year": ("year", "=", current_year - 1),
            r"year (\d{4})": ("year", "=", None),  # Will extract year
            f"q{current_quarter}": ("quarter", "=", current_quarter),
            "current quarter": ("quarter", "=", current_quarter),
            "last quarter": ("quarter", "=", current_quarter - 1 if current_quarter > 1 else 4),
            r"q(\d)": ("quarter", "=", None),  # Will extract quarter
            r"quarter (\d)": ("quarter", "=", None),
        }
        
        for pattern, (column, operator, value) in time_patterns.items():
            if value is None:
                # Extract value from pattern
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    value = int(match.group(1))
                else:
                    continue
            elif not re.search(pattern, query, re.IGNORECASE):
                continue
            
            filters.append(Filter(
                column=column,
                operator=operator,
                value=value
            ))
        
        return filters
    
    def _validate_column(self, column: str, entities: Dict) -> Optional[str]:
        """Validate if a column exists in the entities."""
        # Check in columns
        if entities.get('columns'):
            for col in entities['columns']:
                if col.name == column:
                    return column
        
        # Check in metrics
        if entities.get('metrics'):
            for metric in entities['metrics']:
                if metric.name == column:
                    return column
        
        # Check in dimensions
        if entities.get('dimensions'):
            for dim in entities['dimensions']:
                if dim.name == column:
                    return column
        
        # Common column names that might not be extracted as entities
        common_columns = ['year', 'quarter', 'month', 'date']
        if column in common_columns:
            return column
        
        return None
    
    def _determine_operator(self, text: str) -> str:
        """Determine the operator from text."""
        text_lower = text.lower()
        
        for operator, patterns in self.OPERATOR_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return operator
        
        return "="  # Default
    
    def _parse_value(self, value_str: str) -> Any:
        """Parse string value to appropriate type."""
        value_str = value_str.strip().strip("'\"")
        
        # Try to parse as number
        try:
            if '.' in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass
        
        # Boolean values
        if value_str.lower() in ['true', 'false']:
            return value_str.lower() == 'true'
        
        # Return as string
        return value_str
    
    def _deduplicate_filters(self, filters: List[Filter]) -> List[Filter]:
        """Remove duplicate filters."""
        seen = set()
        unique_filters = []
        
        for filter in filters:
            # Create a simple hash for the filter
            filter_key = (filter.column, filter.operator, str(filter.value))
            if filter_key not in seen:
                seen.add(filter_key)
                unique_filters.append(filter)
        
        return unique_filters
    
    def add_logical_operators(self, filters: List[Filter]) -> List[Filter]:
        """
        Add logical operators (AND/OR) between filters.
        Default to AND unless specified otherwise.
        """
        if len(filters) <= 1:
            return filters
        
        # Default to AND between filters
        for i in range(len(filters) - 1):
            if filters[i].logical_operator is None:
                filters[i].logical_operator = "and"
        
        return filters