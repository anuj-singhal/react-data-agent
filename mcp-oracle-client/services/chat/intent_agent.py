# services/chat/intent_agent.py
"""Advanced Intent Agent for semantic understanding of database queries."""
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from config.settings import settings

logger = logging.getLogger(__name__)

# Optional OpenAI import
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None


class QueryIntent(Enum):
    """Primary query intent types."""
    AGGREGATION = "aggregation"
    COMPARISON = "comparison"
    TREND_ANALYSIS = "trend_analysis"
    FILTERING = "filtering"
    JOINING = "joining"
    FORECASTING = "forecasting"
    LISTING = "listing"
    COUNTING = "counting"
    RANKING = "ranking"
    DETAIL_VIEW = "detail_view"
    STATISTICAL = "statistical"
    DATA_EXPLORATION = "data_exploration"


class TimeGranularity(Enum):
    """Time granularity for temporal queries."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    CUSTOM = "custom"
    NONE = "none"


@dataclass
class SemanticIntent:
    """Structured representation of query intent."""
    primary_intent: str
    secondary_intents: List[str] = None
    domain: Optional[str] = None
    entities: Dict[str, List[str]] = None
    metrics: List[str] = None
    dimensions: List[str] = None
    filters: List[str] = None
    time_range: Optional[Dict[str, str]] = None
    time_granularity: Optional[str] = None
    aggregations: List[str] = None
    sort_criteria: Optional[Dict[str, str]] = None
    limit: Optional[int] = None
    comparison_type: Optional[str] = None
    confidence_score: float = 1.0
    requires_join: bool = False
    is_complex_query: bool = False
    suggested_approach: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None and value != [] and value != {}:
                result[key] = value
        return result


class IntentAgent:
    """Advanced intent extraction and classification for database queries."""
    
    # Domain keywords mapping
    DOMAIN_KEYWORDS = {
        "financial": ["bank", "profit", "revenue", "deposit", "loan", "interest", "nim", "assets", "financial"],
        "sales": ["sales", "sold", "purchase", "order", "customer", "product"],
        "inventory": ["stock", "inventory", "warehouse", "supply"],
        "hr": ["employee", "salary", "department", "manager", "staff"],
        "operations": ["process", "workflow", "efficiency", "performance"]
    }
    
    # Metric indicators
    METRIC_INDICATORS = [
        "total", "sum", "average", "mean", "median", "count", "max", "maximum",
        "min", "minimum", "percentage", "ratio", "rate", "growth", "variance",
        "standard deviation", "profit", "revenue", "cost", "amount"
    ]
    
    # Dimension indicators
    DIMENSION_INDICATORS = [
        "by", "per", "for each", "group by", "categorize", "segment",
        "breakdown", "split by", "across", "within"
    ]
    
    # Time indicators
    TIME_INDICATORS = {
        "daily": ["daily", "per day", "each day", "every day"],
        "weekly": ["weekly", "per week", "each week"],
        "monthly": ["monthly", "per month", "each month", "every month"],
        "quarterly": ["quarterly", "per quarter", "q1", "q2", "q3", "q4"],
        "yearly": ["yearly", "annually", "per year", "each year"]
    }
    
    def __init__(self):
        self.llm = None
        if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
            self.llm = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("OpenAI client initialized for Intent Agent")
    
    async def extract_intent(self, query: str, context: Optional[Dict] = None) -> SemanticIntent:
        """
        Extract structured semantic intent from natural language query.
        
        Args:
            query: User's natural language query
            context: Optional context (previous queries, current filters, etc.)
            
        Returns:
            SemanticIntent object with structured query understanding
        """
        # Try LLM extraction first if available
        if self.llm:
            try:
                intent = await self._extract_with_llm(query, context)
                logger.info(f"LLM extracted intent: {intent.to_dict()}")
                return intent
            except Exception as e:
                logger.error(f"LLM intent extraction failed: {e}")
        
        # Fallback to rule-based extraction
        intent = self._extract_with_rules(query)
        logger.info(f"Rule-based extracted intent: {intent.to_dict()}")
        return intent
    
    async def _extract_with_llm(self, query: str, context: Optional[Dict]) -> SemanticIntent:
        """Extract intent using LLM."""
        
        context_str = json.dumps(context, indent=2) if context else "No previous context"
        
        prompt = f"""Analyze the following database query and extract structured semantic intent.

Query: "{query}"

Previous Context:
{context_str}

Extract and return a JSON object with these fields (only include fields that are relevant):
- primary_intent: Main operation type (aggregation, comparison, trend_analysis, filtering, joining, forecasting, listing, counting, ranking, detail_view, statistical, data_exploration)
- secondary_intents: List of additional intent types if query has multiple goals
- domain: Business domain (financial, sales, inventory, hr, operations, or specific table name)
- entities: Dictionary of entity types and their values (e.g., {{"banks": ["ABC Bank", "XYZ Bank"], "regions": ["North", "South"]}})
- metrics: List of metrics to calculate (e.g., ["total_sales", "average_profit", "count"])
- dimensions: List of dimensions to group/segment by (e.g., ["region", "product_category"])
- filters: List of filter conditions as strings (e.g., ["year = 2024", "region = 'North'"])
- time_range: Dictionary with "start" and "end" if applicable
- time_granularity: Time grouping (hourly, daily, weekly, monthly, quarterly, yearly, none)
- aggregations: Specific aggregation functions needed (sum, avg, count, max, min)
- sort_criteria: Dictionary with "field" and "direction" (asc/desc)
- limit: Number of results to limit (if specified)
- comparison_type: Type of comparison if applicable (period-over-period, cross-entity, benchmark)
- requires_join: Boolean if query needs multiple tables
- is_complex_query: Boolean if query has multiple operations or is complex
- suggested_approach: Brief suggestion on query approach

Important:
- Be precise in identifying metrics vs dimensions
- Extract actual entity names when mentioned
- Preserve exact filter values
- Return valid JSON only"""

        response = await self.llm.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a database query intent analyzer. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        
        try:
            intent_dict = json.loads(response.choices[0].message.content)
            
            # Create SemanticIntent from dictionary
            return SemanticIntent(
                primary_intent=intent_dict.get("primary_intent", "listing"),
                secondary_intents=intent_dict.get("secondary_intents"),
                domain=intent_dict.get("domain"),
                entities=intent_dict.get("entities"),
                metrics=intent_dict.get("metrics"),
                dimensions=intent_dict.get("dimensions"),
                filters=intent_dict.get("filters"),
                time_range=intent_dict.get("time_range"),
                time_granularity=intent_dict.get("time_granularity"),
                aggregations=intent_dict.get("aggregations"),
                sort_criteria=intent_dict.get("sort_criteria"),
                limit=intent_dict.get("limit"),
                comparison_type=intent_dict.get("comparison_type"),
                confidence_score=0.9,  # High confidence for LLM
                requires_join=intent_dict.get("requires_join", False),
                is_complex_query=intent_dict.get("is_complex_query", False),
                suggested_approach=intent_dict.get("suggested_approach")
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Fall back to rule-based
            return self._extract_with_rules(query)
    
    def _extract_with_rules(self, query: str) -> SemanticIntent:
        """Rule-based intent extraction."""
        query_lower = query.lower()
        
        # Determine primary intent
        primary_intent = self._classify_primary_intent(query_lower)
        
        # Extract domain
        domain = self._extract_domain(query_lower)
        
        # Extract metrics and dimensions
        metrics = self._extract_metrics(query_lower)
        dimensions = self._extract_dimensions(query_lower)
        
        # Extract filters
        filters = self._extract_filters(query)
        
        # Extract time information
        time_granularity = self._extract_time_granularity(query_lower)
        time_range = self._extract_time_range(query_lower)
        
        # Extract aggregations
        aggregations = self._extract_aggregations(query_lower)
        
        # Extract sort and limit
        sort_criteria = self._extract_sort_criteria(query_lower)
        limit = self._extract_limit(query_lower)
        
        # Determine complexity
        is_complex = self._is_complex_query(query_lower)
        requires_join = self._requires_join(query_lower)
        
        return SemanticIntent(
            primary_intent=primary_intent,
            domain=domain,
            metrics=metrics if metrics else None,
            dimensions=dimensions if dimensions else None,
            filters=filters if filters else None,
            time_range=time_range,
            time_granularity=time_granularity,
            aggregations=aggregations if aggregations else None,
            sort_criteria=sort_criteria,
            limit=limit,
            confidence_score=0.7,  # Lower confidence for rules
            requires_join=requires_join,
            is_complex_query=is_complex
        )
    
    def _classify_primary_intent(self, query_lower: str) -> str:
        """Classify the primary intent of the query."""
        if any(word in query_lower for word in ["compare", "versus", "vs", "difference"]):
            return QueryIntent.COMPARISON.value
        elif any(word in query_lower for word in ["trend", "over time", "growth", "change"]):
            return QueryIntent.TREND_ANALYSIS.value
        elif any(word in query_lower for word in ["forecast", "predict", "projection"]):
            return QueryIntent.FORECASTING.value
        elif any(word in query_lower for word in ["average", "sum", "total", "mean", "aggregate"]):
            return QueryIntent.AGGREGATION.value
        elif any(word in query_lower for word in ["top", "bottom", "rank", "highest", "lowest"]):
            return QueryIntent.RANKING.value
        elif any(word in query_lower for word in ["count", "how many", "number of"]):
            return QueryIntent.COUNTING.value
        elif any(word in query_lower for word in ["filter", "where", "only", "specific"]):
            return QueryIntent.FILTERING.value
        elif any(word in query_lower for word in ["join", "combine", "merge", "relate"]):
            return QueryIntent.JOINING.value
        elif any(word in query_lower for word in ["statistics", "distribution", "variance"]):
            return QueryIntent.STATISTICAL.value
        elif any(word in query_lower for word in ["show", "list", "display", "get all"]):
            return QueryIntent.LISTING.value
        else:
            return QueryIntent.DATA_EXPLORATION.value
    
    def _extract_domain(self, query_lower: str) -> Optional[str]:
        """Extract the business domain from the query."""
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        
        # Check for table names
        if "uae_banks" in query_lower:
            return "financial"
        
        return None
    
    def _extract_metrics(self, query_lower: str) -> List[str]:
        """Extract metrics from the query."""
        metrics = []
        
        # Common metric patterns
        metric_patterns = {
            "total_sales": ["total sales", "sum of sales"],
            "average_profit": ["average profit", "mean profit", "avg profit"],
            "count": ["count", "number of", "how many"],
            "maximum": ["maximum", "max", "highest"],
            "minimum": ["minimum", "min", "lowest"]
        }
        
        for metric_name, patterns in metric_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                metrics.append(metric_name)
        
        # Extract specific column names that might be metrics
        if "profit" in query_lower and "profit" not in str(metrics):
            metrics.append("profit")
        if "revenue" in query_lower:
            metrics.append("revenue")
        if "deposits" in query_lower:
            metrics.append("deposits")
        if "nim" in query_lower:
            metrics.append("nim")
        
        return metrics
    
    def _extract_dimensions(self, query_lower: str) -> List[str]:
        """Extract dimensions for grouping."""
        dimensions = []
        
        # Look for dimension indicators
        if "by region" in query_lower:
            dimensions.append("region")
        if "by bank" in query_lower or "per bank" in query_lower:
            dimensions.append("bank")
        if "by category" in query_lower:
            dimensions.append("category")
        if "by month" in query_lower:
            dimensions.append("month")
        if "by year" in query_lower:
            dimensions.append("year")
        if "by department" in query_lower:
            dimensions.append("department")
        
        return dimensions
    
    def _extract_filters(self, query: str) -> List[str]:
        """Extract filter conditions."""
        filters = []
        
        # Common filter patterns
        if ">" in query or "<" in query or "=" in query:
            # Simple extraction of comparison operators
            import re
            comparisons = re.findall(r'\w+\s*[><=]+\s*[\w\d]+', query)
            filters.extend(comparisons)
        
        # Year filters
        import re
        years = re.findall(r'\b20\d{2}\b', query)
        for year in years:
            filters.append(f"year = {year}")
        
        return filters
    
    def _extract_time_granularity(self, query_lower: str) -> Optional[str]:
        """Extract time granularity."""
        for granularity, indicators in self.TIME_INDICATORS.items():
            if any(indicator in query_lower for indicator in indicators):
                return granularity
        return None
    
    def _extract_time_range(self, query_lower: str) -> Optional[Dict[str, str]]:
        """Extract time range if specified."""
        import re
        
        # Look for date patterns
        date_pattern = r'\b(\d{4}-\d{2}-\d{2})\b'
        dates = re.findall(date_pattern, query_lower)
        
        if len(dates) >= 2:
            return {"start": dates[0], "end": dates[1]}
        elif len(dates) == 1:
            return {"start": dates[0], "end": "current"}
        
        # Look for relative time
        if "last month" in query_lower:
            return {"start": "last_month", "end": "last_month"}
        elif "last year" in query_lower:
            return {"start": "last_year", "end": "last_year"}
        elif "this year" in query_lower:
            return {"start": "year_start", "end": "current"}
        
        return None
    
    def _extract_aggregations(self, query_lower: str) -> List[str]:
        """Extract aggregation functions."""
        aggregations = []
        
        agg_functions = {
            "sum": ["sum", "total"],
            "avg": ["average", "avg", "mean"],
            "count": ["count", "number of"],
            "max": ["maximum", "max", "highest"],
            "min": ["minimum", "min", "lowest"]
        }
        
        for agg_func, keywords in agg_functions.items():
            if any(keyword in query_lower for keyword in keywords):
                aggregations.append(agg_func)
        
        return aggregations
    
    def _extract_sort_criteria(self, query_lower: str) -> Optional[Dict[str, str]]:
        """Extract sorting criteria."""
        if "top" in query_lower or "highest" in query_lower:
            return {"direction": "desc"}
        elif "bottom" in query_lower or "lowest" in query_lower:
            return {"direction": "asc"}
        elif "order by" in query_lower:
            if "desc" in query_lower:
                return {"direction": "desc"}
            else:
                return {"direction": "asc"}
        return None
    
    def _extract_limit(self, query_lower: str) -> Optional[int]:
        """Extract result limit."""
        import re
        
        # Pattern for "top N" or "first N"
        top_pattern = r'(?:top|first|last)\s+(\d+)'
        match = re.search(top_pattern, query_lower)
        if match:
            return int(match.group(1))
        
        # Pattern for "limit N"
        limit_pattern = r'limit\s+(\d+)'
        match = re.search(limit_pattern, query_lower)
        if match:
            return int(match.group(1))
        
        return None
    
    def _is_complex_query(self, query_lower: str) -> bool:
        """Determine if query is complex."""
        complexity_indicators = [
            "and also", "as well as", "in addition", "furthermore",
            "compare", "analyze", "correlation", "regression"
        ]
        return any(indicator in query_lower for indicator in complexity_indicators)
    
    def _requires_join(self, query_lower: str) -> bool:
        """Determine if query requires joining tables."""
        join_indicators = [
            "join", "combine", "merge", "from multiple", "across tables",
            "relate", "connection between", "link"
        ]
        return any(indicator in query_lower for indicator in join_indicators)


# Integration with existing ChatManager
class EnhancedIntentClassifier:
    """
    Enhanced intent classifier that combines existing classification
    with semantic intent extraction.
    """
    
    def __init__(self, existing_classifier, intent_agent):
        self.existing_classifier = existing_classifier
        self.intent_agent = intent_agent
    
    async def classify_and_extract(
        self, 
        message: str, 
        context: Any
    ) -> Tuple[str, SemanticIntent]:
        """
        Classify conversation intent and extract semantic query intent.
        
        Returns:
            Tuple of (conversation_intent, semantic_intent)
        """
        # Get conversation intent (follow_up, new_query, etc.)
        conversation_intent = await self.existing_classifier.classify_intent(message, context)
        
        # Extract semantic intent if it's a new query or modification
        semantic_intent = None
        if conversation_intent in ["new_query", "modify_query", "execute_immediately"]:
            semantic_intent = await self.intent_agent.extract_intent(
                message,
                {
                    "last_sql": context.last_sql if context else None,
                    "last_result_columns": context.last_result.get("columns") if context and context.last_result else None
                }
            )
        
        return conversation_intent, semantic_intent