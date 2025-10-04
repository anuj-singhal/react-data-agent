# services/intent_agent/parsers/time_parser.py
"""Time reference parsing for queries."""
import re
import logging
from datetime import datetime, timedelta
from typing import Optional
from services.intent_agent.models.intent_schema import TimeContext

logger = logging.getLogger(__name__)

class TimeParser:
    """Parses temporal references in queries."""
    
    def parse_time_context(self, query: str) -> Optional[TimeContext]:
        """
        Parse time-related information from the query.
        
        Args:
            query: Natural language query
            
        Returns:
            TimeContext if time references found, None otherwise
        """
        query_lower = query.lower()
        
        # Check for various time patterns
        context = None
        
        # Check for period references
        period = self._extract_period(query_lower)
        
        # Check for relative references
        reference = self._extract_reference(query_lower)
        
        # Check for specific dates
        specific_value = self._extract_specific_date(query_lower)
        
        # Check for date ranges
        start_date, end_date = self._extract_date_range(query_lower)
        
        # Check for relative offsets
        offset = self._extract_relative_offset(query_lower)
        
        # Build time context if any time information found
        if any([period, reference, specific_value, start_date, end_date, offset]):
            context = TimeContext(
                period=period,
                reference=reference,
                specific_value=specific_value,
                start_date=start_date,
                end_date=end_date,
                relative_offset=offset
            )
        
        return context
    
    def _extract_period(self, query: str) -> Optional[str]:
        """Extract time period (day, week, month, quarter, year)."""
        periods = ["day", "week", "month", "quarter", "year"]
        
        for period in periods:
            if period in query:
                return period
        
        # Check for abbreviations
        if "q1" in query or "q2" in query or "q3" in query or "q4" in query:
            return "quarter"
        
        return None
    
    def _extract_reference(self, query: str) -> Optional[str]:
        """Extract time reference (current, last, next, specific)."""
        references = {
            "current": ["current", "this", "present", "now", "today"],
            "last": ["last", "previous", "past", "prior", "yesterday"],
            "next": ["next", "upcoming", "future", "tomorrow"]
        }
        
        for ref_type, keywords in references.items():
            if any(keyword in query for keyword in keywords):
                return ref_type
        
        # Check if specific dates are mentioned
        if re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', query):
            return "specific"
        
        return None
    
    def _extract_specific_date(self, query: str) -> Optional[str]:
        """Extract specific date references."""
        # Pattern for dates
        date_patterns = [
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # YYYY-MM-DD or YYYY/MM/DD
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',  # MM-DD-YYYY or MM/DD/YYYY
            r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4})',  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_date_range(self, query: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """Extract date range from query."""
        start_date = None
        end_date = None
        
        # Pattern for date ranges
        range_pattern = r'(?:from|between)\s+([^to]+?)\s+(?:to|and)\s+([^\s]+)'
        match = re.search(range_pattern, query, re.IGNORECASE)
        
        if match:
            start_str = match.group(1).strip()
            end_str = match.group(2).strip()
            
            # Try to parse dates
            start_date = self._parse_date_string(start_str)
            end_date = self._parse_date_string(end_str)
        
        return start_date, end_date
    
    def _extract_relative_offset(self, query: str) -> Optional[int]:
        """Extract relative time offset (e.g., 'last 3 months' = -3)."""
        # Pattern for relative offsets
        pattern = r'(?:last|past|previous|next)\s+(\d+)\s+(?:day|week|month|quarter|year)s?'
        match = re.search(pattern, query, re.IGNORECASE)
        
        if match:
            offset = int(match.group(1))
            # Negative for past, positive for future
            if any(word in match.group(0).lower() for word in ['last', 'past', 'previous']):
                return -offset
            return offset
        
        return None
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Try to parse a date string into datetime object."""
        # Common date formats
        formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%m-%d-%Y',
            '%m/%d/%Y',
            '%B %d, %Y',
            '%b %d, %Y',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def calculate_date_filters(self, context: TimeContext) -> tuple[Optional[datetime], Optional[datetime]]:
        """
        Calculate actual date filters from time context.
        
        Args:
            context: TimeContext object
            
        Returns:
            Tuple of (start_date, end_date)
        """
        now = datetime.now()
        start_date = None
        end_date = None
        
        if context.start_date and context.end_date:
            return context.start_date, context.end_date
        
        if context.period and context.reference:
            if context.period == "year":
                if context.reference == "current":
                    start_date = datetime(now.year, 1, 1)
                    end_date = datetime(now.year, 12, 31)
                elif context.reference == "last":
                    start_date = datetime(now.year - 1, 1, 1)
                    end_date = datetime(now.year - 1, 12, 31)
            
            elif context.period == "quarter":
                current_quarter = (now.month - 1) // 3 + 1
                if context.reference == "current":
                    quarter_start_month = (current_quarter - 1) * 3 + 1
                    start_date = datetime(now.year, quarter_start_month, 1)
                    
                    # Calculate end of quarter
                    if current_quarter == 4:
                        end_date = datetime(now.year, 12, 31)
                    else:
                        next_quarter_start = datetime(now.year, quarter_start_month + 3, 1)
                        end_date = next_quarter_start - timedelta(days=1)
                
                elif context.reference == "last":
                    if current_quarter == 1:
                        # Last quarter of previous year
                        start_date = datetime(now.year - 1, 10, 1)
                        end_date = datetime(now.year - 1, 12, 31)
                    else:
                        quarter_start_month = (current_quarter - 2) * 3 + 1
                        start_date = datetime(now.year, quarter_start_month, 1)
                        next_quarter_start = datetime(now.year, quarter_start_month + 3, 1)
                        end_date = next_quarter_start - timedelta(days=1)
            
            elif context.period == "month":
                if context.reference == "current":
                    start_date = datetime(now.year, now.month, 1)
                    # Last day of current month
                    if now.month == 12:
                        end_date = datetime(now.year, 12, 31)
                    else:
                        end_date = datetime(now.year, now.month + 1, 1) - timedelta(days=1)
                
                elif context.reference == "last":
                    if now.month == 1:
                        start_date = datetime(now.year - 1, 12, 1)
                        end_date = datetime(now.year - 1, 12, 31)
                    else:
                        start_date = datetime(now.year, now.month - 1, 1)
                        end_date = datetime(now.year, now.month, 1) - timedelta(days=1)
        
        return start_date, end_date