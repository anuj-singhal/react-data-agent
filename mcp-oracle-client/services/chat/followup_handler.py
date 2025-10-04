# services/chat/followup_handler.py
"""Follow-up question handling for chat conversations."""
import logging
import json
from typing import Dict, Any, Optional
from config.settings import settings
from models.chat import ChatResponse

logger = logging.getLogger(__name__)

# Optional OpenAI import
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

class FollowUpHandler:
    """Handles follow-up questions on query results."""
    
    def __init__(self):
        self.llm = None
        if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
            self.llm = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("OpenAI client initialized for follow-up handling")
    
    async def handle_follow_up(
        self,
        question: str,
        result: Dict[str, Any],
        last_sql: Optional[str],
        session_id: str
    ) -> ChatResponse:
        """
        Handle follow-up question on existing results.
        
        Args:
            question: User's follow-up question
            result: Query result data
            last_sql: Last executed SQL query
            session_id: Session identifier
            
        Returns:
            ChatResponse with answer
        """
        if not result:
            return ChatResponse(
                message="I don't have any previous results to analyze. Please execute a query first.",
                session_id=session_id,
                action_type="follow_up"
            )
        
        try:
            # Use LLM to answer if available
            if self.llm:
                answer = await self._answer_with_llm(question, result, last_sql)
            else:
                answer = self._answer_without_llm(question, result)
            
            # Add data source note
            row_count = result.get('row_count', len(result.get('rows', [])))
            answer += f"\n\n*This answer is based on the {row_count} rows from your last query.*"
            
            return ChatResponse(
                message=answer,
                session_id=session_id,
                action_type="follow_up"
            )
            
        except Exception as e:
            logger.error(f"Follow-up question handling failed: {e}")
            return ChatResponse(
                message=f"I couldn't process your follow-up question: {str(e)}",
                session_id=session_id,
                error=str(e),
                action_type="follow_up"
            )
    
    async def _answer_with_llm(
        self, 
        question: str, 
        result: Dict[str, Any], 
        last_sql: Optional[str]
    ) -> str:
        """
        Use LLM to answer follow-up questions.
        
        Args:
            question: User's question
            result: Query result data
            last_sql: Last executed SQL query
            
        Returns:
            Answer string
        """
        # Prepare data summary for LLM
        data_summary = self._prepare_data_summary(result)
        
        prompt = f"""You are a data analyst assistant. Answer the user's question based ONLY on the query results provided.

{f"SQL Query that generated this data: {last_sql}" if last_sql else ""}

Query Results:
{json.dumps(data_summary, indent=2)}

User's Question:
{question}

Important:
- Answer based ONLY on the data provided above
- If the question cannot be answered from this data, clearly state what information is missing
- Provide specific values, counts, or calculations from the data
- Be precise and factual
- Do not make assumptions about data that isn't shown
- Do not use any markdown formatting in your response"""
        
        response = await self.llm.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful data analyst. Answer questions based strictly on the provided data. Provide plain text answers without markdown formatting."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Low temperature for factual responses
        )
        
        return response.choices[0].message.content.strip()
    
    def _answer_without_llm(self, question: str, result: Dict[str, Any]) -> str:
        """
        Answer follow-up questions without LLM using basic analysis.
        
        Args:
            question: User's question
            result: Query result data
            
        Returns:
            Answer string
        """
        question_lower = question.lower()
        
        if 'columns' not in result or 'rows' not in result:
            return "The data format is not suitable for analysis. Please execute a standard query first."
        
        columns = result['columns']
        rows = result['rows']
        row_count = len(rows)
        col_count = len(columns)
        
        # Provide basic analysis based on keywords
        if "how many" in question_lower:
            if "row" in question_lower:
                return f"The query returned {row_count} rows."
            elif "column" in question_lower:
                return f"The result has {col_count} columns."
            else:
                return f"The query returned {row_count} rows with {col_count} columns."
        
        elif "columns" in question_lower or "fields" in question_lower:
            return f"The columns are: {', '.join(columns)}"
        
        elif "first" in question_lower and "row" in question_lower:
            if rows:
                first_row = dict(zip(columns, rows[0]))
                return f"First row:\n{json.dumps(first_row, indent=2)}"
            return "No data available."
        
        elif "last" in question_lower and "row" in question_lower:
            if rows:
                last_row = dict(zip(columns, rows[-1]))
                return f"Last row:\n{json.dumps(last_row, indent=2)}"
            return "No data available."
        
        elif "sample" in question_lower:
            if rows:
                sample_count = min(3, row_count)
                sample_data = []
                for i in range(sample_count):
                    sample_data.append(dict(zip(columns, rows[i])))
                return f"Sample of {sample_count} rows:\n{json.dumps(sample_data, indent=2)}"
            return "No data available."
        
        # Provide guidance for more specific questions
        return f"""The query returned {row_count} rows with columns: {', '.join(columns)}.

To analyze this data effectively, you can ask specific questions like:
- "What's the maximum/minimum value in [column]?"
- "How many rows have [condition]?"
- "Show me the first/last row"
- "What are the unique values in [column]?"
- "Calculate the average of [column]"

Please ask a more specific question about the data."""
    
    def _prepare_data_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a summary of the data for LLM processing.
        
        Args:
            result: Query result data
            
        Returns:
            Data summary dictionary
        """
        if 'columns' in result and 'rows' in result:
            # Limit data for LLM context (first 100 rows)
            sample_rows = result['rows'][:100]
            return {
                "columns": result['columns'],
                "total_rows": result.get('row_count', len(result['rows'])),
                "sample_data": sample_rows,
                "data_note": "Showing first 100 rows" if len(result['rows']) > 100 else "Showing all rows"
            }
        return result