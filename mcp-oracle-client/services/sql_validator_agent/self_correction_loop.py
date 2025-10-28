"""
Self-Correction Loop - LLM conversation between generator and validator
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class CorrectionAttempt:
    """Represents one correction attempt in the loop"""
    attempt_number: int
    generator_message: str
    validator_message: str
    sql_before: str
    sql_after: str
    confidence_before: float
    confidence_after: float
    issues_identified: List[str]
    improvements_made: List[str]
    timestamp: datetime

@dataclass
class CorrectionResult:
    """Final result of the correction process"""
    success: bool
    final_sql: str
    final_confidence: float
    attempts: List[CorrectionAttempt]
    total_attempts: int
    conversation_log: List[Dict[str, str]]
    recommendation: str

class SelfCorrectionLoop:
    """
    Implements a conversation loop between SQL generator and validator LLMs
    to iteratively improve query quality
    """
    
    def __init__(self, generator_llm, validator_llm, max_attempts: int = 5):
        self.generator = generator_llm  # SQL generation LLM
        self.validator = validator_llm  # SQL validation LLM
        self.max_attempts = max_attempts
        self.confidence_threshold = 0.90  # Target confidence
        
    async def correct_query(self, 
                           nl_query: str,
                           initial_sql: str,
                           context: Dict[str, Any],
                           initial_validation: Any,
                           intent: str = "new_query",
                           previous_messages: str = None,
                           original_sql: str = None) -> CorrectionResult:
        """
        Main correction loop where generator and validator LLMs communicate
        """
        logger.info(f"Starting self-correction loop for query with initial confidence: {initial_validation.overall_confidence:.2%}")
        
        attempts = []
        conversation_log = []
        current_sql = initial_sql
        current_validation = initial_validation
        
        for attempt_num in range(1, self.max_attempts + 1):
            logger.info(f"Correction attempt {attempt_num}/{self.max_attempts}")
            
            # Check if we've reached acceptable confidence
            if current_validation.overall_confidence >= self.confidence_threshold:
                logger.info(f"Target confidence reached: {current_validation.overall_confidence:.2%}")
                break
            
            # Step 1: Validator provides detailed feedback
            validator_feedback = await self._get_validator_feedback(
                nl_query, current_sql, current_validation, context
            )
            
            conversation_log.append({
                "role": "validator",
                "message": validator_feedback,
                "timestamp": datetime.now().isoformat()
            })
            
            # Step 2: Generator responds with improvements
            generator_response, improved_sql = await self._get_generator_improvements(
                nl_query, current_sql, validator_feedback, context,
                intent, previous_messages, original_sql
            )
            
            conversation_log.append({
                "role": "generator",
                "message": generator_response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Step 3: Validate the improved SQL
            new_validation = await self.validator.validate_sql(
                nl_query, improved_sql, context
            )
            
            # Record the attempt
            attempt = CorrectionAttempt(
                attempt_number=attempt_num,
                generator_message=generator_response,
                validator_message=validator_feedback,
                sql_before=current_sql,
                sql_after=improved_sql,
                confidence_before=current_validation.overall_confidence,
                confidence_after=new_validation.overall_confidence,
                issues_identified=self._extract_issues(current_validation),
                improvements_made=self._identify_improvements(current_sql, improved_sql),
                timestamp=datetime.now()
            )
            attempts.append(attempt)
            
            # Check for improvement
            if new_validation.overall_confidence > current_validation.overall_confidence:
                logger.info(f"Improvement achieved: {current_validation.overall_confidence:.2%} -> {new_validation.overall_confidence:.2%}")
                current_sql = improved_sql
                current_validation = new_validation
            else:
                logger.warning(f"No improvement in attempt {attempt_num}, trying different approach")
                # Ask for alternative approach
                conversation_log.append({
                    "role": "system",
                    "message": "Previous attempt did not improve. Try a different approach.",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Determine final status
        success = current_validation.overall_confidence >= self.confidence_threshold
        
        return CorrectionResult(
            success=success,
            final_sql=current_sql,
            final_confidence=current_validation.overall_confidence,
            attempts=attempts,
            total_attempts=len(attempts),
            conversation_log=conversation_log,
            recommendation=self._get_final_recommendation(
                success, current_validation.overall_confidence, len(attempts)
            )
        )
    
    async def _get_validator_feedback(self, nl_query: str, sql_query: str,
                                     validation_result: Any,
                                     context: Dict[str, Any]) -> str:
        """
        Get detailed feedback from the validator LLM
        """
        # Collect all issues from validation stages
        all_issues = []
        for stage, result in validation_result.stage_results.items():
            if not result.passed:
                all_issues.append(f"{stage.value}: {', '.join(result.issues)}")
        
        prompt = f"""
        As a SQL validation expert, provide specific feedback to improve this query.
        
        Natural Language Request: {nl_query}
        
        Current SQL: {sql_query}
        
        Validation Results:
        - Overall Confidence: {validation_result.overall_confidence:.2%}
        - Issues Found:
        {json.dumps(all_issues, indent=2)}
        
        Database Schema:
        Tables: {', '.join(context.get('tables', {}).keys())}
        
        Provide specific, actionable feedback for the SQL generator to improve the query.
        Focus on:
        1. What's wrong with the current query
        2. What specific changes would fix the issues
        3. Why these changes would improve confidence
        
        Be concise and specific.
        """
        
        try:
            response = await self.validator.client.chat.completions.create(
                model=self.validator.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a SQL validation expert providing feedback to improve queries."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to get validator feedback: {e}")
            return f"Issues found: {', '.join(all_issues)}. Please improve the query."
    
    async def _get_generator_improvements(self, nl_query: str, current_sql: str,
                                         validator_feedback: str,
                                         context: Dict[str, Any],
                                         intent: str = "new_query",
                                         previous_messages: str = None,
                                         original_sql: str = None) -> Tuple[str, str]:
        """
        Get improved SQL from the generator based on validator feedback
        Handles both new and modify intents
        """
        prompt = f"""
        As a SQL generation expert, improve the query based on validator feedback.
        
        Intent: {intent}
        Natural Language Request: {nl_query}
        """
        
        if intent == "modify_query" and original_sql:
            prompt += f"""
        Original SQL (before modification): {original_sql}
        Previous Messages: {previous_messages}
        """
        
        prompt += f"""
        Current SQL: {current_sql}
        
        Validator Feedback:
        {validator_feedback}
        
        Database Schema and Context:
        {json.dumps(context, indent=2)}
        
        Please:
        1. Acknowledge the specific issues raised
        2. Explain how you'll address each issue
        3. Provide the improved SQL query
        
        Format your response as:
        ACKNOWLEDGMENT: [Your understanding of the issues]
        IMPROVEMENTS: [Specific changes you're making]
        SQL: [The improved query]
        """
        
        try:
            response = await self.generator.client.chat.completions.create(
                model=self.generator.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a SQL generation expert. Improve queries based on feedback."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse the response
            acknowledgment = ""
            improvements = ""
            sql = current_sql  # Default to current if parsing fails
            
            lines = response_text.split('\n')
            current_section = None
            sql_lines = []
            
            for line in lines:
                if line.startswith("ACKNOWLEDGMENT:"):
                    current_section = "ack"
                    acknowledgment = line.replace("ACKNOWLEDGMENT:", "").strip()
                elif line.startswith("IMPROVEMENTS:"):
                    current_section = "imp"
                    improvements = line.replace("IMPROVEMENTS:", "").strip()
                elif line.startswith("SQL:"):
                    current_section = "sql"
                    sql_line = line.replace("SQL:", "").strip()
                    if sql_line:
                        sql_lines.append(sql_line)
                elif current_section == "sql":
                    sql_lines.append(line)
            
            if sql_lines:
                sql = '\n'.join(sql_lines).strip()
                sql = sql.replace("```sql", "").replace("```", "").strip()
            
            generator_message = f"ACKNOWLEDGED: {acknowledgment}\nIMPROVING: {improvements}"
            
            return generator_message, sql
            
        except Exception as e:
            logger.error(f"Failed to get generator improvements: {e}")
            # Try basic correction
            return "Attempting to fix identified issues", current_sql
    
    async def correct_simple_query(self, nl_query: str, sql_query: str,
                                  context: Dict[str, Any],
                                  validation_result: Any) -> CorrectionResult:
        """
        Simpler correction for non-complex queries
        """
        logger.info("Running simple query correction...")
        
        # For simple queries, try up to 3 attempts
        max_simple_attempts = min(3, self.max_attempts)
        
        return await self.correct_query(
            nl_query, sql_query, context, validation_result
        )
    
    def _extract_issues(self, validation_result: Any) -> List[str]:
        """Extract all issues from validation result"""
        issues = []
        for stage, result in validation_result.stage_results.items():
            issues.extend(result.issues)
        return issues
    
    def _identify_improvements(self, sql_before: str, sql_after: str) -> List[str]:
        """Identify what changed between two SQL queries"""
        improvements = []
        
        before_upper = sql_before.upper()
        after_upper = sql_after.upper()
        
        # Check for structural changes
        if 'JOIN' in after_upper and 'JOIN' not in before_upper:
            improvements.append("Added JOIN clause")
        if 'WHERE' in after_upper and 'WHERE' not in before_upper:
            improvements.append("Added WHERE clause")
        if 'GROUP BY' in after_upper and 'GROUP BY' not in before_upper:
            improvements.append("Added GROUP BY clause")
        if 'ORDER BY' in after_upper and 'ORDER BY' not in before_upper:
            improvements.append("Added ORDER BY clause")
        
        # Check for function additions
        functions = ['SUM', 'AVG', 'COUNT', 'MAX', 'MIN']
        for func in functions:
            if func in after_upper and func not in before_upper:
                improvements.append(f"Added {func} aggregation")
        
        if not improvements:
            improvements.append("Modified query structure")
        
        return improvements
    
    def _get_final_recommendation(self, success: bool, confidence: float, 
                                 attempts: int) -> str:
        """Generate final recommendation based on correction results"""
        if success:
            if attempts == 0:
                return "Query passed validation without corrections"
            elif attempts == 1:
                return "Query successfully corrected in one attempt"
            else:
                return f"Query successfully corrected after {attempts} attempts"
        else:
            if confidence >= 0.85:
                return "Query is mostly correct but may benefit from manual review"
            elif confidence >= 0.75:
                return "Query needs manual review - automated correction partially successful"
            else:
                return "Query requires significant manual intervention"

class LLMConversationManager:
    """
    Manages the conversation between generator and validator LLMs
    """
    
    def __init__(self, generator_llm, validator_llm):
        self.generator = generator_llm
        self.validator = validator_llm
        self.correction_loop = SelfCorrectionLoop(generator_llm, validator_llm)
        
    async def improve_query_through_conversation(self,
                                                nl_query: str,
                                                initial_sql: str,
                                                context: Dict[str, Any],
                                                is_complex: bool = False,
                                                intent: str = "new_query",
                                                previous_messages: str = None,
                                                original_sql: str = None) -> Dict[str, Any]:
        """
        Main entry point for query improvement through LLM conversation
        Handles both new_query and modify_query intents
        """
        logger.info(f"Starting LLM conversation for query improvement (complex={is_complex}, intent={intent})")
        
        # Initial validation
        initial_validation = await self.validator.validate_sql(
            nl_query, initial_sql, context
        )
        
        initial_confidence = initial_validation.overall_confidence
        logger.info(f"Initial validation confidence: {initial_confidence:.2%}")
        
        # Check if correction is needed
        if initial_confidence >= 0.95:
            return {
                "success": True,
                "needs_correction": False,
                "final_sql": initial_sql,
                "confidence": initial_confidence,
                "message": "Query validated successfully without corrections"
            }
        
        # Run correction loop
        if is_complex:
            # For complex queries, allow more attempts
            self.correction_loop.max_attempts = 5
        else:
            # For simple queries, limit attempts
            self.correction_loop.max_attempts = 3
        
        correction_result = await self.correction_loop.correct_query(
            nl_query, initial_sql, context, initial_validation,
            intent, previous_messages, original_sql
        )
        
        # Prepare response
        improvement = correction_result.final_confidence - initial_confidence
        
        return {
            "success": correction_result.success,
            "needs_correction": True,
            "final_sql": correction_result.final_sql,
            "confidence": correction_result.final_confidence,
            "initial_confidence": initial_confidence,
            "improvement": improvement,
            "attempts": correction_result.total_attempts,
            "conversation_log": correction_result.conversation_log,
            "recommendation": correction_result.recommendation,
            "message": self._generate_user_message(correction_result, initial_confidence)
        }
    
    def _generate_user_message(self, correction_result: CorrectionResult,
                              initial_confidence: float) -> str:
        """Generate a user-friendly message about the correction process"""
        
        if correction_result.success:
            if correction_result.total_attempts == 0:
                return "✅ Query validated successfully!"
            else:
                improvement = correction_result.final_confidence - initial_confidence
                return (f"✅ Query improved through {correction_result.total_attempts} correction(s). "
                       f"Confidence increased from {initial_confidence:.0%} to "
                       f"{correction_result.final_confidence:.0%} (+{improvement:.0%})")
        else:
            if correction_result.final_confidence >= 0.85:
                return (f"⚠️ Query partially improved to {correction_result.final_confidence:.0%} confidence. "
                       f"Manual review recommended for best results.")
            else:
                return (f"⚠️ Query needs manual review. Automated correction achieved "
                       f"{correction_result.final_confidence:.0%} confidence.")