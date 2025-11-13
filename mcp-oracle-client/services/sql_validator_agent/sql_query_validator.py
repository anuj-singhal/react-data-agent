"""
LLM Interface for Agentic SQL Generation
Unified interface for different LLM providers
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationStage(Enum):
    SYNTAX = "syntax_validation"
    SCHEMA = "schema_validation"
    SEMANTIC = "semantic_validation"
    # EXECUTION = "execution_validation"
    # CORRECTNESS = "correctness_validation"  # Replaced performance with correctness

@dataclass
class ValidationResult:
    stage: ValidationStage
    passed: bool
    confidence: float
    issues: List[str]
    suggestions: List[str]
    details: Dict[str, Any]

@dataclass
class OverallValidation:
    sql_query: str
    overall_confidence: float
    stage_results: Dict[ValidationStage, ValidationResult]
    final_verdict: str
    corrected_sql: Optional[str] = None
    explanation: str = ""

class SQLValidator:
    """Unified interface for LLM providers used in agentic SQL generation"""
    
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if self.provider == "openai":
            self.model = model or "gpt-4o-mini"
            if self.api_key:
                self.client = AsyncOpenAI(api_key=self.api_key)
            else:
                logger.warning("No OpenAI API key provided")
                self.client = None
        else:
            self.model = None
            self.client = None

        self.stage_weights = {
            ValidationStage.SYNTAX: 0.40,
            ValidationStage.SCHEMA: 0.40,
            ValidationStage.SEMANTIC: 0.20,
            # ValidationStage.EXECUTION: 0.15,
            # ValidationStage.CORRECTNESS: 0.15  # Correctness instead of performance
        }
    
    async def _validate(self, prompt):
        """Generate SQL using OpenAI"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert SQL validator specialized in different SQL Validation. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            result = result.replace("```json", "").replace("```", "").strip()
            response = json.loads(result)
            
            return response
            
        except Exception as e:
            print(f"Error with OpenAI: {e}")
            return None

    async def validate_syntax(self, sql_query:str) -> ValidationResult:
        """Generate SQL using OpenAI"""
        prompt = self._create_syntax_validation_prompt(sql_query)
        response = await self._validate(prompt)

        return ValidationResult(
                stage=ValidationStage.SYNTAX,
                passed=response.get("valid", False),
                confidence=response.get("confidence", 0.0),
                issues=response.get("issues", []),
                suggestions=response.get("suggestions", []),
                details=response
            )
    
    async def validate_schema(self, sql_query:str, context) -> ValidationResult:
        """Generate SQL using OpenAI"""
        prompt = self._create_schema_validation_prompt(sql_query, context)
        response = await self._validate(prompt)
        
        return ValidationResult(
                stage=ValidationStage.SCHEMA,
                passed=response.get("valid", False),
                confidence=response.get("confidence", 0.0),
                issues=response.get("issues", []),
                suggestions=response.get("suggestions", []),
                details=response
            )
    
    async def validate_semantics(self, nl_query, sql_query:str) -> ValidationResult:
        """Generate SQL using OpenAI"""
        prompt = self._create_semantic_validation_prompt(nl_query, sql_query)
        response = await self._validate(prompt)
        passed = response.get("semantic_match", False) and response.get("alignment_score", 0) > 0.7

        return ValidationResult(
                stage=ValidationStage.SEMANTIC,
                passed=passed,
                confidence=response.get("confidence", 0.0),
                issues=response.get("missing_elements", []),
                suggestions=response.get("suggestions", []),
                details=response
            )
    
    async def validate_sql(self, nl_query: str, sql_query: str, context) -> OverallValidation:
        """Main validation pipeline"""
        logger.info(f"Starting validation pipeline for query: {sql_query[:50]}...")
        
        stage_results = {}
        
        # Stage 1: Syntax Validation
        logger.info("Stage 1: Validating syntax...")
        stage_results[ValidationStage.SYNTAX] = await self.validate_syntax(sql_query)
        print(stage_results[ValidationStage.SYNTAX].confidence)
        # Stage 2: Schema Validation
        logger.info("Stage 2: Validating schema...")
        stage_results[ValidationStage.SCHEMA] = await self.validate_schema(sql_query, context)
        print(stage_results[ValidationStage.SCHEMA])
        # Stage 3: Semantic Validation
        logger.info("Stage 3: Validating semantics...")
        stage_results[ValidationStage.SEMANTIC] = await self.validate_semantics(nl_query, sql_query)
        print(stage_results[ValidationStage.SEMANTIC])
        # # Stage 4: Execution Validation
        # logger.info("Stage 4: Validating execution...")
        # stage_results[ValidationStage.EXECUTION] = self.validate_execution(sql_query)
        
        # # Stage 5: Correctness Validation
        # logger.info("Stage 5: Validating correctness...")
        # stage_results[ValidationStage.CORRECTNESS] = self.validate_correctness(nl_query, sql_query)
        
        # Calculate overall confidence
        overall_confidence, verdict = self._calculate_overall_confidence(stage_results)
        print("Overall confidence: ", overall_confidence)
        print("Verdict : ", verdict)
        # Generate explanation
        explanation = self._generate_explanation(stage_results, overall_confidence)
        print("Explanation : ", explanation)
        # Determine if SQL needs correction
        corrected_sql = None
        if overall_confidence < 0.7:
            corrected_sql = self._suggest_correction(sql_query, stage_results)
        
        logger.info(f"Validation complete. Overall confidence: {overall_confidence:.2%}")
        
        return OverallValidation(
            sql_query=sql_query,
            overall_confidence=overall_confidence,
            stage_results=stage_results,
            final_verdict=verdict,
            corrected_sql=corrected_sql,
            explanation=explanation
        )

    def _simulate_execution(self, sql_query: str) -> Dict[str, Any]:
        """Simulate SQL execution for demo purposes"""
        try:
            # For demo purposes, return simulated results
            # In production, execute against actual database
            if sql_query.strip().upper().startswith('SELECT'):
                return {
                    "success": True,
                    "rows_returned": 10,
                    "execution_time": 0.045,
                    "message": "Query executed successfully"
                }
            else:
                return {
                    "success": True,
                    "rows_affected": 0,
                    "execution_time": 0.023,
                    "message": "Non-SELECT query validated"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": 0,
                "message": f"Execution failed: {str(e)}"
            }
    
    def _calculate_overall_confidence(self, stage_results: Dict[ValidationStage, ValidationResult]) -> Tuple[float, str]:
        """Calculate overall confidence score using weighted average"""
        total_weight = 0
        weighted_sum = 0
        
        for stage, result in stage_results.items():
            weight = self.stage_weights.get(stage, 0.1)
            # Apply penalty for failed stages
            score = result.confidence if result.passed else result.confidence * 0.5
            weighted_sum += score * weight
            total_weight += weight
        
        overall_confidence = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Determine verdict
        if overall_confidence >= 0.85:
            verdict = "HIGHLY CONFIDENT - Query is well-formed and business logic is correct"
        elif overall_confidence >= 0.70:
            verdict = "CONFIDENT - Query is valid with minor suggestions"
        elif overall_confidence >= 0.50:
            verdict = "MODERATE - Query works but needs improvements"
        else:
            verdict = "LOW CONFIDENCE - Query has significant issues"
        
        return overall_confidence, verdict
    
    def _generate_explanation(self, stage_results: Dict[ValidationStage, ValidationResult], 
                            overall_confidence: float) -> str:
        """Generate human-readable explanation of validation results"""
        explanation_parts = []
        
        explanation_parts.append(f"Overall Confidence: {overall_confidence:.2%}\n")
        explanation_parts.append("=" * 50 + "\n")
        
        for stage, result in stage_results.items():
            status = "✓ PASSED" if result.passed else "✗ FAILED"
            explanation_parts.append(f"\n{stage.value.upper()} {status} (Confidence: {result.confidence:.2%})")
            
            if result.issues:
                explanation_parts.append("\n  Issues:")
                for issue in result.issues:
                    explanation_parts.append(f"    - {issue}")
            
            if result.suggestions:
                explanation_parts.append("\n  Suggestions:")
                for suggestion in result.suggestions[:2]:
                    explanation_parts.append(f"    - {suggestion}")
        
        return "\n".join(explanation_parts)
    
    def _suggest_correction(self, sql_query: str, 
                           stage_results: Dict[ValidationStage, ValidationResult]) -> str:
        """Suggest corrected SQL based on validation results"""
        corrected = sql_query
        
        # Apply corrections from syntax validation if available
        if ValidationStage.SYNTAX in stage_results:
            syntax_details = stage_results[ValidationStage.SYNTAX].details
            if "corrected_sql" in syntax_details:
                corrected = syntax_details["corrected_sql"]
        
        return corrected

    def _create_syntax_validation_prompt(self, sql_query: str) -> str:
        """Create prompt for syntax validation with banking examples"""
        prompt = f"""
            You are an expert SQL syntax validator for banking systems. Analyze the given SQL query for syntax correctness.
            If all the syntax is correct and the query is valid then make confidence: 1.0
            ## SYNTAX_VALIDATION

            ### Few-Shot Examples Using Banking Tables:

            Example 1:
            Input SQL: SELECT * FORM BANKS WHERE BANK_NAME = 'ADCB'
            Valid: False
            Confidence: 0.95
            Issues: ["SQL keyword 'FORM' is incorrect, should be 'FROM'"]
            Suggestions: ["Replace 'FORM' with 'FROM'"]
            Corrected SQL: SELECT * FROM BANKS WHERE BANK_NAME = 'ADCB'

            Example 2:
            Input SQL: SELECT b.BANK_NAME, fp.YTD_PROFIT FROM BANKS b JOIN FINANCIAL_PERFORMANCE fp ON b.BANK_ID = fp.BANK_ID WHERE fp.YEAR = 2024 AND fp.QUARTER = 4
            Valid: True
            Confidence: 0.98
            Issues: []
            Suggestions: []
            Analysis: Proper JOIN syntax with correct table aliases and WHERE conditions

            Example 3:
            Input SQL: SELECT BANK_NAME, SUM(YTD_PROFIT) AS total_profit FROM BANKS b JOIN FINANCIAL_PERFORMANCE fp ON b.BANK_ID = fp.BANK_ID GROUP BY b.BANK_NAME HAVING SUM(YTD_PROFIT) > 1000
            Valid: True
            Confidence: 0.97
            Issues: []
            Suggestions: ["Consider using table alias consistently for BANK_NAME in SELECT"]
            Analysis: Valid GROUP BY with HAVING clause for aggregation

            Example 4:
            Input SQL: UPDATE FINANCIAL_PERFORMANCE SET NPL_RATIO = 3.5 WHERE BANK_ID = 1 AND YEAR = 2024 AND QUARTER = 4
            Valid: True
            Confidence: 0.96
            Issues: []
            Suggestions: []
            Analysis: Valid UPDATE statement with compound WHERE condition

            Example 5:
            Input SQL: SELECT BANK_NAME, AVG(NIM) OVER (PARTITION BY YEAR ORDER BY QUARTER) as avg_nim FROM BANKS b JOIN FINANCIAL_PERFORMANCE fp USING(BANK_ID)
            Valid: True
            Confidence: 0.94
            Issues: []
            Suggestions: ["Window function syntax is correct"]
            Analysis: Valid window function with PARTITION BY and ORDER BY

            ### Now validate this SQL query:
            Input SQL: {sql_query}

            Provide response in JSON format:
            {{
                "valid": boolean,
                "confidence": float (0-1),
                "issues": [list of syntax issues],
                "suggestions": [list of improvement suggestions],
                "corrected_sql": "corrected query if issues found",
                "analysis": "detailed analysis"
            }}
            """
        return prompt
    
    def _create_schema_validation_prompt(self, sql_query: str, context) -> str:
        """Create prompt for schema validation with banking tables"""

        prompt = f""":"""
        
        # Add table information
        for table_name, table_info in context['tables'].items():
            prompt += f"\n\nTABLE: {table_name}"
            prompt += f"\nDescription: {table_info['description']}"
            prompt += "\nColumns:"
            
            for col in table_info['columns']:
                pk = " [PK]" if col.get('is_pk') else ""
                prompt += f"\n  - {col['name']} ({col['type']}){pk}: {col['description']}"
        
        # Add relationships
        if context['relationships']:
            prompt += "\n\nRELATIONSHIPS:"
            for rel in context['relationships']:
                prompt += f"\n- {rel['from']} -> {rel['to']} ({rel['type']})"

        prompt_final = f"""
You are a database schema expert for banking systems. Validate if the SQL query correctly only references tables and columns from the given schema.
If all the schema is correct and the query is valid then make confidence: 1.0
## SCHEMA_VALIDATION
{prompt}

### Few-Shot Examples:

Example 1:
SQL: SELECT b.BANK_NAME, fp.TOTAL_INCOME FROM BANKS b JOIN FINANCIAL_PERFORMANCE fp ON b.BANK_ID = fp.BANK_ID
Valid: False
Confidence: 0.95
Issues: ["Column 'TOTAL_INCOME' does not exist in FINANCIAL_PERFORMANCE table, perhaps you meant 'YTD_INCOME'"]
Suggestions: ["Use 'YTD_INCOME' instead of 'TOTAL_INCOME'"]
Schema Mapping: {{"tables_used": ["BANKS", "FINANCIAL_PERFORMANCE"], "invalid_columns": ["TOTAL_INCOME"]}}

Example 2:
SQL: SELECT b.BANK_NAME, fp.NPL_RATIO, md.CET1 FROM BANKS b JOIN FINANCIAL_PERFORMANCE fp ON b.BANK_ID = fp.BANK_ID JOIN MARKET_DATA md ON b.BANK_ID = md.BANK_ID AND fp.YEAR = md.YEAR AND fp.QUARTER = md.QUARTER
Valid: True
Confidence: 0.98
Issues: []
Suggestions: ["Three-way join correctly implemented with matching year and quarter"]
Schema Mapping: {{"tables_used": ["BANKS", "FINANCIAL_PERFORMANCE", "MARKET_DATA"], "columns_validated": true}}

Example 3:
SQL: SELECT BANK_CODE, AVG(DEPOSITS) as avg_deposits FROM BANKS b JOIN FINANCIAL_PERFORMANCE fp ON b.ID = fp.BANK_ID GROUP BY BANK_CODE
Valid: False
Confidence: 0.93
Issues: ["Column 'ID' does not exist in BANKS table, should be 'BANK_ID'"]
Suggestions: ["Use 'BANK_ID' instead of 'ID' for the join condition"]
Schema Mapping: {{"tables_used": ["BANKS", "FINANCIAL_PERFORMANCE"], "invalid_columns": ["ID"]}}

Example 4:
SQL: SELECT YEAR, QUARTER, SUM(LOANS_ADVANCES) as total_loans, AVG(NPL_RATIO) as avg_npl FROM FINANCIAL_PERFORMANCE GROUP BY YEAR, QUARTER
Valid: True
Confidence: 0.97
Issues: []
Suggestions: ["All columns exist and aggregation is properly grouped"]
Schema Mapping: {{"tables_used": ["FINANCIAL_PERFORMANCE"], "columns_validated": true}}

### Now validate this SQL query:
SQL: {sql_query}

Provide response in JSON format:
{{
    "valid": boolean,
    "confidence": float (0-1),
    "issues": [list of schema-related issues],
    "suggestions": [list of suggestions],
    "schema_mapping": {{
        "tables_used": [list of tables],
        "columns_validated": boolean,
        "invalid_columns": [list of invalid columns if any]
    }}
}}
"""
        return prompt_final
    
    def _create_semantic_validation_prompt(self, nl_query: str, sql_query: str) -> str:
        """Create prompt for semantic validation with banking context"""
        prompt = f"""
You are an expert in validating if SQL queries correctly capture the intent of banking-related business questions.
If all the semantic is correct then make semantic_match true and alignment_score 1.0

## SEMANTIC_VALIDATION

### Few-Shot Banking Examples:

Example 1:
Question: "Show me the top 5 banks by profit in Q4 2024"
SQL: SELECT b.BANK_NAME, fp.QUARTERLY_PROFIT FROM BANKS b JOIN FINANCIAL_PERFORMANCE fp ON b.BANK_ID = fp.BANK_ID WHERE fp.YEAR = 2024 AND fp.QUARTER = 4 ORDER BY fp.QUARTERLY_PROFIT DESC LIMIT 5
Alignment Score: 0.95
Confidence: 0.92
Semantic Match: True
Intent Captured: True
Missing Elements: []
Analysis: Query correctly filters for Q4 2024, orders by profit descending, and limits to 5 banks

Example 2:
Question: "Find banks with NPL ratio above 5%"
SQL: SELECT b.BANK_NAME, fp.NPL_RATIO FROM BANKS b JOIN FINANCIAL_PERFORMANCE fp ON b.BANK_ID = fp.BANK_ID WHERE fp.NPL_RATIO < 5
Alignment Score: 0.20
Confidence: 0.95
Semantic Match: False
Intent Captured: False
Missing Elements: ["Wrong comparison operator - should be > 5, not < 5"]
Analysis: Query logic is inverted - finds banks with NPL below 5% instead of above

Example 3:
Question: "What's the average Net Interest Margin for each bank in 2023?"
SQL: SELECT b.BANK_NAME, AVG(fp.NIM) as avg_nim FROM BANKS b JOIN FINANCIAL_PERFORMANCE fp ON b.BANK_ID = fp.BANK_ID WHERE fp.YEAR = 2023 GROUP BY b.BANK_NAME
Alignment Score: 0.98
Confidence: 0.94
Semantic Match: True
Intent Captured: True
Missing Elements: []
Analysis: Correctly calculates average NIM per bank for 2023 with proper grouping

Example 4:
Question: "Compare deposit growth quarter-over-quarter for all banks"
SQL: SELECT b.BANK_NAME, fp.YEAR, fp.QUARTER, fp.DEPOSITS, LAG(fp.DEPOSITS) OVER (PARTITION BY b.BANK_ID ORDER BY fp.YEAR, fp.QUARTER) as prev_deposits, (fp.DEPOSITS - LAG(fp.DEPOSITS) OVER (PARTITION BY b.BANK_ID ORDER BY fp.YEAR, fp.QUARTER)) / LAG(fp.DEPOSITS) OVER (PARTITION BY b.BANK_ID ORDER BY fp.YEAR, fp.QUARTER) * 100 as qoq_growth FROM BANKS b JOIN FINANCIAL_PERFORMANCE fp ON b.BANK_ID = fp.BANK_ID
Alignment Score: 0.96
Confidence: 0.91
Semantic Match: True
Intent Captured: True
Missing Elements: []
Analysis: Uses window function LAG correctly to calculate quarter-over-quarter growth

Example 5:
Question: "Show banks with improving cost-to-income ratio"
SQL: SELECT BANK_NAME FROM BANKS
Alignment Score: 0.15
Confidence: 0.93
Semantic Match: False
Intent Captured: False
Missing Elements: ["No analysis of cost-to-income ratio trend", "Missing join with FINANCIAL_PERFORMANCE", "No comparison across time periods"]
Analysis: Query only returns bank names without any cost-to-income analysis

### Now validate this query pair:
Question: {nl_query}
SQL: {sql_query}

Provide response in JSON format:
{{
    "alignment_score": float (0-1),
    "confidence": float (0-1),
    "semantic_match": boolean,
    "intent_captured": boolean,
    "missing_elements": [list of missing semantic elements],
    "suggestions": [list of suggestions to better capture intent],
    "analysis": "detailed semantic analysis"
}}
"""
        return prompt
    
# Singleton instance
sql_validator = SQLValidator()        