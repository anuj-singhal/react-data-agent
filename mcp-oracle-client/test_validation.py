"""
Multi-Stage SQL Validation Pipeline with OpenAI GPT-4o-mini
Specialized for Banking Data with Correctness Validation
Author: SQL Validation Framework
Version: 2.0.0
"""

import json
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging
import os
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ValidationStage(Enum):
    SYNTAX = "syntax_validation"
    SCHEMA = "schema_validation"
    SEMANTIC = "semantic_validation"
    EXECUTION = "execution_validation"
    CORRECTNESS = "correctness_validation"  # Replaced performance with correctness

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

class OpenAILLMValidator:
    """OpenAI GPT-4o-mini based validator for SQL queries"""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
    
    def validate(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API for validation"""
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert SQL validator specialized in banking data analysis. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            # Return default response on error
            return {
                "valid": False,
                "confidence": 0.0,
                "issues": [f"Validation error: {str(e)}"],
                "suggestions": ["Check API connection and retry"],
                "details": {}
            }

class BankingSQLValidator:
    """
    Multi-Stage SQL Validation Pipeline for Banking Data
    """
    
    def __init__(self, llm_validator: Optional[OpenAILLMValidator] = None):
        self.schema = self._load_banking_schema()
        self.llm = llm_validator or OpenAILLMValidator()
        self.stage_weights = {
            ValidationStage.SYNTAX: 0.20,
            ValidationStage.SCHEMA: 0.25,
            ValidationStage.SEMANTIC: 0.25,
            ValidationStage.EXECUTION: 0.15,
            ValidationStage.CORRECTNESS: 0.15  # Correctness instead of performance
        }
    
    def _load_banking_schema(self) -> Dict[str, Any]:
        """Load the banking data dictionary"""
        return {
            "BANKS": {
                "description": "Master table containing bank information including identification details and codes for financial institutions",
                "columns": {
                    "BANK_ID": {"type": "NUMBER(38,0)", "primary_key": True, "nullable": False, "description": "Unique identifier for each bank"},
                    "BANK_NAME": {"type": "VARCHAR2(26)", "nullable": True, "description": "Official registered name of the bank"},
                    "BANK_CODE": {"type": "VARCHAR2(26)", "nullable": True, "description": "Short code or abbreviation for the bank"}
                }
            },
            "FINANCIAL_PERFORMANCE": {
                "description": "Quarterly and yearly financial performance metrics for banks",
                "columns": {
                    "PERFORMANCE_ID": {"type": "NUMBER(38,0)", "primary_key": True, "nullable": False, "description": "Unique identifier for each performance record"},
                    "YEAR": {"type": "NUMBER(38,0)", "nullable": True, "description": "Fiscal year"},
                    "QUARTER": {"type": "NUMBER(38,0)", "nullable": True, "description": "Quarter (1-4)"},
                    "BANK_ID": {"type": "NUMBER(38,0)", "nullable": True, "description": "Reference to BANKS table"},
                    "YTD_INCOME": {"type": "NUMBER(38,1)", "nullable": True, "description": "Year-to-date income in millions"},
                    "YTD_PROFIT": {"type": "NUMBER(38,1)", "nullable": True, "description": "Year-to-date profit in millions"},
                    "QUARTERLY_PROFIT": {"type": "NUMBER(38,1)", "nullable": True, "description": "Quarter profit in millions"},
                    "LOANS_ADVANCES": {"type": "NUMBER(38,1)", "nullable": True, "description": "Total loans and advances in millions"},
                    "NIM": {"type": "NUMBER(38,1)", "nullable": True, "description": "Net Interest Margin percentage"},
                    "DEPOSITS": {"type": "NUMBER(38,1)", "nullable": True, "description": "Total deposits in millions"},
                    "CASA": {"type": "NUMBER(38,1)", "nullable": True, "description": "Current and Savings Account deposits in millions"},
                    "COST_INCOME": {"type": "NUMBER(38,1)", "nullable": True, "description": "Cost-to-Income ratio percentage"},
                    "NPL_RATIO": {"type": "NUMBER(38,1)", "nullable": True, "description": "Non-Performing Loans ratio percentage"},
                    "COR": {"type": "NUMBER(38,2)", "nullable": True, "description": "Cost of Risk percentage"},
                    "STAGE3_COVER": {"type": "NUMBER(38,1)", "nullable": True, "description": "Stage 3 coverage ratio percentage"},
                    "ROTE": {"type": "NUMBER(38,1)", "nullable": True, "description": "Return on Tangible Equity percentage"}
                }
            },
            "MARKET_DATA": {
                "description": "Market and capital-related metrics for banks",
                "columns": {
                    "MARKET_ID": {"type": "NUMBER(38,0)", "primary_key": True, "nullable": False, "description": "Unique identifier for market data"},
                    "YEAR": {"type": "NUMBER(38,0)", "nullable": True, "description": "Year of market data"},
                    "QUARTER": {"type": "NUMBER(38,0)", "nullable": True, "description": "Quarter (1-4)"},
                    "BANK_ID": {"type": "NUMBER(38,0)", "nullable": True, "description": "Reference to BANKS table"},
                    "CET1": {"type": "NUMBER(38,1)", "nullable": True, "description": "Common Equity Tier 1 capital ratio percentage"},
                    "CET_CAPITAL": {"type": "NUMBER(38,1)", "nullable": True, "description": "CET1 capital amount in millions"},
                    "RWA": {"type": "NUMBER(38,1)", "nullable": True, "description": "Risk-Weighted Assets in millions"},
                    "SHARE_PRICE": {"type": "NUMBER(38,2)", "nullable": True, "description": "Stock price in local currency"},
                    "MARKET_CAP_AED_BN": {"type": "NUMBER(38,2)", "nullable": True, "description": "Market cap in billions AED"},
                    "MARKET_CAP_USD_BN": {"type": "NUMBER(38,2)", "nullable": True, "description": "Market cap in billions USD"},
                    "PE_RATIO": {"type": "NUMBER(38,2)", "nullable": True, "description": "Price-to-Earnings ratio"},
                    "PB_RATIO": {"type": "NUMBER(38,2)", "nullable": True, "description": "Price-to-Book ratio"}
                }
            }
        }
    
    def create_syntax_validation_prompt(self, sql_query: str) -> str:
        """Create prompt for syntax validation with banking examples"""
        prompt = f"""
You are an expert SQL syntax validator for banking systems. Analyze the given SQL query for syntax correctness.

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
    
    def create_schema_validation_prompt(self, sql_query: str) -> str:
        """Create prompt for schema validation with banking tables"""
        schema_str = json.dumps({
            table: {
                "columns": list(details["columns"].keys()),
                "description": details["description"]
            }
            for table, details in self.schema.items()
        }, indent=2)
        
        prompt = f"""
You are a database schema expert for banking systems. Validate if the SQL query correctly references tables and columns from the given schema.

## SCHEMA_VALIDATION

### Banking Database Schema:
{schema_str}

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
        return prompt
    
    def create_semantic_validation_prompt(self, nl_query: str, sql_query: str) -> str:
        """Create prompt for semantic validation with banking context"""
        prompt = f"""
You are an expert in validating if SQL queries correctly capture the intent of banking-related business questions.

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
    
    def create_execution_validation_prompt(self, sql_query: str, execution_result: Dict) -> str:
        """Create prompt for execution validation with banking context"""
        prompt = f"""
You are a SQL execution expert for banking systems. Analyze the execution results and validate query behavior.

## EXECUTION_VALIDATION

### Few-Shot Banking Examples:

Example 1:
SQL: SELECT COUNT(*) FROM FINANCIAL_PERFORMANCE WHERE NPL_RATIO < 0
Execution: Success
Result: 0 rows
Valid: True (syntactically) but Suspicious (logically)
Confidence: 0.85
Issues: ["NPL ratio should not be negative - verify data quality or query logic"]
Suggestions: ["Review if negative NPL ratio makes business sense"]

Example 2:
SQL: SELECT b.BANK_NAME, fp.ROTE FROM BANKS b JOIN FINANCIAL_PERFORMANCE fp ON b.BANK_ID = fp.BANK_ID WHERE fp.ROTE > 50
Execution: Success
Result: 0 rows
Valid: True
Confidence: 0.90
Issues: ["ROTE above 50% is extremely rare - verify if threshold is correct"]
Suggestions: ["Typical ROTE ranges from 5-20%, consider adjusting threshold"]

Example 3:
SQL: SELECT YEAR, QUARTER, AVG(CET1) as avg_cet1 FROM MARKET_DATA GROUP BY YEAR, QUARTER HAVING AVG(CET1) < 10
Execution: Success
Result: 2 rows
Valid: True
Confidence: 0.88
Issues: ["CET1 below 10% indicates regulatory concern - ensure this is intended query"]
Suggestions: ["Results show banks below regulatory minimum, verify if this is the intended analysis"]

Example 4:
SQL: UPDATE FINANCIAL_PERFORMANCE SET DEPOSITS = DEPOSITS * 1.1 WHERE YEAR = 2024
Execution: Success
Rows Affected: 20
Valid: True
Confidence: 0.75
Issues: ["Bulk update affecting multiple banks - ensure this is authorized"]
Suggestions: ["Consider adding specific BANK_ID filter", "Run in transaction for safety"]

### Now validate this execution:
SQL: {sql_query}
Execution Result: {json.dumps(execution_result, indent=2)}

Provide response in JSON format:
{{
    "valid": boolean,
    "confidence": float (0-1),
    "execution_success": boolean,
    "issues": [list of execution issues or warnings],
    "suggestions": [list of suggestions],
    "business_logic_notes": "any banking business logic observations"
}}
"""
        return prompt
    
    def create_correctness_validation_prompt(self, nl_query: str, sql_query: str) -> str:
        """Create prompt for correctness validation - replacing performance validation"""
        prompt = f"""
You are a banking SQL expert validating the logical correctness and business accuracy of queries.

## CORRECTNESS_VALIDATION

### Few-Shot Banking Examples:

Example 1:
Question: "Calculate the loan-to-deposit ratio for each bank"
SQL: SELECT b.BANK_NAME, fp.LOANS_ADVANCES / fp.DEPOSITS * 100 as LDR FROM BANKS b JOIN FINANCIAL_PERFORMANCE fp ON b.BANK_ID = fp.BANK_ID
Correctness Score: 0.60
Confidence: 0.90
Issues: ["Missing handling for NULL values", "Should aggregate if multiple quarters exist", "Division by zero not handled"]
Suggestions: ["Use NULLIF(fp.DEPOSITS, 0) to handle zero deposits", "Add WHERE clause for specific period or use AVG()"]
Business Logic: ["LDR calculation is correct but needs period specification"]

Example 2:
Question: "Find banks with deteriorating asset quality"
SQL: SELECT b.BANK_NAME, fp1.NPL_RATIO as current_npl, fp2.NPL_RATIO as prev_npl FROM BANKS b JOIN FINANCIAL_PERFORMANCE fp1 ON b.BANK_ID = fp1.BANK_ID JOIN FINANCIAL_PERFORMANCE fp2 ON b.BANK_ID = fp2.BANK_ID WHERE fp1.YEAR = 2024 AND fp1.QUARTER = 4 AND fp2.YEAR = 2024 AND fp2.QUARTER = 3 AND fp1.NPL_RATIO > fp2.NPL_RATIO
Correctness Score: 0.95
Confidence: 0.93
Issues: []
Suggestions: ["Consider adding percentage point change calculation"]
Business Logic: ["Correctly compares sequential quarters to identify deterioration"]

Example 3:
Question: "Calculate CASA ratio for all banks"
SQL: SELECT BANK_NAME, CASA FROM BANKS b JOIN FINANCIAL_PERFORMANCE fp ON b.BANK_ID = fp.BANK_ID
Correctness Score: 0.40
Confidence: 0.88
Issues: ["CASA ratio should be CASA/DEPOSITS * 100, not just CASA value", "Missing calculation"]
Suggestions: ["Calculate as: (CASA / DEPOSITS) * 100 as CASA_RATIO"]
Business Logic: ["CASA ratio is a percentage, not absolute value"]

Example 4:
Question: "Show year-over-year profit growth"
SQL: SELECT b.BANK_NAME, ((fp1.YTD_PROFIT - fp2.YTD_PROFIT) / fp2.YTD_PROFIT) * 100 as yoy_growth FROM BANKS b JOIN FINANCIAL_PERFORMANCE fp1 ON b.BANK_ID = fp1.BANK_ID JOIN FINANCIAL_PERFORMANCE fp2 ON b.BANK_ID = fp2.BANK_ID WHERE fp1.YEAR = 2024 AND fp1.QUARTER = 4 AND fp2.YEAR = 2023 AND fp2.QUARTER = 4
Correctness Score: 0.92
Confidence: 0.91
Issues: ["Should handle cases where previous year profit is 0 or negative"]
Suggestions: ["Add NULLIF(fp2.YTD_PROFIT, 0) to prevent division by zero"]
Business Logic: ["YoY calculation is correct using Q4 for full year comparison"]

Example 5:
Question: "Calculate tier 1 capital adequacy"
SQL: SELECT BANK_NAME, CET1 FROM BANKS b JOIN MARKET_DATA md ON b.BANK_ID = md.BANK_ID
Correctness Score: 0.85
Confidence: 0.89
Issues: ["CET1 is already a ratio, not a calculation", "Missing period specification"]
Suggestions: ["Add WHERE clause for specific period", "CET1 = (CET_CAPITAL / RWA) * 100 if recalculation needed"]
Business Logic: ["CET1 is pre-calculated, query is correct if just retrieving the ratio"]

### Now validate correctness for:
Question: {nl_query}
SQL: {sql_query}

Provide response in JSON format:
{{
    "correctness_score": float (0-1),
    "confidence": float (0-1),
    "issues": [list of correctness issues],
    "suggestions": [list of corrections needed],
    "business_logic": [list of business logic validations],
    "calculation_accuracy": "Assessment of any calculations in the query"
}}
"""
        return prompt
    
    def validate_syntax(self, sql_query: str) -> ValidationResult:
        """Stage 1: Syntax Validation"""
        prompt = self.create_syntax_validation_prompt(sql_query)
        response = self.llm.validate(prompt)
        
        return ValidationResult(
            stage=ValidationStage.SYNTAX,
            passed=response.get("valid", False),
            confidence=response.get("confidence", 0.0),
            issues=response.get("issues", []),
            suggestions=response.get("suggestions", []),
            details=response
        )
    
    def validate_schema(self, sql_query: str) -> ValidationResult:
        """Stage 2: Schema Validation"""
        prompt = self.create_schema_validation_prompt(sql_query)
        response = self.llm.validate(prompt)
        
        return ValidationResult(
            stage=ValidationStage.SCHEMA,
            passed=response.get("valid", False),
            confidence=response.get("confidence", 0.0),
            issues=response.get("issues", []),
            suggestions=response.get("suggestions", []),
            details=response
        )
    
    def validate_semantics(self, nl_query: str, sql_query: str) -> ValidationResult:
        """Stage 3: Semantic Validation"""
        prompt = self.create_semantic_validation_prompt(nl_query, sql_query)
        response = self.llm.validate(prompt)
        
        passed = response.get("semantic_match", False) and response.get("alignment_score", 0) > 0.7
        
        return ValidationResult(
            stage=ValidationStage.SEMANTIC,
            passed=passed,
            confidence=response.get("confidence", 0.0),
            issues=response.get("missing_elements", []),
            suggestions=response.get("suggestions", []),
            details=response
        )
    
    def validate_execution(self, sql_query: str) -> ValidationResult:
        """Stage 4: Execution Validation"""
        execution_result = self._simulate_execution(sql_query)
        prompt = self.create_execution_validation_prompt(sql_query, execution_result)
        response = self.llm.validate(prompt)
        
        return ValidationResult(
            stage=ValidationStage.EXECUTION,
            passed=execution_result["success"] and response.get("valid", False),
            confidence=response.get("confidence", 0.0),
            issues=response.get("issues", []),
            suggestions=response.get("suggestions", []),
            details={**response, **execution_result}
        )
    
    def validate_correctness(self, nl_query: str, sql_query: str) -> ValidationResult:
        """Stage 5: Correctness Validation (replaces performance)"""
        prompt = self.create_correctness_validation_prompt(nl_query, sql_query)
        response = self.llm.validate(prompt)
        
        passed = response.get("correctness_score", 0) > 0.7
        
        return ValidationResult(
            stage=ValidationStage.CORRECTNESS,
            passed=passed,
            confidence=response.get("confidence", 0.0),
            issues=response.get("issues", []),
            suggestions=response.get("suggestions", []),
            details=response
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
    
    def calculate_overall_confidence(self, stage_results: Dict[ValidationStage, ValidationResult]) -> Tuple[float, str]:
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
    
    def validate_sql(self, nl_query: str, sql_query: str) -> OverallValidation:
        """Main validation pipeline"""
        logger.info(f"Starting validation pipeline for query: {sql_query[:50]}...")
        
        stage_results = {}
        
        # Stage 1: Syntax Validation
        logger.info("Stage 1: Validating syntax...")
        stage_results[ValidationStage.SYNTAX] = self.validate_syntax(sql_query)
        
        # Stage 2: Schema Validation
        logger.info("Stage 2: Validating schema...")
        stage_results[ValidationStage.SCHEMA] = self.validate_schema(sql_query)
        
        # Stage 3: Semantic Validation
        logger.info("Stage 3: Validating semantics...")
        stage_results[ValidationStage.SEMANTIC] = self.validate_semantics(nl_query, sql_query)
        
        # Stage 4: Execution Validation
        logger.info("Stage 4: Validating execution...")
        stage_results[ValidationStage.EXECUTION] = self.validate_execution(sql_query)
        
        # Stage 5: Correctness Validation
        logger.info("Stage 5: Validating correctness...")
        stage_results[ValidationStage.CORRECTNESS] = self.validate_correctness(nl_query, sql_query)
        
        # Calculate overall confidence
        overall_confidence, verdict = self.calculate_overall_confidence(stage_results)
        
        # Generate explanation
        explanation = self._generate_explanation(stage_results, overall_confidence)
        
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


def main():
    """Main function to demonstrate the banking SQL validation pipeline"""
    
    # Initialize validator
    validator = BankingSQLValidator()
    
    # Banking-specific test cases
    test_cases = [
        {
            "description": "Valid query - Top banks by profit",
            "nl_query": "Show me the top 5 banks by profit in Q4 2024",
            "sql_query": """
                SELECT b.BANK_NAME, fp.QUARTERLY_PROFIT 
                FROM BANKS b 
                JOIN FINANCIAL_PERFORMANCE fp ON b.BANK_ID = fp.BANK_ID 
                WHERE fp.YEAR = 2024 AND fp.QUARTER = 4 
                ORDER BY fp.QUARTERLY_PROFIT DESC 
                LIMIT 5
            """
        },
        {
            "description": "Query with syntax error",
            "nl_query": "Find banks with high NPL ratio",
            "sql_query": """
                SELCT b.BANK_NAME, fp.NPL_RATIO 
                FORM BANKS b 
                JOIN FINANCIAL_PERFORMANCE fp ON b.BANK_ID = fp.BANK_ID 
                WHERE fp.NPL_RATIO > 5
            """
        },
        {
            "description": "Semantically incorrect query",
            "nl_query": "Show banks with improving cost-to-income ratio",
            "sql_query": """
                SELECT BANK_NAME 
                FROM BANKS
            """
        },
        {
            "description": "Incorrect calculation",
            "nl_query": "Calculate CASA ratio for all banks in Q4 2024",
            "sql_query": """
                SELECT b.BANK_NAME, fp.CASA 
                FROM BANKS b 
                JOIN FINANCIAL_PERFORMANCE fp ON b.BANK_ID = fp.BANK_ID
                WHERE fp.YEAR = 2024 AND fp.QUARTER = 4
            """
        },
        {
            "description": "Complex valid query - YoY comparison",
            "nl_query": "Compare year-over-year profit growth for all banks",
            "sql_query": """
                SELECT 
                    b.BANK_NAME,
                    fp1.YTD_PROFIT as current_profit,
                    fp2.YTD_PROFIT as prev_profit,
                    ((fp1.YTD_PROFIT - fp2.YTD_PROFIT) / NULLIF(fp2.YTD_PROFIT, 0)) * 100 as yoy_growth
                FROM BANKS b
                JOIN FINANCIAL_PERFORMANCE fp1 ON b.BANK_ID = fp1.BANK_ID
                JOIN FINANCIAL_PERFORMANCE fp2 ON b.BANK_ID = fp2.BANK_ID
                WHERE fp1.YEAR = 2024 AND fp1.QUARTER = 4
                  AND fp2.YEAR = 2023 AND fp2.QUARTER = 4
                ORDER BY yoy_growth DESC
            """
        },
        {
            "description": "Risk metrics analysis",
            "nl_query": "Find banks with CET1 ratio below regulatory minimum",
            "sql_query": """
                SELECT b.BANK_NAME, md.CET1, md.RWA, md.CET_CAPITAL
                FROM BANKS b
                JOIN MARKET_DATA md ON b.BANK_ID = md.BANK_ID
                WHERE md.CET1 < 10.5
                  AND md.YEAR = 2024
                  AND md.QUARTER = 4
            """
        }
    ]
    
    # Run validation for each test case
    print("\n" + "="*80)
    print("BANKING SQL VALIDATION PIPELINE - Using GPT-4o-mini")
    print("="*80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}: {test_case['description']}")
        print(f"{'='*80}")
        print(f"Business Question: {test_case['nl_query']}")
        print(f"SQL Query: {test_case['sql_query'].strip()}")
        print("-" * 80)
        
        # Validate
        result = validator.validate_sql(
            nl_query=test_case['nl_query'],
            sql_query=test_case['sql_query'].strip()
        )
        
        # Print results
        print(f"\n{result.explanation}")
        
        if result.corrected_sql and result.corrected_sql != result.sql_query:
            print(f"\n{'='*50}")
            print("SUGGESTED CORRECTION:")
            print(result.corrected_sql)
        
        print(f"\n{'='*50}")
        print(f"FINAL VERDICT: {result.final_verdict}")
        print(f"{'='*80}\n")
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION PIPELINE TEST COMPLETE")
    print("="*80)
    print(f"Total test cases: {len(test_cases)}")
    print("\nPipeline Features:")
    print("✓ Syntax validation with banking SQL patterns")
    print("✓ Schema validation against banking data dictionary")  
    print("✓ Semantic alignment with banking business questions")
    print("✓ Execution validation with business logic checks")
    print("✓ Correctness validation for calculations and ratios")
    print("✓ Powered by OpenAI GPT-4o-mini model")
    print("✓ Banking-specific few-shot examples")


if __name__ == "__main__":
    # Set your OpenAI API key as environment variable
    # export OPENAI_API_KEY='your-api-key-here'
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
        print("Please set it using: export OPENAI_API_KEY='your-api-key-here'")
    else:
        main()