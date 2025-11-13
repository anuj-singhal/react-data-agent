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


@dataclass
class ClassificationResponse:
    classification: str
    message: str
    action_type: str
    answer: str

class NLClassificationAgent:
    """Advanced intent extraction and classification for database queries."""

    def __init__(self):
        self.llm = None
        if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
            self.llm = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("OpenAI client initialized for Intent Agent")
    

        
    def _get_classification_prompt(self) -> str:
        """
        Returns the classification prompt with table descriptions and few-shot examples.
        """
        return """You are a classification agent that determines whether a user's query is related to banking/financial data or not.

## Available Data Tables:

1. **BANKS Table**: Master table containing bank information including identification details and codes for financial institutions. Contains data about:
   - Bank names (e.g., Mashreq, ADCB, DIB, ENBD, FAB)
   - Bank codes and identifiers
   - Establishment dates
   - Headquarters locations (Dubai, Abu Dhabi)

2. **FINANCIAL_PERFORMANCE Table**: Quarterly and yearly financial performance metrics for banks including:
   - Income metrics (YTD income, YTD profit, quarterly profit)
   - Loan and deposit information (loans/advances, deposits, CASA)
   - Financial ratios (NIM, Cost-to-Income ratio, NPL ratio, ROTE)
   - Risk metrics (Cost of Risk, Stage 3 coverage)
   - Time periods (years 2020-2024, quarters 1-4)

3. **MARKET_DATA Table**: Market and capital-related metrics for banks including:
   - Regulatory capital ratios (CET1, CET capital, RWA)
   - Stock market data (share prices, market capitalization in AED/USD)
   - Valuation metrics (PE ratio, PB ratio)
   - Time periods (years and quarters)

## Classification Task:
Classify the user query as either:
- "process": The query is related to banking, financial performance, market data, or any metrics available in the above tables
- "not_process": The query is unrelated to banking/financial data (general questions, personal topics, non-financial subjects)

## Few-Shot Examples:

User: "What is the NPL ratio for ADCB in Q3 2023?"
Classification: process
Reason: Query asks about NPL ratio (available in FINANCIAL_PERFORMANCE table) for a specific bank (ADCB)
Answer: None

User: "Show me the quarterly profit trends for all UAE banks"
Classification: process
Reason: Query asks about quarterly profit data which is available in FINANCIAL_PERFORMANCE table
Message: None

User: "Modify the query based on quarterly results"
Classification: process
Reason: Its a modify Query asks about quarterly profit data which is available in FINANCIAL_PERFORMANCE table
Message: None

User: "Which bank has the highest or maximum profit"
Classification: process
Reason: Its a follow up Query asks about profit data of bank which is available in FINANCIAL_PERFORMANCE table
Message: None

User: "What's the weather like today?"
Classification: not_process
Reason: Weather is unrelated to banking/financial data
Answer: current weather is windy/cloudy/rainy


User: "Compare the Cost-to-Income ratios between FAB and ENBD"
Classification: process
Reason: Query asks about Cost-to-Income ratio comparison between banks (data available in FINANCIAL_PERFORMANCE)
Answer: None

User: "Can you explain quantum physics?"
Classification: not_process
Reason: Physics is unrelated to banking/financial domain
Answer: quantum physics is ...

User: "What is the market cap of Mashreq bank in USD?"
Classification: process
Reason: Query asks about market capitalization which is available in MARKET_DATA table
Answer: None

User: "Tell me a joke"
Classification: not_process
Reason: Entertainment request unrelated to financial data
Answer: joke is...

User: "Show the CET1 capital ratios for all banks in 2023"
Classification: process
Reason: Query asks about CET1 ratios which are available in MARKET_DATA table
Answer: joke is...

User: "What's your favorite color?"
Classification: not_process
Reason: Personal preference question unrelated to banking data
Answer: favorite color is blue

User: "How do I cook pasta?"
Classification: not_process
Reason: Cooking instructions are unrelated to banking/financial data\
Answer: Cooking instructions are ...

User: "What is the capital of India?"
Classification: not_process
Reason: Asking capital of a country is not related to banking
Answer: Capital of India is Delhi

User: "Show me the PE ratios for banks headquartered in Dubai"
Classification: process
Reason: Query combines headquarters info (BANKS table) with PE ratios (MARKET_DATA table)

User: "What is machine learning?"
Classification: not_process
Reason: General technology question unrelated to banking/financial data
Answer: None

User: "List all banks established before 1970"
Classification: process
Reason: Query asks about bank establishment dates which are available in BANKS table
Answer: None

## Instructions:
1. Analyze the user query carefully
2. Determine if it relates to any banking, financial, or market data from the tables described above
3. Consider queries about bank names, financial metrics, ratios, time periods, comparisons, trends, or analysis as "process"
4. Consider general questions, personal topics, or non-financial subjects as "not_process"
5. In case of not_process Answer the question asked as per your understanding and add into Answer:
6. In case of process make answer None.
7. NL Query can be a modified query related to financial data.
8. NL query can be a followup query related to the results displayed. 

Respond with ONLY a JSON object in this format:
{
    "classification": "process" or "not_process",
    "confidence": 0.0 to 1.0,
    "reason": "brief explanation"
    "answer": "in case of not_process provide answer"
}

User Query: """

    async def classify_query(self, user_query: str) -> ClassificationResponse:
        """
        Classify whether the user query is related to banking/financial data.
        
        Args:
            user_query: The user's natural language query
            session_id: Current session identifier
            
        Returns:
            ChatResponse with appropriate message and action type
        """
        try:
            # Prepare the prompt
            full_prompt = self._get_classification_prompt() + user_query
            
            # Call OpenAI API
            response = await self.llm.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise classification agent. Respond only with valid JSON."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=150
            )
            
            # Parse the response
            result_text = response.choices[0].message.content.strip()
            answer = ""
            # Handle potential JSON parsing issues
            try:
                # Remove any markdown code blocks if present
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0].strip()
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0].strip()
                
                result = json.loads(result_text)
                classification = result.get("classification", "not_process")
                confidence = result.get("confidence", 0.5)
                reason = result.get("reason", "")
                answer = result.get("answer", "")
                
            except json.JSONDecodeError:
                # Fallback: try to extract classification from text
                if "process" in result_text.lower() and "not_process" not in result_text.lower():
                    classification = "process"
                else:
                    classification = "not_process"
                confidence = 0.5
                reason = "Classification extracted from response"
            
            # Return appropriate response based on classification
            if classification == "not_process":
                return ClassificationResponse(
                    classification = classification,
                    message=" - But, I don't quite understand the relation of this question with data agent, Please re-phrase your question.",
                    action_type="general",
                    answer = answer
                )
            else:
                # For "process" classification, you would typically proceed with the data agent
                return ClassificationResponse(
                    classification = classification,
                    message="Processing your financial data query...",
                    action_type="general",
                    answer = answer
                )
                
        except Exception as e:
            # Handle any errors in classification
            print(f"Classification error: {str(e)}")
            # Default to not_process for safety
            return ClassificationResponse(
                classification = "not_process",
                message="I don't quite understand please rephrase the question",
                action_type="error",
                answer = "Error in processing"
            )
