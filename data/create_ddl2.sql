-- =====================================================
-- Oracle Database DDL Script for Banking Data Warehouse
-- =====================================================
-- This script creates the complete schema for banking analytics
-- including tables for banks, financial performance, and market data
-- =====================================================

-- Drop tables if they exist (in reverse order of dependencies)
BEGIN
   EXECUTE IMMEDIATE 'DROP TABLE MARKET_DATA CASCADE CONSTRAINTS';
EXCEPTION
   WHEN OTHERS THEN
      IF SQLCODE != -942 THEN
         RAISE;
      END IF;
END;
/

BEGIN
   EXECUTE IMMEDIATE 'DROP TABLE FINANCIAL_PERFORMANCE CASCADE CONSTRAINTS';
EXCEPTION
   WHEN OTHERS THEN
      IF SQLCODE != -942 THEN
         RAISE;
      END IF;
END;
/

BEGIN
   EXECUTE IMMEDIATE 'DROP TABLE BANKS CASCADE CONSTRAINTS';
EXCEPTION
   WHEN OTHERS THEN
      IF SQLCODE != -942 THEN
         RAISE;
      END IF;
END;
/

-- =====================================================
-- Create BANKS table
-- Master table containing bank information
-- =====================================================
CREATE TABLE BANKS (
    BANK_ID           NUMBER(38,0) NOT NULL,
    BANK_NAME         VARCHAR2(26),
    BANK_CODE         VARCHAR2(26),
    ESTABLISHED_DATE  DATE,
    HEADQUARTERS      VARCHAR2(100),
    CONSTRAINT PK_BANKS PRIMARY KEY (BANK_ID)
);

-- Create index on frequently queried columns
CREATE INDEX IDX_BANKS_CODE ON BANKS(BANK_CODE);
CREATE INDEX IDX_BANKS_NAME ON BANKS(BANK_NAME);

-- Add comments to the table and columns
COMMENT ON TABLE BANKS IS 'Master table containing bank information including identification details and codes for financial institutions';
COMMENT ON COLUMN BANKS.BANK_ID IS 'Unique identifier for each bank in the system';
COMMENT ON COLUMN BANKS.BANK_NAME IS 'Official registered name of the bank';
COMMENT ON COLUMN BANKS.BANK_CODE IS 'Short code or abbreviation used to identify the bank in transactions and reports';
COMMENT ON COLUMN BANKS.ESTABLISHED_DATE IS 'Date when the bank was established';
COMMENT ON COLUMN BANKS.HEADQUARTERS IS 'Location of the bank''s headquarters';

-- =====================================================
-- Create FINANCIAL_PERFORMANCE table
-- Quarterly and yearly financial performance metrics
-- =====================================================
CREATE TABLE FINANCIAL_PERFORMANCE (
    PERFORMANCE_ID    NUMBER(38,0) NOT NULL,
    YEAR             NUMBER(38,0),
    QUARTER          NUMBER(38,0),
    BANK_ID          NUMBER(38,0),
    YTD_INCOME       NUMBER(38,1),
    YTD_PROFIT       NUMBER(38,1),
    QUARTERLY_PROFIT NUMBER(38,1),
    LOANS_ADVANCES   NUMBER(38,1),
    NIM              NUMBER(38,1),
    DEPOSITS         NUMBER(38,1),
    CASA             NUMBER(38,1),
    COST_INCOME      NUMBER(38,1),
    NPL_RATIO        NUMBER(38,1),
    COR              NUMBER(38,2),
    STAGE3_COVER     NUMBER(38,1),
    ROTE             NUMBER(38,1),
    CONSTRAINT PK_FINANCIAL_PERFORMANCE PRIMARY KEY (PERFORMANCE_ID),
    CONSTRAINT FK_FIN_PERF_BANK FOREIGN KEY (BANK_ID) 
        REFERENCES BANKS(BANK_ID),
    CONSTRAINT CHK_QUARTER CHECK (QUARTER IN (1, 2, 3, 4)),
    CONSTRAINT CHK_YEAR CHECK (YEAR >= 2000 AND YEAR <= 2100)
);

-- Create indexes for performance optimization
CREATE INDEX IDX_FIN_PERF_BANK_ID ON FINANCIAL_PERFORMANCE(BANK_ID);
CREATE INDEX IDX_FIN_PERF_YEAR_QTR ON FINANCIAL_PERFORMANCE(YEAR, QUARTER);
CREATE INDEX IDX_FIN_PERF_COMPOSITE ON FINANCIAL_PERFORMANCE(BANK_ID, YEAR, QUARTER);

-- Add comments to the table and columns
COMMENT ON TABLE FINANCIAL_PERFORMANCE IS 'Quarterly and yearly financial performance metrics for banks including income, profit, loans, deposits, and key financial ratios';
COMMENT ON COLUMN FINANCIAL_PERFORMANCE.PERFORMANCE_ID IS 'Unique identifier for each financial performance record';
COMMENT ON COLUMN FINANCIAL_PERFORMANCE.YEAR IS 'Fiscal year of the financial performance data';
COMMENT ON COLUMN FINANCIAL_PERFORMANCE.QUARTER IS 'Quarter of the fiscal year (1-4) for the performance data';
COMMENT ON COLUMN FINANCIAL_PERFORMANCE.BANK_ID IS 'Reference to the bank this performance data belongs to';
COMMENT ON COLUMN FINANCIAL_PERFORMANCE.YTD_INCOME IS 'Year-to-date total income in millions of currency units';
COMMENT ON COLUMN FINANCIAL_PERFORMANCE.YTD_PROFIT IS 'Year-to-date net profit in millions of currency units';
COMMENT ON COLUMN FINANCIAL_PERFORMANCE.QUARTERLY_PROFIT IS 'Net profit for the specific quarter in millions of currency units';
COMMENT ON COLUMN FINANCIAL_PERFORMANCE.LOANS_ADVANCES IS 'Total loans and advances issued by the bank in millions';
COMMENT ON COLUMN FINANCIAL_PERFORMANCE.NIM IS 'Net Interest Margin - percentage difference between interest earned and interest paid';
COMMENT ON COLUMN FINANCIAL_PERFORMANCE.DEPOSITS IS 'Total customer deposits held by the bank in millions';
COMMENT ON COLUMN FINANCIAL_PERFORMANCE.CASA IS 'Current Account and Savings Account deposits in millions - low-cost funding source';
COMMENT ON COLUMN FINANCIAL_PERFORMANCE.COST_INCOME IS 'Cost-to-Income ratio - operating expenses as percentage of operating income';
COMMENT ON COLUMN FINANCIAL_PERFORMANCE.NPL_RATIO IS 'Non-Performing Loans ratio - percentage of loans in default or close to default';
COMMENT ON COLUMN FINANCIAL_PERFORMANCE.COR IS 'Cost of Risk - provision for loan losses as percentage of total loans';
COMMENT ON COLUMN FINANCIAL_PERFORMANCE.STAGE3_COVER IS 'Stage 3 coverage ratio - provisions for impaired loans as percentage of Stage 3 loans';
COMMENT ON COLUMN FINANCIAL_PERFORMANCE.ROTE IS 'Return on Tangible Equity - net income as percentage of tangible shareholder equity';

-- =====================================================
-- Create MARKET_DATA table
-- Market and capital-related metrics for banks
-- =====================================================
CREATE TABLE MARKET_DATA (
    MARKET_ID         NUMBER(38,0) NOT NULL,
    YEAR             NUMBER(38,0),
    QUARTER          NUMBER(38,0),
    BANK_ID          NUMBER(38,0),
    CET1             NUMBER(38,1),
    CET_CAPITAL      NUMBER(38,1),
    RWA              NUMBER(38,1),
    SHARE_PRICE      NUMBER(38,2),
    MARKET_CAP_AED_BN NUMBER(38,2),
    MARKET_CAP_USD_BN NUMBER(38,2),
    PE_RATIO         NUMBER(38,2),
    PB_RATIO         NUMBER(38,2),
    CONSTRAINT PK_MARKET_DATA PRIMARY KEY (MARKET_ID),
    CONSTRAINT FK_MARKET_DATA_BANK FOREIGN KEY (BANK_ID) 
        REFERENCES BANKS(BANK_ID),
    CONSTRAINT CHK_MKT_QUARTER CHECK (QUARTER IN (1, 2, 3, 4)),
    CONSTRAINT CHK_MKT_YEAR CHECK (YEAR >= 2000 AND YEAR <= 2100)
);

-- Create indexes for performance optimization
CREATE INDEX IDX_MARKET_DATA_BANK_ID ON MARKET_DATA(BANK_ID);
CREATE INDEX IDX_MARKET_DATA_YEAR_QTR ON MARKET_DATA(YEAR, QUARTER);
CREATE INDEX IDX_MARKET_DATA_COMPOSITE ON MARKET_DATA(BANK_ID, YEAR, QUARTER);

-- Add comments to the table and columns
COMMENT ON TABLE MARKET_DATA IS 'Market and capital-related metrics for banks including regulatory capital ratios, share prices, and market capitalization';
COMMENT ON COLUMN MARKET_DATA.MARKET_ID IS 'Unique identifier for each market data record';
COMMENT ON COLUMN MARKET_DATA.YEAR IS 'Year for which the market data is recorded';
COMMENT ON COLUMN MARKET_DATA.QUARTER IS 'Quarter of the year (1-4) for the market data';
COMMENT ON COLUMN MARKET_DATA.BANK_ID IS 'Reference to the bank this market data belongs to';
COMMENT ON COLUMN MARKET_DATA.CET1 IS 'Common Equity Tier 1 capital ratio - core capital as percentage of risk-weighted assets';
COMMENT ON COLUMN MARKET_DATA.CET_CAPITAL IS 'Common Equity Tier 1 capital amount in millions';
COMMENT ON COLUMN MARKET_DATA.RWA IS 'Risk-Weighted Assets - total assets adjusted for credit risk in millions';
COMMENT ON COLUMN MARKET_DATA.SHARE_PRICE IS 'Stock price of the bank''s shares in local currency';
COMMENT ON COLUMN MARKET_DATA.MARKET_CAP_AED_BN IS 'Market capitalization in billions of UAE Dirhams';
COMMENT ON COLUMN MARKET_DATA.MARKET_CAP_USD_BN IS 'Market capitalization in billions of US Dollars';
COMMENT ON COLUMN MARKET_DATA.PE_RATIO IS 'Price-to-Earnings ratio - share price relative to earnings per share';
COMMENT ON COLUMN MARKET_DATA.PB_RATIO IS 'Price-to-Book ratio - market value relative to book value';

-- =====================================================
-- Create sequences for auto-incrementing IDs (optional)
-- =====================================================
CREATE SEQUENCE SEQ_BANKS_ID 
    START WITH 100 
    INCREMENT BY 1 
    NOCACHE 
    NOCYCLE;

CREATE SEQUENCE SEQ_FINANCIAL_PERFORMANCE_ID 
    START WITH 1000 
    INCREMENT BY 1 
    NOCACHE 
    NOCYCLE;

CREATE SEQUENCE SEQ_MARKET_DATA_ID 
    START WITH 1000 
    INCREMENT BY 1 
    NOCACHE 
    NOCYCLE;

-- =====================================================
-- Create views for common queries
-- =====================================================

-- View: Latest financial performance for each bank
CREATE OR REPLACE VIEW V_LATEST_FINANCIAL_PERFORMANCE AS
SELECT 
    b.BANK_NAME,
    b.BANK_CODE,
    fp.YEAR,
    fp.QUARTER,
    fp.YTD_INCOME,
    fp.YTD_PROFIT,
    fp.QUARTERLY_PROFIT,
    fp.LOANS_ADVANCES,
    fp.NIM,
    fp.DEPOSITS,
    fp.CASA,
    fp.COST_INCOME,
    fp.NPL_RATIO,
    fp.COR,
    fp.STAGE3_COVER,
    fp.ROTE
FROM FINANCIAL_PERFORMANCE fp
JOIN BANKS b ON fp.BANK_ID = b.BANK_ID
WHERE (fp.BANK_ID, fp.YEAR, fp.QUARTER) IN (
    SELECT BANK_ID, MAX(YEAR), MAX(QUARTER)
    FROM FINANCIAL_PERFORMANCE
    WHERE YEAR = (SELECT MAX(YEAR) FROM FINANCIAL_PERFORMANCE)
    GROUP BY BANK_ID
);

-- View: Latest market data for each bank
CREATE OR REPLACE VIEW V_LATEST_MARKET_DATA AS
SELECT 
    b.BANK_NAME,
    b.BANK_CODE,
    md.YEAR,
    md.QUARTER,
    md.CET1,
    md.CET_CAPITAL,
    md.RWA,
    md.SHARE_PRICE,
    md.MARKET_CAP_AED_BN,
    md.MARKET_CAP_USD_BN,
    md.PE_RATIO,
    md.PB_RATIO
FROM MARKET_DATA md
JOIN BANKS b ON md.BANK_ID = b.BANK_ID
WHERE (md.BANK_ID, md.YEAR, md.QUARTER) IN (
    SELECT BANK_ID, MAX(YEAR), MAX(QUARTER)
    FROM MARKET_DATA
    WHERE YEAR = (SELECT MAX(YEAR) FROM MARKET_DATA)
    GROUP BY BANK_ID
);

-- View: Comprehensive bank performance dashboard
CREATE OR REPLACE VIEW V_BANK_PERFORMANCE_DASHBOARD AS
SELECT 
    b.BANK_ID,
    b.BANK_NAME,
    b.BANK_CODE,
    b.HEADQUARTERS,
    fp.YEAR,
    fp.QUARTER,
    fp.YTD_INCOME,
    fp.YTD_PROFIT,
    fp.QUARTERLY_PROFIT,
    fp.LOANS_ADVANCES,
    fp.NIM,
    fp.DEPOSITS,
    fp.CASA,
    fp.COST_INCOME,
    fp.NPL_RATIO,
    fp.COR,
    fp.STAGE3_COVER,
    fp.ROTE,
    md.CET1,
    md.CET_CAPITAL,
    md.RWA,
    md.SHARE_PRICE,
    md.MARKET_CAP_AED_BN,
    md.MARKET_CAP_USD_BN,
    md.PE_RATIO,
    md.PB_RATIO
FROM BANKS b
LEFT JOIN FINANCIAL_PERFORMANCE fp ON b.BANK_ID = fp.BANK_ID
LEFT JOIN MARKET_DATA md ON b.BANK_ID = md.BANK_ID 
    AND fp.YEAR = md.YEAR 
    AND fp.QUARTER = md.QUARTER
ORDER BY b.BANK_ID, fp.YEAR DESC, fp.QUARTER DESC;

-- =====================================================
-- Grant permissions (adjust as needed for your environment)
-- =====================================================
-- GRANT SELECT ON BANKS TO readonly_user;
-- GRANT SELECT ON FINANCIAL_PERFORMANCE TO readonly_user;
-- GRANT SELECT ON MARKET_DATA TO readonly_user;
-- GRANT SELECT ON V_LATEST_FINANCIAL_PERFORMANCE TO readonly_user;
-- GRANT SELECT ON V_LATEST_MARKET_DATA TO readonly_user;
-- GRANT SELECT ON V_BANK_PERFORMANCE_DASHBOARD TO readonly_user;

-- =====================================================
-- Success message
-- =====================================================
BEGIN
    DBMS_OUTPUT.PUT_LINE('====================================================');
    DBMS_OUTPUT.PUT_LINE('Banking Data Warehouse Schema Created Successfully!');
    DBMS_OUTPUT.PUT_LINE('====================================================');
    DBMS_OUTPUT.PUT_LINE('Tables created:');
    DBMS_OUTPUT.PUT_LINE('  - BANKS');
    DBMS_OUTPUT.PUT_LINE('  - FINANCIAL_PERFORMANCE');
    DBMS_OUTPUT.PUT_LINE('  - MARKET_DATA');
    DBMS_OUTPUT.PUT_LINE('Views created:');
    DBMS_OUTPUT.PUT_LINE('  - V_LATEST_FINANCIAL_PERFORMANCE');
    DBMS_OUTPUT.PUT_LINE('  - V_LATEST_MARKET_DATA');
    DBMS_OUTPUT.PUT_LINE('  - V_BANK_PERFORMANCE_DASHBOARD');
    DBMS_OUTPUT.PUT_LINE('====================================================');
END;
/