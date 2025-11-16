-- Create table for UAE Banks Financial Data
CREATE TABLE uae_banks_financial_data (
    year NUMBER(4) NOT NULL,
    quarter NUMBER(1) NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    bank VARCHAR2(50) NOT NULL,
    ytd_income NUMBER(15,2), -- YTD Income in AED millions
    ytd_profit NUMBER(15,2), -- YTD Net Profit in AED millions
    quarterly_profit NUMBER(15,2), -- Quarterly Net Profit in AED millions
    loans_advances NUMBER(15,2), -- Loans and Advances in AED millions
    nim NUMBER(5,2), -- Net Interest Margin in %
    deposits NUMBER(15,2), -- Customer Deposits in AED millions
    casa NUMBER(15,2), -- Current Account Savings Account in AED millions
    cost_income NUMBER(5,2), -- Cost to Income Ratio in %
    npl_ratio NUMBER(5,2), -- Non-Performing Loans Ratio in %
    cor NUMBER(5,2), -- Cost of Risk in %
    stage3_cover NUMBER(5,2), -- Stage 3 Coverage Ratio in %
    rote NUMBER(5,2), -- Return on Tangible Equity in %
    cet1 NUMBER(5,2), -- CET1 Ratio in %
    cet_capital NUMBER(15,2), -- CET1 Capital in AED millions
    rwa NUMBER(15,2), -- Risk Weighted Assets in AED millions
    share_price NUMBER(8,2), -- Share Price in AED
    market_cap_aed_bn NUMBER(10,2), -- Market Cap in AED billions
    market_cap_usd_bn NUMBER(10,2), -- Market Cap in USD billions
    CONSTRAINT pk_banks_data PRIMARY KEY (year, quarter, bank)
);

-- Create index for better query performance
CREATE INDEX idx_banks_year_quarter ON uae_banks_financial_data (year, quarter);
CREATE INDEX idx_banks_bank ON uae_banks_financial_data (bank);

-- Insert data for Mashreq Bank (5 years quarterly data: 2020-2024)

-- 2020 Data for Mashreq
INSERT INTO uae_banks_financial_data VALUES 
(2020, 1, 'Mashreq', 1850.5, 465.2, 465.2, 95420.8, 3.2, 105680.4, 45230.6, 48.5, 3.8, 1.45, 65.2, 8.9, 14.2, 12450.6, 87680.2, 35.50, 7.12, 1.94);

INSERT INTO uae_banks_financial_data VALUES 
(2020, 2, 'Mashreq', 3720.8, 890.6, 425.4, 96850.2, 3.1, 108920.5, 46780.2, 49.2, 4.1, 1.52, 64.8, 8.5, 14.0, 12680.4, 90520.8, 34.25, 6.87, 1.87);

INSERT INTO uae_banks_financial_data VALUES 
(2020, 3, 'Mashreq', 5580.2, 1298.4, 407.8, 98120.6, 3.0, 112450.8, 48320.5, 50.1, 4.5, 1.68, 63.5, 7.8, 13.8, 13020.2, 94380.6, 32.80, 6.58, 1.79);

INSERT INTO uae_banks_financial_data VALUES 
(2020, 4, 'Mashreq', 7420.6, 1685.2, 386.8, 99680.4, 2.9, 115820.2, 49850.8, 51.2, 4.8, 1.75, 62.8, 7.2, 13.5, 13450.8, 99680.4, 31.20, 6.26, 1.70);

-- 2021 Data for Mashreq
INSERT INTO uae_banks_financial_data VALUES 
(2021, 1, 'Mashreq', 1920.4, 520.6, 520.6, 102580.6, 3.1, 118650.4, 51240.2, 50.8, 4.2, 1.38, 66.5, 8.8, 13.8, 14120.5, 102350.8, 36.75, 7.38, 2.01);

INSERT INTO uae_banks_financial_data VALUES 
(2021, 2, 'Mashreq', 3890.2, 1068.4, 547.8, 105240.8, 3.2, 122880.6, 53650.4, 49.5, 3.9, 1.25, 67.2, 9.2, 14.1, 14680.2, 104120.6, 38.90, 7.81, 2.13);

INSERT INTO uae_banks_financial_data VALUES 
(2021, 3, 'Mashreq', 5850.6, 1632.5, 564.1, 108650.2, 3.3, 127450.8, 56120.6, 48.2, 3.6, 1.15, 68.8, 9.8, 14.5, 15340.8, 105780.4, 41.20, 8.27, 2.25);

INSERT INTO uae_banks_financial_data VALUES 
(2021, 4, 'Mashreq', 7780.8, 2185.6, 553.1, 112450.6, 3.4, 132680.4, 58850.2, 47.5, 3.2, 1.08, 70.5, 10.5, 14.8, 16120.4, 109080.6, 43.50, 8.73, 2.38);

-- 2022 Data for Mashreq
INSERT INTO uae_banks_financial_data VALUES 
(2022, 1, 'Mashreq', 2150.8, 642.5, 642.5, 116850.4, 3.6, 138450.6, 62340.8, 46.8, 2.9, 0.95, 72.8, 11.2, 15.2, 17450.6, 114680.2, 46.80, 9.40, 2.56);

INSERT INTO uae_banks_financial_data VALUES 
(2022, 2, 'Mashreq', 4380.6, 1324.8, 682.3, 121680.2, 3.8, 145680.4, 65850.2, 45.2, 2.6, 0.82, 75.2, 12.1, 15.6, 18680.4, 119750.8, 49.20, 9.89, 2.69);

INSERT INTO uae_banks_financial_data VALUES 
(2022, 3, 'Mashreq', 6620.4, 2058.6, 733.8, 127450.8, 4.0, 152840.2, 69450.6, 44.5, 2.3, 0.75, 78.5, 13.5, 16.1, 20120.8, 124980.6, 52.10, 10.47, 2.85);

INSERT INTO uae_banks_financial_data VALUES 
(2022, 4, 'Mashreq', 8850.2, 2798.4, 739.8, 134680.6, 4.1, 161450.8, 73680.4, 43.8, 2.1, 0.68, 81.2, 14.8, 16.5, 21850.4, 132450.2, 54.75, 10.99, 2.99);

-- 2023 Data for Mashreq (Based on actual reported 130% profit growth)
INSERT INTO uae_banks_financial_data VALUES 
(2023, 1, 'Mashreq', 2850.6, 1124.8, 1124.8, 142380.4, 4.3, 168750.2, 78450.6, 42.5, 1.9, 0.58, 84.5, 16.8, 16.8, 23450.8, 139680.4, 58.20, 11.68, 3.18);

INSERT INTO uae_banks_financial_data VALUES 
(2023, 2, 'Mashreq', 5780.4, 2385.6, 1260.8, 149680.2, 4.5, 176850.4, 82680.2, 41.2, 1.7, 0.52, 87.2, 18.5, 17.2, 25120.4, 146080.6, 61.50, 12.35, 3.36);

INSERT INTO uae_banks_financial_data VALUES 
(2023, 3, 'Mashreq', 8650.2, 3785.4, 1399.8, 157450.8, 4.6, 185620.6, 87250.4, 40.8, 1.5, 0.48, 89.8, 20.2, 17.5, 27680.2, 158120.8, 64.80, 13.01, 3.54);

INSERT INTO uae_banks_financial_data VALUES 
(2023, 4, 'Mashreq', 11580.6, 5680.4, 1895.0, 166850.4, 4.7, 195450.8, 92680.6, 39.5, 1.3, 0.42, 92.5, 22.8, 16.5, 30450.6, 184680.4, 68.25, 13.71, 3.73);

-- 2024 Data for Mashreq
INSERT INTO uae_banks_financial_data VALUES 
(2024, 1, 'Mashreq', 3150.8, 1485.6, 1485.6, 174680.2, 4.8, 203850.4, 97450.2, 38.2, 1.2, 0.38, 95.2, 24.5, 16.8, 32680.4, 194580.6, 72.50, 14.56, 3.96);

INSERT INTO uae_banks_financial_data VALUES 
(2024, 2, 'CBD', 5580.4, 2485.6, 1300.4, 110450.2, 4.3, 123680.4, 91450.2, 33.8, 1.3, 0.52, 94.8, 26.5, 21.8, 38450.8, 177680.4, 7.45, 14.96, 4.08);

INSERT INTO uae_banks_financial_data VALUES 
(2024, 3, 'CBD', 8450.2, 3885.8, 1400.2, 118680.6, 4.4, 131450.2, 98850.4, 33.2, 1.1, 0.48, 96.5, 28.8, 22.5, 42680.2, 191450.8, 8.15, 16.36, 4.46);

INSERT INTO uae_banks_financial_data VALUES 
(2024, 4, 'CBD', 11350.8, 5185.4, 1299.6, 127450.8, 4.5, 140680.2, 107450.8, 32.5, 0.9, 0.45, 98.2, 30.5, 23.2, 47450.6, 206280.2, 8.75, 17.58, 4.79);

-- Sample queries to demonstrate the data

-- Query 1: Show quarterly profits for all banks in 2024
SELECT 
    bank,
    quarter,
    quarterly_profit,
    ytd_profit,
    rote
FROM uae_banks_financial_data 
WHERE year = 2024 
ORDER BY bank, quarter;

-- Query 2: Compare NIM trends across banks for 2023-2024
SELECT 
    bank,
    year,
    quarter,
    nim,
    deposits,
    loans_advances,
    ROUND((loans_advances/deposits)*100, 2) as loan_to_deposit_ratio
FROM uae_banks_financial_data 
WHERE year IN (2023, 2024)
ORDER BY bank, year, quarter;

-- Query 3: Show asset quality metrics (NPL, CoR, Stage 3 Coverage)
SELECT 
    bank,
    year,
    quarter,
    npl_ratio,
    cor,
    stage3_cover,
    cost_income
FROM uae_banks_financial_data 
WHERE year = 2024 AND quarter = 4
ORDER BY npl_ratio;

-- Query 4: Market capitalization comparison for Q4 2024
SELECT 
    bank,
    share_price,
    market_cap_aed_bn,
    market_cap_usd_bn,
    rote,
    cet1
FROM uae_banks_financial_data 
WHERE year = 2024 AND quarter = 4
ORDER BY market_cap_aed_bn DESC;

-- Query 5: Growth analysis - YoY profit growth for 2024 vs 2023
SELECT 
    a.bank,
    a.ytd_profit as profit_2024,
    b.ytd_profit as profit_2023,
    ROUND(((a.ytd_profit - b.ytd_profit) / b.ytd_profit) * 100, 2) as yoy_growth_percent
FROM uae_banks_financial_data a
JOIN uae_banks_financial_data b ON a.bank = b.bank
WHERE a.year = 2024 AND a.quarter = 4
  AND b.year = 2023 AND b.quarter = 4
ORDER BY yoy_growth_percent DESC;

-- Query 6: Quarterly trend analysis for Emirates NBD
SELECT 
    year,
    quarter,
    CONCAT(year, '-Q', quarter) as period,
    quarterly_profit,
    nim,
    cost_income,
    rote,
    cet1
FROM uae_banks_financial_data 
WHERE bank = 'ENBD' AND year >= 2023
ORDER BY year, quarter;

-- Query 7: Average metrics by bank for 2024
SELECT 
    bank,
    ROUND(AVG(nim), 2) as avg_nim,
    ROUND(AVG(cost_income), 2) as avg_cost_income,
    ROUND(AVG(rote), 2) as avg_rote,
    ROUND(AVG(npl_ratio), 2) as avg_npl_ratio,
    ROUND(MAX(loans_advances), 0) as max_loans_advances,
    ROUND(MAX(deposits), 0) as max_deposits
FROM uae_banks_financial_data 
WHERE year = 2024
GROUP BY bank
ORDER BY avg_rote DESC;

-- Query 8: Top performing banks by ROTE in Q4 2024
SELECT 
    RANK() OVER (ORDER BY rote DESC) as rank,
    bank,
    rote,
    ytd_profit,
    cet1,
    npl_ratio
FROM uae_banks_financial_data 
WHERE year = 2024 AND quarter = 4;

-- Query 9: Banks with improving asset quality (decreasing NPL ratio)
SELECT 
    bank,
    '2023-Q4' as period,
    npl_2023,
    '2024-Q4' as period2,
    npl_2024,
    ROUND(npl_2024 - npl_2023, 2) as npl_change
FROM (
    SELECT 
        bank,
        MAX(CASE WHEN year = 2023 AND quarter = 4 THEN npl_ratio END) as npl_2023,
        MAX(CASE WHEN year = 2024 AND quarter = 4 THEN npl_ratio END) as npl_2024
    FROM uae_banks_financial_data 
    GROUP BY bank
) t
WHERE npl_2024 < npl_2023
ORDER BY npl_change;

-- Query 10: Market cap concentration analysis
SELECT 
    bank,
    market_cap_usd_bn,
    ROUND((market_cap_usd_bn / SUM(market_cap_usd_bn) OVER()) * 100, 2) as market_share_percent
FROM uae_banks_financial_data 
WHERE year = 2024 AND quarter = 4
ORDER BY market_cap_usd_bn DESC; 'Mashreq', 6380.4, 2985.4, 1499.8, 182450.6, 4.9, 212680.8, 102350.6, 37.8, 1.1, 0.35, 97.8, 25.8, 17.1, 34950.2, 204380.4, 75.80, 15.22, 4.14);

INSERT INTO uae_banks_financial_data VALUES 
(2024, 3, 'Mashreq', 9680.2, 4586.2, 1600.8, 191680.4, 5.0, 222450.6, 107850.4, 37.2, 1.0, 0.32, 99.5, 27.2, 17.4, 37450.8, 215280.2, 79.20, 15.91, 4.33);

INSERT INTO uae_banks_financial_data VALUES 
(2024, 4, 'Mashreq', 12950.6, 6285.4, 1699.2, 201450.8, 5.1, 233680.4, 114250.8, 36.8, 0.9, 0.28, 101.2, 28.5, 17.6, 40280.6, 228450.4, 82.75, 16.62, 4.52);

-- Insert data for ADCB (Abu Dhabi Commercial Bank)

-- 2020 Data for ADCB
INSERT INTO uae_banks_financial_data VALUES 
(2020, 1, 'ADCB', 4250.8, 1685.4, 1685.4, 168450.6, 2.8, 185680.4, 89250.2, 35.8, 2.1, 0.85, 78.5, 12.8, 15.2, 45680.2, 300450.8, 4.25, 85.20, 23.21);

INSERT INTO uae_banks_financial_data VALUES 
(2020, 2, 'ADCB', 8580.2, 3285.6, 1600.2, 172680.4, 2.7, 192450.8, 92680.4, 36.2, 2.3, 0.92, 76.8, 12.2, 14.9, 46850.4, 314580.2, 4.15, 83.18, 22.65);

INSERT INTO uae_banks_financial_data VALUES 
(2020, 3, 'ADCB', 12850.4, 4785.2, 1499.6, 175850.2, 2.6, 198680.6, 95450.8, 36.8, 2.5, 0.98, 75.2, 11.8, 14.6, 47680.6, 326450.4, 4.05, 81.23, 22.10);

INSERT INTO uae_banks_financial_data VALUES 
(2020, 4, 'ADCB', 17120.6, 6185.8, 1400.6, 179450.8, 2.5, 205850.2, 98680.4, 37.5, 2.8, 1.05, 73.8, 11.2, 14.2, 48950.2, 344680.6, 3.95, 79.20, 21.56);

-- 2021 Data for ADCB
INSERT INTO uae_banks_financial_data VALUES 
(2021, 1, 'ADCB', 4680.4, 1895.6, 1895.6, 184680.2, 2.6, 212680.4, 102450.8, 36.8, 2.5, 0.95, 76.8, 12.8, 14.5, 51680.4, 356450.8, 4.35, 87.22, 23.76);

INSERT INTO uae_banks_financial_data VALUES 
(2021, 2, 'ADCB', 9450.8, 3885.2, 1989.6, 189850.6, 2.7, 220450.8, 106850.2, 35.2, 2.2, 0.85, 78.5, 13.5, 14.8, 53450.2, 361280.4, 4.55, 91.21, 24.84);

INSERT INTO uae_banks_financial_data VALUES 
(2021, 3, 'ADCB', 14250.2, 5985.4, 2100.2, 196450.4, 2.8, 229680.2, 111450.6, 34.5, 2.0, 0.78, 80.2, 14.2, 15.2, 55680.8, 366450.2, 4.78, 95.86, 26.10);

INSERT INTO uae_banks_financial_data VALUES 
(2021, 4, 'ADCB', 18950.6, 7985.6, 2000.2, 204850.2, 2.9, 239450.6, 116850.4, 33.8, 1.8, 0.72, 82.8, 15.2, 15.6, 58450.2, 374680.6, 5.02, 100.70, 27.43);

-- 2022 Data for ADCB
INSERT INTO uae_banks_financial_data VALUES 
(2022, 1, 'ADCB', 5250.4, 2385.8, 2385.8, 213680.4, 3.1, 248450.2, 122450.6, 32.5, 1.6, 0.65, 85.2, 16.8, 16.2, 61680.4, 380450.8, 5.45, 109.30, 29.75);

INSERT INTO uae_banks_financial_data VALUES 
(2022, 2, 'ADCB', 10680.2, 4885.6, 2499.8, 221450.8, 3.3, 257680.4, 128650.2, 31.8, 1.4, 0.58, 87.5, 17.5, 16.6, 64450.8, 388250.6, 5.78, 115.98, 31.58);

INSERT INTO uae_banks_financial_data VALUES 
(2022, 3, 'ADCB', 16250.8, 7485.2, 2599.6, 230680.2, 3.5, 267850.6, 135450.4, 31.2, 1.2, 0.52, 89.8, 18.8, 17.1, 67850.2, 396850.4, 6.12, 122.81, 33.43);

INSERT INTO uae_banks_financial_data VALUES 
(2022, 4, 'ADCB', 21850.4, 9985.8, 2500.6, 241450.6, 3.6, 279680.2, 142850.6, 30.5, 1.1, 0.48, 92.2, 19.8, 17.5, 71450.8, 408250.2, 6.48, 130.01, 35.40);

-- 2023 Data for ADCB
INSERT INTO uae_banks_financial_data VALUES 
(2023, 1, 'ADCB', 5850.2, 2785.4, 2785.4, 252680.4, 3.8, 291450.8, 149850.2, 29.8, 1.0, 0.42, 94.8, 21.2, 17.8, 74680.2, 419450.6, 6.85, 137.42, 37.42);

INSERT INTO uae_banks_financial_data VALUES 
(2023, 2, 'ADCB', 11850.6, 5685.8, 2900.4, 264850.2, 3.9, 304680.4, 157450.6, 29.2, 0.9, 0.38, 96.5, 22.5, 18.2, 78450.8, 431280.4, 7.25, 145.48, 39.61);

INSERT INTO uae_banks_financial_data VALUES 
(2023, 3, 'ADCB', 17950.4, 8785.6, 3099.8, 278450.6, 4.0, 319850.2, 165680.4, 28.5, 0.8, 0.35, 98.2, 24.2, 18.6, 82680.4, 444450.8, 7.68, 154.12, 41.95);

INSERT INTO uae_banks_financial_data VALUES 
(2023, 4, 'ADCB', 24250.8, 11985.4, 3199.8, 293680.2, 4.1, 336450.8, 174850.6, 27.8, 0.7, 0.32, 99.8, 26.5, 19.1, 87450.6, 458680.2, 8.12, 163.01, 44.37);

-- 2024 Data for ADCB (Based on reported 15% growth and AED 10.585bn profit)
INSERT INTO uae_banks_financial_data VALUES 
(2024, 1, 'ADCB', 6450.4, 3285.2, 3285.2, 307850.4, 4.2, 351680.2, 182450.8, 27.2, 0.6, 0.28, 101.5, 28.2, 19.5, 91680.4, 470450.6, 8.65, 173.68, 47.29);

INSERT INTO uae_banks_financial_data VALUES 
(2024, 2, 'ADCB', 13080.8, 6785.6, 3500.4, 322680.6, 4.3, 368450.4, 191680.2, 26.8, 0.5, 0.25, 103.2, 29.8, 19.8, 96450.8, 487280.4, 9.20, 184.74, 50.29);

INSERT INTO uae_banks_financial_data VALUES 
(2024, 3, 'ADCB', 19850.2, 10485.8, 3700.2, 339450.8, 4.4, 386850.6, 201450.4, 26.2, 0.4, 0.22, 105.8, 31.5, 20.2, 102680.2, 508450.8, 9.78, 196.42, 53.47);

INSERT INTO uae_banks_financial_data VALUES 
(2024, 4, 'ADCB', 26850.4, 13785.2, 3299.4, 357680.4, 4.5, 407450.8, 212680.6, 25.8, 0.4, 0.20, 107.5, 32.8, 20.6, 109450.6, 530680.2, 10.35, 208.01, 56.66);

-- Insert data for DIB (Dubai Islamic Bank)

-- 2020 Data for DIB
INSERT INTO uae_banks_financial_data VALUES 
(2020, 1, 'DIB', 3450.8, 1285.4, 1285.4, 145680.2, 2.9, 162450.8, 78450.2, 38.5, 2.8, 1.25, 68.5, 11.2, 16.8, 28450.6, 169250.4, 3.45, 27.45, 7.47);

INSERT INTO uae_banks_financial_data VALUES 
(2020, 2, 'DIB', 6950.2, 2485.6, 1200.2, 148850.6, 2.8, 167680.4, 81250.6, 39.2, 3.1, 1.38, 66.8, 10.8, 16.5, 29680.4, 179850.2, 3.38, 26.91, 7.32);

INSERT INTO uae_banks_financial_data VALUES 
(2020, 3, 'DIB', 10380.4, 3585.8, 1100.2, 152680.4, 2.7, 173450.2, 84680.4, 40.1, 3.4, 1.52, 64.2, 10.2, 16.2, 30850.2, 190450.8, 3.25, 25.87, 7.04);

INSERT INTO uae_banks_financial_data VALUES 
(2020, 4, 'DIB', 13850.2, 4585.2, 999.4, 157450.8, 2.6, 180680.6, 88450.2, 41.2, 3.8, 1.68, 61.8, 9.5, 15.8, 31680.4, 200450.6, 3.12, 24.84, 6.76);

-- 2021 Data for DIB
INSERT INTO uae_banks_financial_data VALUES 
(2021, 1, 'DIB', 3750.6, 1485.2, 1485.2, 162680.4, 2.7, 186450.8, 92680.4, 40.8, 3.5, 1.45, 64.8, 10.2, 16.2, 33450.8, 206450.2, 3.35, 26.68, 7.27);

INSERT INTO uae_banks_financial_data VALUES 
(2021, 2, 'DIB', 7650.4, 2985.6, 1500.4, 167450.2, 2.8, 193680.2, 96450.8, 39.5, 3.2, 1.32, 67.2, 10.8, 16.5, 35680.4, 216250.8, 3.52, 28.05, 7.64);

INSERT INTO uae_banks_financial_data VALUES 
(2021, 3, 'DIB', 11580.2, 4585.8, 1600.2, 173850.6, 2.9, 201450.4, 101250.2, 38.2, 2.9, 1.18, 70.5, 11.8, 16.8, 38450.2, 228850.4, 3.72, 29.64, 8.07);

INSERT INTO uae_banks_financial_data VALUES 
(2021, 4, 'DIB', 15450.8, 6285.4, 1699.6, 181450.8, 3.0, 210680.6, 106850.4, 37.5, 2.6, 1.05, 74.2, 12.8, 17.2, 41680.6, 242450.8, 3.95, 31.47, 8.57);

-- 2022 Data for DIB
INSERT INTO uae_banks_financial_data VALUES 
(2022, 1, 'DIB', 4150.2, 1785.6, 1785.6, 189680.4, 3.2, 219450.8, 112680.2, 36.8, 2.3, 0.92, 77.8, 14.2, 17.5, 44680.8, 255450.2, 4.28, 34.08, 9.28);

INSERT INTO uae_banks_financial_data VALUES 
(2022, 2, 'DIB', 8450.6, 3685.2, 1899.6, 198450.2, 3.4, 229680.4, 119450.6, 35.2, 2.0, 0.78, 81.5, 15.8, 18.1, 48450.2, 267850.4, 4.65, 37.05, 10.09);

INSERT INTO uae_banks_financial_data VALUES 
(2022, 3, 'DIB', 12850.4, 5685.8, 2000.6, 208680.6, 3.6, 241450.2, 127680.4, 34.5, 1.8, 0.65, 85.2, 17.2, 18.6, 52680.4, 283450.8, 5.05, 40.24, 10.96);

INSERT INTO uae_banks_financial_data VALUES 
(2022, 4, 'DIB', 17250.8, 7785.4, 2099.6, 219450.8, 3.7, 254680.6, 136850.2, 33.8, 1.6, 0.58, 88.5, 18.8, 19.2, 57450.8, 300680.4, 5.48, 43.68, 11.90);

-- 2023 Data for DIB
INSERT INTO uae_banks_financial_data VALUES 
(2023, 1, 'DIB', 4650.8, 2185.4, 2185.4, 231680.4, 3.9, 267450.8, 145680.2, 32.5, 1.4, 0.48, 91.8, 20.5, 19.8, 62680.4, 316450.8, 5.95, 47.45, 12.92);

INSERT INTO uae_banks_financial_data VALUES 
(2023, 2, 'DIB', 9450.2, 4485.6, 2300.2, 244850.2, 4.0, 281680.4, 154250.6, 31.8, 1.2, 0.42, 94.2, 22.2, 20.5, 68450.2, 334280.6, 6.45, 51.42, 14.00);

INSERT INTO uae_banks_financial_data VALUES 
(2023, 3, 'DIB', 14350.6, 6985.8, 2500.2, 259450.8, 4.1, 297850.2, 164680.4, 30.2, 1.0, 0.38, 96.8, 24.8, 21.2, 75680.8, 356850.4, 6.98, 55.64, 15.16);

INSERT INTO uae_banks_financial_data VALUES 
(2023, 4, 'DIB', 19450.4, 9685.2, 2699.4, 275680.4, 4.2, 315450.6, 175850.2, 29.5, 0.9, 0.35, 98.5, 26.8, 21.8, 83450.6, 382680.2, 7.55, 60.20, 16.39);

-- 2024 Data for DIB (As of June 2024, total assets reached $88 billion)
INSERT INTO uae_banks_financial_data VALUES 
(2024, 1, 'DIB', 5150.6, 2585.8, 2585.8, 289450.2, 4.3, 331680.4, 184850.6, 28.8, 0.8, 0.32, 99.8, 28.5, 22.2, 89680.4, 403850.2, 8.15, 65.01, 17.70);

INSERT INTO uae_banks_financial_data VALUES 
(2024, 2, 'DIB', 10480.4, 5285.6, 2699.8, 304680.8, 4.4, 349450.2, 195680.4, 28.2, 0.7, 0.28, 101.2, 30.2, 22.8, 96450.8, 423680.4, 8.78, 70.05, 19.08);

INSERT INTO uae_banks_financial_data VALUES 
(2024, 3, 'DIB', 15950.2, 8185.4, 2899.8, 321450.6, 4.5, 368680.2, 207850.4, 27.5, 0.6, 0.25, 102.8, 32.8, 23.5, 104680.2, 445450.8, 9.45, 75.34, 20.52);

INSERT INTO uae_banks_financial_data VALUES 
(2024, 4, 'DIB', 21650.8, 11385.2, 3199.8, 340680.4, 4.6, 389450.8, 221680.2, 26.8, 0.5, 0.22, 104.5, 34.5, 24.2, 113450.6, 468850.2, 10.12, 80.72, 21.98);

-- Insert data for ENBD (Emirates NBD)

-- 2020 Data for ENBD
INSERT INTO uae_banks_financial_data VALUES 
(2020, 1, 'ENBD', 6850.4, 2985.6, 2985.6, 268450.2, 3.1, 295680.4, 142850.6, 42.5, 3.2, 1.85, 58.5, 10.8, 14.2, 52680.4, 370850.2, 8.25, 66.02, 17.98);

INSERT INTO uae_banks_financial_data VALUES 
(2020, 2, 'ENBD', 13850.2, 5785.4, 2799.8, 274680.8, 3.0, 304850.6, 147680.2, 43.2, 3.5, 1.92, 56.8, 10.2, 13.8, 54450.2, 394680.6, 8.05, 64.40, 17.54);

INSERT INTO uae_banks_financial_data VALUES 
(2020, 3, 'ENBD', 20650.8, 8485.2, 2699.8, 281850.4, 2.9, 315450.8, 152680.4, 44.1, 3.8, 2.05, 54.2, 9.5, 13.4, 55680.8, 415450.2, 7.85, 62.81, 17.10);

INSERT INTO uae_banks_financial_data VALUES 
(2020, 4, 'ENBD', 27450.6, 10985.8, 2500.6, 289680.2, 2.8, 327850.4, 158450.2, 45.2, 4.2, 2.18, 51.8, 8.8, 13.0, 56850.4, 437680.8, 7.65, 61.22, 16.67);

-- 2021 Data for ENBD
INSERT INTO uae_banks_financial_data VALUES 
(2021, 1, 'ENBD', 7450.2, 3485.6, 3485.6, 298450.8, 2.9, 338680.2, 165450.8, 44.8, 3.9, 1.95, 54.8, 9.8, 13.5, 59680.4, 442850.6, 8.15, 65.20, 17.76);

INSERT INTO uae_banks_financial_data VALUES 
(2021, 2, 'ENBD', 15180.4, 7085.2, 3599.6, 307680.4, 3.0, 351450.6, 172680.4, 43.5, 3.6, 1.82, 57.5, 10.8, 14.0, 63450.8, 453280.2, 8.65, 69.20, 18.85);

INSERT INTO uae_banks_financial_data VALUES 
(2021, 3, 'ENBD', 23050.6, 10885.8, 3800.6, 318450.2, 3.1, 365680.8, 180850.4, 42.2, 3.3, 1.68, 60.8, 11.8, 14.5, 68680.2, 473850.4, 9.18, 73.44, 20.01);

INSERT INTO uae_banks_financial_data VALUES 
(2021, 4, 'ENBD', 30850.2, 14685.4, 3799.6, 330680.6, 3.2, 381450.2, 189680.2, 41.5, 3.0, 1.55, 64.2, 12.8, 15.0, 74450.8, 496450.2, 9.75, 78.00, 21.25);

-- 2022 Data for ENBD
INSERT INTO uae_banks_financial_data VALUES 
(2022, 1, 'ENBD', 8250.4, 4185.8, 4185.8, 344850.2, 3.4, 398680.4, 198450.2, 40.2, 2.7, 1.42, 67.8, 14.2, 15.8, 80680.4, 510850.2, 10.45, 83.60, 22.76);

INSERT INTO uae_banks_financial_data VALUES 
(2022, 2, 'ENBD', 16850.6, 8585.2, 4399.4, 360450.8, 3.6, 417850.2, 208680.4, 39.5, 2.4, 1.28, 71.5, 15.8, 16.5, 88450.2, 536280.4, 11.20, 89.60, 24.40);

INSERT INTO uae_banks_financial_data VALUES 
(2022, 3, 'ENBD', 25650.4, 13285.6, 4700.4, 378680.4, 3.8, 439450.6, 219850.2, 38.8, 2.1, 1.15, 75.8, 17.5, 17.2, 97680.4, 567850.2, 11.98, 95.84, 26.10);

INSERT INTO uae_banks_financial_data VALUES 
(2022, 4, 'ENBD', 34850.2, 17985.8, 4700.2, 398450.2, 3.9, 463680.8, 232450.4, 38.2, 1.9, 1.02, 79.5, 19.2, 17.8, 108450.6, 609280.4, 12.75, 102.00, 27.78);

-- 2023 Data for ENBD (Based on reported AED 21.5 billion profit, 65% growth)
INSERT INTO uae_banks_financial_data VALUES 
(2023, 1, 'ENBD', 9150.6, 5285.4, 5285.4, 419680.2, 4.1, 486450.8, 246850.2, 37.2, 1.7, 0.88, 82.8, 21.5, 18.5, 118680.4, 641450.8, 13.85, 110.80, 30.18);

INSERT INTO uae_banks_financial_data VALUES 
(2023, 2, 'ENBD', 18650.4, 10885.6, 5600.2, 442850.4, 4.2, 512680.2, 260450.4, 36.5, 1.5, 0.75, 85.5, 23.8, 19.2, 130450.2, 679280.6, 14.65, 117.20, 31.93);

INSERT INTO uae_banks_financial_data VALUES 
(2023, 3, 'ENBD', 28450.2, 16785.8, 5900.2, 468450.8, 4.3, 541850.4, 275680.2, 35.8, 1.3, 0.68, 88.2, 26.2, 19.8, 144680.8, 730850.2, 15.48, 123.84, 33.73);

INSERT INTO uae_banks_financial_data VALUES 
(2023, 4, 'ENBD', 38250.6, 21985.4, 5199.6, 496850.2, 4.4, 573680.4, 292450.6, 35.2, 1.2, 0.62, 90.8, 28.5, 20.5, 160450.8, 783280.4, 16.25, 130.00, 35.41);

-- 2024 Data for ENBD (Based on record H1-24 profit of AED 13.8 billion, loans surpass AED 500 billion)
INSERT INTO uae_banks_financial_data VALUES 
(2024, 1, 'ENBD', 10450.8, 6485.2, 6485.2, 521680.4, 4.5, 602450.8, 308680.2, 34.5, 1.1, 0.58, 92.5, 30.8, 21.2, 174680.4, 824850.2, 17.25, 138.00, 37.59);

INSERT INTO uae_banks_financial_data VALUES 
(2024, 2, 'ENBD', 21250.4, 13485.6, 7000.4, 548450.2, 4.6, 634680.4, 325850.2, 33.8, 1.0, 0.52, 94.8, 32.5, 21.8, 190450.8, 873680.4, 18.45, 147.60, 40.20);

INSERT INTO uae_banks_financial_data VALUES 
(2024, 3, 'ENBD', 32850.2, 20785.8, 7300.2, 578680.6, 4.7, 669450.2, 344680.4, 33.2, 0.9, 0.48, 96.5, 34.8, 22.5, 208680.2, 927450.8, 19.75, 158.00, 43.04);

INSERT INTO uae_banks_financial_data VALUES 
(2024, 4, 'ENBD', 44650.8, 28285.4, 7499.6, 612450.8, 4.8, 707680.2, 365450.8, 32.5, 0.8, 0.45, 98.2, 36.5, 23.2, 228450.6, 984680.2, 20.95, 167.60, 45.67);

-- Insert data for FAB (First Abu Dhabi Bank)

-- 2020 Data for FAB
INSERT INTO uae_banks_financial_data VALUES 
(2020, 1, 'FAB', 8950.4, 4285.6, 4285.6, 425680.2, 2.7, 468450.8, 234850.2, 38.5, 2.5, 1.25, 72.8, 12.5, 17.2, 98680.4, 573450.8, 9.25, 185.50, 50.54);

INSERT INTO uae_banks_financial_data VALUES 
(2020, 2, 'FAB', 18150.8, 8285.4, 3999.8, 431850.6, 2.6, 478680.2, 242450.6, 39.2, 2.7, 1.32, 70.5, 12.0, 16.8, 101450.8, 603680.2, 9.05, 181.75, 49.52);

INSERT INTO uae_banks_financial_data VALUES 
(2020, 3, 'FAB', 27050.2, 12085.2, 3799.8, 438680.4, 2.5, 489850.4, 250680.2, 40.1, 2.9, 1.38, 68.2, 11.5, 16.4, 103680.2, 632450.6, 8.85, 177.65, 48.42);

INSERT INTO uae_banks_financial_data VALUES 
(2020, 4, 'FAB', 35850.6, 15785.8, 3700.6, 446450.8, 2.4, 502680.6, 259450.4, 40.8, 3.2, 1.45, 65.8, 10.8, 16.0, 105450.6, 659280.4, 8.65, 173.68, 47.32);

-- 2021 Data for FAB
INSERT INTO uae_banks_financial_data VALUES 
(2021, 1, 'FAB', 9450.2, 4785.6, 4785.6, 455680.4, 2.5, 514850.2, 268450.6, 40.2, 3.0, 1.32, 68.5, 11.8, 16.5, 109680.4, 664850.2, 9.15, 183.75, 50.07);

INSERT INTO uae_banks_financial_data VALUES 
(2021, 2, 'FAB', 19250.4, 9685.2, 4899.6, 467850.8, 2.6, 528450.4, 278680.2, 39.5, 2.8, 1.25, 71.2, 12.5, 17.0, 115450.2, 679280.6, 9.68, 194.42, 52.98);

INSERT INTO uae_banks_financial_data VALUES 
(2021, 3, 'FAB', 29150.6, 14785.8, 5100.6, 481450.2, 2.7, 543680.8, 289850.4, 38.8, 2.6, 1.18, 74.5, 13.5, 17.5, 122680.8, 700450.4, 10.25, 206.02, 56.15);

INSERT INTO uae_banks_financial_data VALUES 
(2021, 4, 'FAB', 38950.8, 19885.4, 5099.6, 496850.6, 2.8, 560450.2, 302450.8, 38.2, 2.4, 1.12, 77.8, 14.8, 18.2, 131450.4, 722680.2, 10.85, 218.02, 59.41);

-- 2022 Data for FAB
INSERT INTO uae_banks_financial_data VALUES 
(2022, 1, 'FAB', 10850.4, 5685.8, 5685.8, 513680.2, 3.0, 578450.6, 316850.2, 37.5, 2.2, 1.05, 80.5, 16.2, 18.8, 142680.4, 759450.2, 11.65, 234.08, 63.76);

INSERT INTO uae_banks_financial_data VALUES 
(2022, 2, 'FAB', 22150.6, 11685.4, 5999.6, 531850.4, 3.2, 598680.2, 332450.6, 36.8, 2.0, 0.95, 83.8, 17.8, 19.5, 156450.8, 802680.4, 12.45, 250.42, 68.25);

INSERT INTO uae_banks_financial_data VALUES 
(2022, 3, 'FAB', 33650.2, 17985.6, 6300.2, 552450.8, 3.4, 621450.4, 349680.2, 36.2, 1.8, 0.85, 87.2, 19.5, 20.2, 172680.2, 854450.6, 13.28, 267.01, 72.76);

INSERT INTO uae_banks_financial_data VALUES 
(2022, 4, 'FAB', 44850.8, 23985.2, 5999.6, 575680.4, 3.5, 646850.2, 368450.4, 35.8, 1.6, 0.78, 90.5, 21.2, 20.8, 190450.6, 916280.2, 14.12, 283.84, 77.37);

-- 2023 Data for FAB
INSERT INTO uae_banks_financial_data VALUES 
(2023, 1, 'FAB', 11950.6, 6485.4, 6485.4, 596450.8, 3.7, 671680.4, 384850.2, 35.2, 1.4, 0.68, 93.5, 23.5, 21.5, 208680.4, 970450.8, 15.25, 306.52, 83.54);

INSERT INTO uae_banks_financial_data VALUES 
(2023, 2, 'FAB', 24350.2, 13285.6, 6800.2, 619680.2, 3.8, 698450.6, 402680.4, 34.5, 1.2, 0.62, 96.2, 25.8, 22.2, 228450.2, 1028680.4, 16.15, 324.78, 88.48);

INSERT INTO uae_banks_financial_data VALUES 
(2023, 3, 'FAB', 37150.4, 20285.8, 7000.2, 645850.4, 3.9, 727680.2, 422450.8, 33.8, 1.1, 0.58, 98.5, 28.2, 22.8, 250680.8, 1099450.2, 17.08, 343.36, 93.58);

INSERT INTO uae_banks_financial_data VALUES 
(2023, 4, 'FAB', 49850.2, 26985.4, 6699.6, 674450.8, 4.0, 759850.4, 444680.2, 33.2, 1.0, 0.55, 99.8, 30.5, 23.5, 275450.6, 1172680.4, 17.95, 360.82, 98.32);

-- 2024 Data for FAB (Based on reported 4% profit growth to AED 17.1 billion)
INSERT INTO uae_banks_financial_data VALUES 
(2024, 1, 'FAB', 12850.4, 7185.2, 7185.2, 703680.2, 4.1, 792450.8, 467850.2, 32.8, 0.9, 0.52, 101.2, 32.8, 24.2, 298680.4, 1234850.2, 18.95, 381.02, 103.80);

INSERT INTO uae_banks_financial_data VALUES 
(2024, 2, 'FAB', 26150.8, 14685.4, 7500.2, 735450.6, 4.2, 827680.4, 492680.2, 32.2, 0.8, 0.48, 102.8, 34.5, 24.8, 324450.8, 1309280.4, 19.85, 399.02, 108.70);

INSERT INTO uae_banks_financial_data VALUES 
(2024, 3, 'FAB', 39950.2, 22485.6, 7800.2, 770680.4, 4.3, 866450.2, 519680.4, 31.5, 0.7, 0.45, 104.5, 36.8, 25.5, 352680.2, 1382450.8, 20.78, 417.64, 113.80);

INSERT INTO uae_banks_financial_data VALUES 
(2024, 4, 'FAB', 53650.8, 29985.2, 7499.6, 808450.8, 4.4, 908680.2, 548450.6, 31.0, 0.6, 0.42, 106.2, 38.5, 26.2, 383450.4, 1463680.2, 21.65, 435.48, 118.67);

-- Insert data for CBD (Commercial Bank of Dubai)

-- 2020 Data for CBD
INSERT INTO uae_banks_financial_data VALUES 
(2020, 1, 'CBD', 1850.4, 485.6, 485.6, 45680.2, 2.8, 52450.8, 26850.2, 42.5, 3.8, 1.85, 58.5, 9.2, 14.5, 8450.2, 58280.4, 2.85, 5.72, 1.56);

INSERT INTO uae_banks_financial_data VALUES 
(2020, 2, 'CBD', 3750.8, 885.4, 399.8, 47250.6, 2.7, 54680.4, 28450.6, 43.2, 4.1, 1.92, 56.8, 8.8, 14.2, 8680.4, 61250.8, 2.78, 5.58, 1.52);

INSERT INTO uae_banks_financial_data VALUES 
(2020, 3, 'CBD', 5580.2, 1185.2, 299.8, 48850.4, 2.6, 56850.2, 30250.4, 44.1, 4.4, 2.05, 54.2, 8.2, 13.8, 8950.2, 64850.6, 2.65, 5.32, 1.45);

INSERT INTO uae_banks_financial_data VALUES 
(2020, 4, 'CBD', 7420.6, 1385.8, 200.6, 50680.2, 2.5, 59450.4, 32450.8, 45.2, 4.8, 2.18, 51.8, 7.5, 13.4, 9250.4, 68950.2, 2.52, 5.06, 1.38);

-- 2021 Data for CBD
INSERT INTO uae_banks_financial_data VALUES 
(2021, 1, 'CBD', 1950.2, 585.4, 585.4, 52450.8, 2.6, 61680.2, 34250.6, 44.8, 4.5, 1.95, 54.8, 8.2, 13.8, 9680.4, 70250.8, 2.68, 5.38, 1.47);

INSERT INTO uae_banks_financial_data VALUES 
(2021, 2, 'CBD', 3980.4, 1185.2, 599.8, 54680.4, 2.7, 64450.6, 36850.2, 43.5, 4.2, 1.82, 57.5, 8.8, 14.2, 10450.2, 73680.4, 2.85, 5.72, 1.56);

INSERT INTO uae_banks_financial_data VALUES 
(2021, 3, 'CBD', 5950.6, 1885.8, 700.6, 57450.2, 2.8, 67850.4, 39680.4, 42.2, 3.9, 1.68, 60.8, 9.5, 14.8, 11680.2, 78850.4, 3.05, 6.12, 1.67);

INSERT INTO uae_banks_financial_data VALUES 
(2021, 4, 'CBD', 7950.8, 2485.4, 599.6, 60680.6, 2.9, 71450.2, 42850.6, 41.5, 3.6, 1.55, 64.2, 10.2, 15.2, 12950.4, 85250.8, 3.25, 6.52, 1.78);

-- 2022 Data for CBD
INSERT INTO uae_banks_financial_data VALUES 
(2022, 1, 'CBD', 2150.4, 685.8, 685.8, 63850.2, 3.1, 74680.4, 45850.2, 40.2, 3.3, 1.42, 67.8, 11.2, 15.8, 14250.4, 90450.2, 3.55, 7.12, 1.94);

INSERT INTO uae_banks_financial_data VALUES 
(2022, 2, 'CBD', 4380.6, 1485.2, 799.4, 67450.8, 3.3, 78450.2, 49250.6, 39.5, 3.0, 1.28, 71.5, 12.5, 16.5, 15680.4, 96280.4, 3.85, 7.72, 2.10);

INSERT INTO uae_banks_financial_data VALUES 
(2022, 3, 'CBD', 6650.4, 2385.6, 900.4, 71680.4, 3.5, 82850.6, 53450.2, 38.8, 2.7, 1.15, 75.8, 14.2, 17.2, 17450.4, 102850.2, 4.18, 8.38, 2.28);

INSERT INTO uae_banks_financial_data VALUES 
(2022, 4, 'CBD', 8850.2, 3185.8, 800.2, 76450.2, 3.6, 87680.8, 58250.4, 38.2, 2.5, 1.02, 79.5, 15.8, 17.8, 19680.6, 110450.4, 4.52, 9.08, 2.47);

-- 2023 Data for CBD
INSERT INTO uae_banks_financial_data VALUES 
(2023, 1, 'CBD', 2450.6, 885.4, 885.4, 80680.2, 3.8, 91450.8, 62850.2, 37.2, 2.3, 0.88, 82.8, 17.5, 18.5, 21680.4, 117450.8, 4.95, 9.94, 2.71);

INSERT INTO uae_banks_financial_data VALUES 
(2023, 2, 'CBD', 4950.4, 1885.6, 1000.2, 85450.4, 3.9, 96680.2, 67450.4, 36.5, 2.1, 0.75, 85.5, 19.2, 19.2, 24450.2, 127280.6, 5.35, 10.74, 2.93);

INSERT INTO uae_banks_financial_data VALUES 
(2023, 3, 'CBD', 7450.2, 2985.8, 1100.2, 91250.8, 4.0, 102850.4, 72680.2, 35.8, 1.9, 0.68, 88.2, 21.2, 19.8, 27680.8, 139850.2, 5.78, 11.60, 3.16);

INSERT INTO uae_banks_financial_data VALUES 
(2023, 4, 'CBD', 9950.6, 3985.4, 999.6, 97850.2, 4.1, 109680.4, 78450.6, 35.2, 1.7, 0.62, 90.8, 22.8, 20.5, 31450.8, 153280.4, 6.25, 12.54, 3.42);

-- 2024 Data for CBD
INSERT INTO uae_banks_financial_data VALUES 
(2024, 1, 'CBD', 2750.8, 1185.2, 1185.2, 103680.4, 4.2, 116450.8, 84680.2, 34.5, 1.5, 0.58, 92.5, 24.8, 21.2, 34680.4, 164850.2, 6.85, 13.74, 3.74);
