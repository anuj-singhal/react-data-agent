// utils/constants.js
export const API_URL = 'http://localhost:8000';

export const ROWS_PER_PAGE = 20;

export const SAMPLE_QUERIES = [
  "Show all tables",
  "SELECT * FROM uae_banks_financial_data WHERE ROWNUM <= 10",
  "Get top 5 banks by profit",
  "Show banks with deposits > 10000",
  "Test connection"
];

export const HEALTH_CHECK_INTERVAL = 30000; // 30 seconds