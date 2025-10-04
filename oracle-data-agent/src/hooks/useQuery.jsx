// hooks/useQuery.js
import { useState } from 'react';
import apiService from '../services/api';

export const useQuery = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [sqlQuery, setSqlQuery] = useState('');
  const [queryHistory, setQueryHistory] = useState([]);

  const executeQuery = async (query) => {
    if (!query.trim()) {
      setError('Please enter a query');
      return;
    }

    console.log('Executing query:', query);
    setLoading(true);
    setError(null);
    setResult(null);
    setSqlQuery('');

    try {
      const data = await apiService.executeQuery(query);
      console.log('Response data:', data);

      setSqlQuery(data.sql_query || '');
      
      if (data.error) {
        setError(data.error);
      } else if (data.result) {
        setResult(data.result);
        
        // Add to history
        setQueryHistory(prev => [{
          natural: data.natural_query,
          sql: data.sql_query,
          timestamp: new Date().toLocaleTimeString(),
          rowCount: data.result?.row_count || data.result?.rows?.length || 0
        }, ...prev.slice(0, 9)]);
      }
    } catch (err) {
      console.error('Query execution error:', err);
      setError(`Failed to execute query: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const clearError = () => setError(null);
  const clearResults = () => {
    setResult(null);
    setSqlQuery('');
  };

  return {
    loading,
    result,
    error,
    sqlQuery,
    queryHistory,
    executeQuery,
    clearError,
    clearResults
  };
};