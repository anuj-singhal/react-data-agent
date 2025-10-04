// App.js
import React, { useState } from 'react';
import Header from './components/layout/Header';
import QueryPanel from './components/query/QueryPanel';
import ResultsPanel from './components/results/ResultsPanel';
import { useHealthCheck } from './hooks/useHealthCheck';

function App() {
  const [leftPanelCollapsed, setLeftPanelCollapsed] = useState(false);
  const [leftPanelWidth, setLeftPanelWidth] = useState(500); // Increased default width for chat
  const [queryHistory, setQueryHistory] = useState([]);
  const [currentResult, setCurrentResult] = useState(null);
  const [currentSqlQuery, setCurrentSqlQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const { healthStatus, checkHealth } = useHealthCheck();

  const handleQueryExecute = (sql, result) => {
    setCurrentSqlQuery(sql);
    setCurrentResult(result);
    setError(null);
    
    // Add to history
    if (result && !result.error) {
      const historyEntry = {
        natural: sql, // You might want to track the natural language query separately
        sql: sql,
        timestamp: new Date().toLocaleTimeString(),
        rowCount: result?.row_count || result?.rows?.length || 0
      };
      
      setQueryHistory(prev => [historyEntry, ...prev.slice(0, 9)]);
    } else if (result?.error) {
      setError(result.error);
    }
  };

  const handlePanelWidthChange = (newWidth) => {
    setLeftPanelWidth(newWidth);
  };

  return (
    <div className="h-screen flex flex-col bg-gray-50 dark:bg-gray-900">
      <Header 
        healthStatus={healthStatus} 
        onRefreshHealth={checkHealth} 
      />
      
      <div className="flex-1 flex overflow-hidden">
        <QueryPanel 
          sqlQuery={currentSqlQuery}
          queryHistory={queryHistory}
          onQueryExecute={handleQueryExecute}
          collapsed={leftPanelCollapsed}
          setCollapsed={setLeftPanelCollapsed}
          width={leftPanelWidth}
          onWidthChange={handlePanelWidthChange}
        />
        
        <ResultsPanel 
          result={currentResult}
          loading={loading}
          error={error}
        />
      </div>
    </div>
  );
}

export default App;