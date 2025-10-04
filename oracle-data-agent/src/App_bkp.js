import React, { useState, useEffect } from 'react';
import { AlertCircle, Database, Loader2, CheckCircle, Code, Table, RefreshCw, Download, Search, Play, History, Copy, ChevronLeft, ChevronRight } from 'lucide-react';

const API_URL = 'http://localhost:8000';

export default function OracleDataAgent() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [sqlQuery, setSqlQuery] = useState('');
  const [healthStatus, setHealthStatus] = useState(null);
  const [queryHistory, setQueryHistory] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [rowsPerPage] = useState(20);
  const [copySuccess, setCopySuccess] = useState(false);
  const [leftPanelCollapsed, setLeftPanelCollapsed] = useState(false);

  // Check health status on mount
  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      const response = await fetch(`${API_URL}/health`);
      const data = await response.json();
      setHealthStatus(data);
    } catch (err) {
      setHealthStatus({ status: 'error', mcp_connected: false, available_tools: [] });
    }
  };

  const executeQuery = async () => {
    if (!query.trim()) {
      setError('Please enter a query');
      return;
    }

    console.log('Executing query:', query);
    setLoading(true);
    setError(null);
    setResult(null);
    setSqlQuery('');
    setCurrentPage(1);
    setSearchTerm('');

    try {
      const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query.trim() }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
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

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      e.preventDefault();
      executeQuery();
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    });
  };

  const exportToCSV = () => {
    if (!result || !result.columns || !result.rows) return;

    const csvContent = [
      result.columns.join(','),
      ...result.rows.map(row => 
        row.map(cell => 
          typeof cell === 'string' && cell.includes(',') 
            ? `"${cell}"` 
            : cell ?? ''
        ).join(',')
      )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `query_result_${new Date().toISOString()}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  const filteredRows = () => {
    if (!result || !result.rows || !searchTerm) return result?.rows || [];
    
    return result.rows.filter(row =>
      row.some(cell =>
        String(cell ?? '').toLowerCase().includes(searchTerm.toLowerCase())
      )
    );
  };

  const paginatedRows = () => {
    const filtered = filteredRows();
    const start = (currentPage - 1) * rowsPerPage;
    const end = start + rowsPerPage;
    return filtered.slice(start, end);
  };

  const totalPages = () => {
    const filtered = filteredRows();
    return Math.ceil(filtered.length / rowsPerPage);
  };

  // Sample queries for quick access
  const sampleQueries = [
    "Show all tables",
    "SELECT * FROM uae_banks_financial_data WHERE ROWNUM <= 10",
    "Get top 5 banks by profit",
    "Show banks with deposits > 10000",
    "Test connection"
  ];

  return (
    <div className="h-screen flex flex-col bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 py-3 shadow-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Database className="w-6 h-6 text-blue-600" />
            <h1 className="text-xl font-bold text-gray-800 dark:text-white">Oracle Data Agent</h1>
          </div>
          
          <div className="flex items-center gap-4">
            <button
              onClick={checkHealth}
              className="p-1.5 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Refresh connection status"
            >
              <RefreshCw className="w-4 h-4 text-gray-600 dark:text-gray-400" />
            </button>
            <div className="flex items-center gap-2">
              {healthStatus?.mcp_connected ? (
                <>
                  <CheckCircle className="w-4 h-4 text-green-500" />
                  <span className="text-sm text-green-600 dark:text-green-400">Connected</span>
                </>
              ) : (
                <>
                  <AlertCircle className="w-4 h-4 text-red-500" />
                  <span className="text-sm text-red-600 dark:text-red-400">Disconnected</span>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content - Split Panel */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* LEFT PANEL - Query Interface */}
        <div className={`${leftPanelCollapsed ? 'w-12' : 'w-[450px]'} transition-all duration-300 flex flex-col bg-gray-50 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 relative`}>
          
          {/* Collapse Toggle Button */}
          <button
            onClick={() => setLeftPanelCollapsed(!leftPanelCollapsed)}
            className="absolute -right-3 top-1/2 transform -translate-y-1/2 z-10 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-full p-1 shadow-md hover:bg-gray-50 dark:hover:bg-gray-700"
            title={leftPanelCollapsed ? "Expand panel" : "Collapse panel"}
          >
            {leftPanelCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>

          {!leftPanelCollapsed && (
            <>
              {/* Natural Language Query Section */}
              <div className="p-4 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-2 mb-3">
                  <Search className="w-4 h-4 text-gray-600" />
                  <h3 className="font-semibold text-gray-800 dark:text-white">Natural Language Query</h3>
                </div>
                
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={handleKeyPress}
                  placeholder="Enter your query in natural language or SQL..."
                  className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg 
                            bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                            focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none
                            text-sm"
                  rows={4}
                  disabled={loading}
                />
                
                <div className="flex items-center justify-between mt-3">
                  <span className="text-xs text-gray-500">Press Ctrl+Enter to execute</span>
                  <button
                    onClick={executeQuery}
                    disabled={loading || !query.trim()}
                    className={`px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2
                      ${loading || !query.trim() 
                        ? 'bg-gray-400 cursor-not-allowed' 
                        : 'bg-blue-600 hover:bg-blue-700 active:scale-95'} 
                      text-white text-sm`}
                  >
                    {loading ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Executing
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        Execute Query
                      </>
                    )}
                  </button>
                </div>

                {/* Sample Queries */}
                <div className="mt-3">
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">Quick queries:</p>
                  <div className="flex flex-wrap gap-1">
                    {sampleQueries.map((sample, idx) => (
                      <button
                        key={idx}
                        onClick={() => setQuery(sample)}
                        className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300
                                  hover:bg-gray-200 dark:hover:bg-gray-600 rounded transition-colors"
                        title={sample}
                      >
                        {sample.length > 35 ? sample.substring(0, 35) + '...' : sample}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Generated SQL Section */}
              <div className="p-4 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Code className="w-4 h-4 text-purple-600" />
                    <h3 className="font-semibold text-gray-800 dark:text-white text-sm">Generated SQL</h3>
                  </div>
                  {sqlQuery && (
                    <button
                      onClick={() => copyToClipboard(sqlQuery)}
                      className="p-1.5 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                      title="Copy SQL"
                    >
                      {copySuccess ? (
                        <CheckCircle className="w-4 h-4 text-green-600" />
                      ) : (
                        <Copy className="w-4 h-4 text-gray-600" />
                      )}
                    </button>
                  )}
                </div>
                
                <div className="max-h-32 overflow-auto">
                  {sqlQuery ? (
                    <pre className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg text-xs">
                      <code className="text-gray-800 dark:text-gray-200 font-mono">{sqlQuery}</code>
                    </pre>
                  ) : (
                    <div className="text-gray-400 dark:text-gray-500 text-sm italic">
                      SQL query will appear here after execution
                    </div>
                  )}
                </div>
              </div>

              {/* Query History Section */}
              <div className="flex-1 p-4 bg-white dark:bg-gray-800 overflow-y-auto">
                <div className="flex items-center gap-2 mb-3">
                  <History className="w-4 h-4 text-gray-600" />
                  <h3 className="font-semibold text-gray-800 dark:text-white text-sm">Query History</h3>
                </div>
                
                {queryHistory.length > 0 ? (
                  <div className="space-y-2">
                    {queryHistory.map((item, idx) => (
                      <div
                        key={idx}
                        onClick={() => setQuery(item.natural)}
                        className="p-3 bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 
                                  rounded-lg cursor-pointer transition-colors border border-gray-200 dark:border-gray-600"
                      >
                        <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
                          <span>{item.timestamp}</span>
                          <span className="font-medium">{item.rowCount} rows</span>
                        </div>
                        <div className="text-xs text-gray-800 dark:text-gray-200 font-medium mb-1">
                          {item.natural.length > 60 ? item.natural.substring(0, 60) + '...' : item.natural}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-500 font-mono">
                          {item.sql.length > 60 ? item.sql.substring(0, 60) + '...' : item.sql}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-gray-400 dark:text-gray-500 text-sm italic text-center mt-8">
                    Query history will appear here
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        {/* RIGHT PANEL - Data Results */}
        <div className="flex-1 flex flex-col bg-white dark:bg-gray-800">
          {/* Results Header */}
          <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Table className="w-4 h-4 text-gray-600" />
                <h2 className="font-semibold text-gray-800 dark:text-white">Query Results</h2>
                {result?.row_count !== undefined && (
                  <span className="px-2 py-0.5 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 text-xs rounded-full">
                    {result.row_count} total rows
                  </span>
                )}
                {result?.rows && filteredRows().length !== result.rows.length && (
                  <span className="px-2 py-0.5 bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400 text-xs rounded-full">
                    {filteredRows().length} filtered
                  </span>
                )}
              </div>
              
              {result && result.columns && (
                <div className="flex items-center gap-2">
                  <div className="relative">
                    <Search className="w-4 h-4 absolute left-2 top-1/2 transform -translate-y-1/2 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search in results..."
                      value={searchTerm}
                      onChange={(e) => {
                        setSearchTerm(e.target.value);
                        setCurrentPage(1);
                      }}
                      className="pl-8 pr-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded 
                                bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                                focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                  <button
                    onClick={exportToCSV}
                    className="px-3 py-1.5 text-sm bg-green-600 hover:bg-green-700 text-white rounded transition-colors flex items-center gap-1"
                    title="Export to CSV"
                  >
                    <Download className="w-4 h-4" />
                    Export
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Results Content with Scroll */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {loading && (
              <div className="flex-1 flex items-center justify-center">
                <div className="text-center">
                  <Loader2 className="w-8 h-8 animate-spin text-blue-600 mx-auto mb-2" />
                  <p className="text-gray-600 dark:text-gray-400">Executing query...</p>
                </div>
              </div>
            )}

            {error && !loading && (
              <div className="m-4">
                <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
                  <div className="flex items-start gap-2">
                    <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <h3 className="font-semibold text-red-800 dark:text-red-300">Execution Error</h3>
                      <p className="text-sm text-red-700 dark:text-red-400 mt-1">{error}</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {result && !loading && !error && (
              <>
                {result.columns && result.rows ? (
                  <div className="flex-1 flex flex-col overflow-hidden">
                    {/* Scrollable Table Container */}
                    <div className="flex-1 overflow-auto p-4">
                      <div className="inline-block min-w-full align-middle">
                        <div className="overflow-hidden border border-gray-200 dark:border-gray-700 rounded-lg">
                          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                            <thead className="bg-gray-50 dark:bg-gray-700 sticky top-0 z-10">
                              <tr>
                                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider bg-gray-50 dark:bg-gray-700">
                                  #
                                </th>
                                {result.columns.map((col, idx) => (
                                  <th 
                                    key={idx} 
                                    className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider bg-gray-50 dark:bg-gray-700 whitespace-nowrap"
                                  >
                                    {col}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                              {paginatedRows().length > 0 ? (
                                paginatedRows().map((row, rowIdx) => (
                                  <tr key={rowIdx} className="hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
                                    <td className="px-3 py-2 text-sm text-gray-500 dark:text-gray-400 font-medium">
                                      {(currentPage - 1) * rowsPerPage + rowIdx + 1}
                                    </td>
                                    {row.map((cell, cellIdx) => (
                                      <td 
                                        key={cellIdx} 
                                        className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100 whitespace-nowrap"
                                      >
                                        {cell !== null && cell !== undefined ? (
                                          typeof cell === 'number' ? (
                                            <span className="font-mono">
                                              {Number.isInteger(cell) ? cell : parseFloat(cell).toFixed(2)}
                                            </span>
                                          ) : (
                                            <span title={String(cell)}>
                                              {String(cell).length > 50 
                                                ? String(cell).substring(0, 50) + '...' 
                                                : String(cell)}
                                            </span>
                                          )
                                        ) : (
                                          <span className="text-gray-400 italic">NULL</span>
                                        )}
                                      </td>
                                    ))}
                                  </tr>
                                ))
                              ) : (
                                <tr>
                                  <td 
                                    colSpan={result.columns.length + 1} 
                                    className="px-3 py-8 text-center text-gray-500 dark:text-gray-400"
                                  >
                                    No matching results found
                                  </td>
                                </tr>
                              )}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </div>

                    {/* Pagination - Fixed at Bottom */}
                    {totalPages() > 1 && (
                      <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                        <div className="flex items-center justify-between">
                          <div className="text-sm text-gray-700 dark:text-gray-300">
                            Showing {((currentPage - 1) * rowsPerPage) + 1} to {Math.min(currentPage * rowsPerPage, filteredRows().length)} of {filteredRows().length} results
                          </div>
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => setCurrentPage(1)}
                              disabled={currentPage === 1}
                              className={`px-3 py-1 text-sm rounded ${
                                currentPage === 1
                                  ? 'bg-gray-200 dark:bg-gray-700 text-gray-400 cursor-not-allowed'
                                  : 'bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600'
                              }`}
                            >
                              First
                            </button>
                            <button
                              onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                              disabled={currentPage === 1}
                              className={`px-3 py-1 text-sm rounded ${
                                currentPage === 1
                                  ? 'bg-gray-200 dark:bg-gray-700 text-gray-400 cursor-not-allowed'
                                  : 'bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600'
                              }`}
                            >
                              Previous
                            </button>
                            
                            <span className="px-3 py-1 text-sm text-gray-700 dark:text-gray-300">
                              Page {currentPage} of {totalPages()}
                            </span>
                            
                            <button
                              onClick={() => setCurrentPage(prev => Math.min(totalPages(), prev + 1))}
                              disabled={currentPage === totalPages()}
                              className={`px-3 py-1 text-sm rounded ${
                                currentPage === totalPages()
                                  ? 'bg-gray-200 dark:bg-gray-700 text-gray-400 cursor-not-allowed'
                                  : 'bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600'
                              }`}
                            >
                              Next
                            </button>
                            <button
                              onClick={() => setCurrentPage(totalPages())}
                              disabled={currentPage === totalPages()}
                              className={`px-3 py-1 text-sm rounded ${
                                currentPage === totalPages()
                                  ? 'bg-gray-200 dark:bg-gray-700 text-gray-400 cursor-not-allowed'
                                  : 'bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600'
                              }`}
                            >
                              Last
                            </button>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex-1 overflow-auto p-4">
                    <pre className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg text-sm">
                      {JSON.stringify(result, null, 2)}
                    </pre>
                  </div>
                )}
              </>
            )}

            {!result && !loading && !error && (
              <div className="flex-1 flex items-center justify-center text-gray-400 dark:text-gray-500">
                <div className="text-center">
                  <Table className="w-12 h-12 mx-auto mb-2" />
                  <p className="text-lg font-medium">No Results Yet</p>
                  <p className="text-sm mt-1">Execute a query to see results here</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}