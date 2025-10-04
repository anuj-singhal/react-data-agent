// components/query/QueryInput.jsx
import React from 'react';
import { Search, Play, Loader2 } from 'lucide-react';
import { SAMPLE_QUERIES } from '../../utils/constants';

const QueryInput = ({ query, setQuery, onExecute, loading }) => {
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      e.preventDefault();
      onExecute();
    }
  };

  return (
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
          onClick={onExecute}
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
          {SAMPLE_QUERIES.map((sample, idx) => (
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
  );
};

export default QueryInput;