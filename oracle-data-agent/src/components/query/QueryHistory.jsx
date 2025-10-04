// components/query/QueryHistory.jsx
import React from 'react';
import { History } from 'lucide-react';

const QueryHistory = ({ queryHistory, onSelectQuery }) => {
  return (
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
              onClick={() => onSelectQuery(item.natural)}
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
  );
};

export default QueryHistory;