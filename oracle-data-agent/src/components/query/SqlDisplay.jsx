// components/query/SqlDisplay.jsx
import React, { useState } from 'react';
import { Code, Copy, CheckCircle, Maximize2, Minimize2 } from 'lucide-react';
import { formatSQL } from '../../utils/helpers';

const SqlDisplay = ({ sqlQuery }) => {
  const [copySuccess, setCopySuccess] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    });
  };

  const formattedSQL = formatSQL(sqlQuery);

  return (
    <div className="p-4 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Code className="w-4 h-4 text-purple-600" />
          <h3 className="font-semibold text-gray-800 dark:text-white text-sm">Generated SQL</h3>
        </div>
        {sqlQuery && (
          <div className="flex items-center gap-1">
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="p-1.5 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title={isExpanded ? "Minimize" : "Expand"}
            >
              {isExpanded ? (
                <Minimize2 className="w-4 h-4 text-gray-600" />
              ) : (
                <Maximize2 className="w-4 h-4 text-gray-600" />
              )}
            </button>
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
          </div>
        )}
      </div>
      
      <div className={`${isExpanded ? 'max-h-96' : 'max-h-48'} overflow-auto transition-all duration-300`}>
        {sqlQuery ? (
          <pre className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg text-xs overflow-x-auto">
            <code className="text-gray-800 dark:text-gray-200 font-mono whitespace-pre">
              {formattedSQL}
            </code>
          </pre>
        ) : (
          <div className="text-gray-400 dark:text-gray-500 text-sm italic">
            SQL query will appear here after execution
          </div>
        )}
      </div>
    </div>
  );
};

export default SqlDisplay;