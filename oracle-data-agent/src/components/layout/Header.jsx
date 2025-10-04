// components/layout/Header.jsx
import React from 'react';
import { Database, RefreshCw, CheckCircle, AlertCircle } from 'lucide-react';

const Header = ({ healthStatus, onRefreshHealth }) => {
  return (
    <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 py-3 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Database className="w-6 h-6 text-blue-600" />
          <h1 className="text-xl font-bold text-gray-800 dark:text-white">Oracle Data Agent</h1>
        </div>
        
        <div className="flex items-center gap-4">
          <button
            onClick={onRefreshHealth}
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
  );
};

export default Header;