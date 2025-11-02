// components/results/ResultsHeader.jsx (Modified to add Chart button)
import React from 'react';
import { Table, Search, Download, Columns, Rows, BarChart3 } from 'lucide-react';

const ResultsHeader = ({ 
  result, 
  searchTerm, 
  setSearchTerm, 
  filteredCount, 
  onExport,
  onShowChart 
}) => {
  return (
    <div className="flex-shrink-0 px-4 py-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 flex-wrap">
          <Table className="w-4 h-4 text-gray-600" />
          <h2 className="font-semibold text-gray-800 dark:text-white">Query Results</h2>
          {result?.columns && (
            <span className="px-2 py-0.5 bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 text-xs rounded-full flex items-center gap-1">
              <Columns className="w-3 h-3" />
              {result.columns.length} cols
            </span>
          )}
          {result?.row_count !== undefined && (
            <span className="px-2 py-0.5 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 text-xs rounded-full flex items-center gap-1">
              <Rows className="w-3 h-3" />
              {result.row_count} rows
            </span>
          )}
          {result?.rows && searchTerm && filteredCount !== result.rows.length && (
            <span className="px-2 py-0.5 bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400 text-xs rounded-full">
              {filteredCount} filtered
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
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-8 pr-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded 
                          bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                          focus:ring-2 focus:ring-blue-500 focus:border-transparent
                          focus:outline-none"
              />
            </div>
            
            {/* Chart Button */}
            <button
              onClick={onShowChart}
              className="px-3 py-1.5 text-sm bg-purple-600 hover:bg-purple-700 text-white rounded 
                        transition-colors flex items-center gap-1 active:scale-95"
              title="Visualize Data"
            >
              <BarChart3 className="w-4 h-4" />
              Chart
            </button>
            
            <button
              onClick={onExport}
              className="px-3 py-1.5 text-sm bg-green-600 hover:bg-green-700 text-white rounded 
                        transition-colors flex items-center gap-1 active:scale-95"
              title="Export to CSV"
            >
              <Download className="w-4 h-4" />
              Export
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultsHeader;