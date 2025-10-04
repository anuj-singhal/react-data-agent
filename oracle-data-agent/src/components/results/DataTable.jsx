// components/results/DataTable.jsx
import React from 'react';
import { formatCellValue } from '../../utils/helpers';

const DataTable = ({ columns, rows, currentPage, rowsPerPage }) => {
  // Calculate minimum column width based on column name length
  const getColumnWidth = (columnName) => {
    const minWidth = 100; // Minimum width in pixels
    const charWidth = 8; // Approximate width per character
    const padding = 20; // Extra padding
    const calculatedWidth = columnName.length * charWidth + padding;
    return Math.max(minWidth, calculatedWidth);
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-white dark:bg-gray-800">
      {/* Fixed height scrollable container */}
      <div className="flex-1 relative overflow-auto border border-gray-200 dark:border-gray-700 rounded-lg m-4">
        {/* Scrollable table wrapper - both horizontal and vertical */}
        <div className="min-w-full inline-block align-middle">
          <table className="border-collapse">
            <thead className="sticky top-0 z-20 bg-gray-50 dark:bg-gray-700">
              <tr>
                <th 
                  className="sticky left-0 z-30 px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider bg-gray-50 dark:bg-gray-700 border-b border-r border-gray-200 dark:border-gray-600"
                  style={{ minWidth: '60px', width: '60px' }}
                >
                  #
                </th>
                {columns.map((col, idx) => (
                  <th 
                    key={idx} 
                    className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider bg-gray-50 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600"
                    style={{ 
                      minWidth: `${getColumnWidth(col)}px`,
                      width: `${getColumnWidth(col)}px`
                    }}
                  >
                    <div className="truncate" title={col}>
                      {col}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800">
              {rows.length > 0 ? (
                rows.map((row, rowIdx) => (
                  <tr 
                    key={rowIdx} 
                    className="hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors border-b border-gray-100 dark:border-gray-700"
                  >
                    <td 
                      className="sticky left-0 z-10 px-3 py-2 text-sm text-gray-500 dark:text-gray-400 font-medium bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-600"
                      style={{ minWidth: '60px', width: '60px' }}
                    >
                      {(currentPage - 1) * rowsPerPage + rowIdx + 1}
                    </td>
                    {row.map((cell, cellIdx) => {
                      const formatted = formatCellValue(cell);
                      const columnWidth = getColumnWidth(columns[cellIdx]);
                      return (
                        <td 
                          key={cellIdx} 
                          className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100"
                          style={{ 
                            minWidth: `${columnWidth}px`,
                            width: `${columnWidth}px`,
                            maxWidth: `${columnWidth}px`
                          }}
                        >
                          <div 
                            className="truncate"
                            title={formatted.fullText || formatted.text}
                          >
                            {formatted.isNull ? (
                              <span className="text-gray-400 italic">NULL</span>
                            ) : formatted.isNumber ? (
                              <span className="font-mono">{formatted.text}</span>
                            ) : (
                              <span>{formatted.text}</span>
                            )}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))
              ) : (
                <tr>
                  <td 
                    colSpan={columns.length + 1} 
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
      
      {/* Scroll indicators */}
      {rows.length > 0 && (
        <div className="px-4 pb-2 text-xs text-gray-500 dark:text-gray-400 text-center">
          Tip: Use arrow keys or scroll to navigate large tables
        </div>
      )}
    </div>
  );
};

export default DataTable;