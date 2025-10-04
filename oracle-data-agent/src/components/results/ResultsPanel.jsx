// components/results/ResultsPanel.jsx
import React, { useState, useEffect } from 'react';
import { Table } from 'lucide-react';
import ResultsHeader from './ResultsHeader';
import DataTable from './DataTable';
import Pagination from './Pagination';
import LoadingSpinner from '../common/LoadingSpinner';
import ErrorDisplay from '../common/ErrorDisplay';
import { 
  filterRows, 
  paginateRows, 
  calculateTotalPages, 
  exportToCSV 
} from '../../utils/helpers';
import { ROWS_PER_PAGE } from '../../utils/constants';

const ResultsPanel = ({ result, loading, error }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);

  // Reset pagination when result changes
  useEffect(() => {
    setCurrentPage(1);
    setSearchTerm('');
  }, [result]);

  // Reset pagination when search term changes
  const handleSearchChange = (term) => {
    setSearchTerm(term);
    setCurrentPage(1);
  };

  // Calculate filtered and paginated data
  const filteredData = result?.rows ? filterRows(result.rows, searchTerm) : [];
  const paginatedData = paginateRows(filteredData, currentPage, ROWS_PER_PAGE);
  const totalPages = calculateTotalPages(filteredData.length, ROWS_PER_PAGE);

  const handleExport = () => {
    if (result?.columns && result?.rows) {
      exportToCSV(result.columns, result.rows);
    }
  };

  // Empty state
  if (!result && !loading && !error) {
    return (
      <div className="flex-1 flex flex-col bg-white dark:bg-gray-800 overflow-hidden">
        <ResultsHeader 
          result={null}
          searchTerm=""
          setSearchTerm={() => {}}
          filteredCount={0}
          onExport={() => {}}
        />
        <div className="flex-1 flex items-center justify-center text-gray-400 dark:text-gray-500">
          <div className="text-center">
            <Table className="w-12 h-12 mx-auto mb-2" />
            <p className="text-lg font-medium">No Results Yet</p>
            <p className="text-sm mt-1">Execute a query to see results here</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col bg-white dark:bg-gray-800 overflow-hidden">
      <ResultsHeader 
        result={result}
        searchTerm={searchTerm}
        setSearchTerm={handleSearchChange}
        filteredCount={filteredData.length}
        onExport={handleExport}
      />

      <div className="flex-1 flex flex-col overflow-hidden">
        {loading && <LoadingSpinner message="Executing query..." />}
        
        {error && !loading && <ErrorDisplay error={error} title="Execution Error" />}

        {result && !loading && !error && (
          <>
            {result.columns && result.rows ? (
              <div className="flex-1 flex flex-col overflow-hidden">
                <DataTable 
                  columns={result.columns}
                  rows={paginatedData}
                  currentPage={currentPage}
                  rowsPerPage={ROWS_PER_PAGE}
                />
                
                {totalPages > 1 && (
                  <Pagination 
                    currentPage={currentPage}
                    totalPages={totalPages}
                    totalRows={filteredData.length}
                    rowsPerPage={ROWS_PER_PAGE}
                    onPageChange={setCurrentPage}
                  />
                )}
              </div>
            ) : (
              <div className="flex-1 overflow-auto p-4">
                <pre className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg text-sm overflow-auto">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default ResultsPanel;