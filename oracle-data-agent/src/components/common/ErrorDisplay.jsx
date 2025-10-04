// components/common/ErrorDisplay.jsx
import React from 'react';
import { AlertCircle } from 'lucide-react';

const ErrorDisplay = ({ error, title = 'Error' }) => {
  if (!error) return null;

  return (
    <div className="m-4">
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <div className="flex items-start gap-2">
          <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
          <div>
            <h3 className="font-semibold text-red-800 dark:text-red-300">{title}</h3>
            <p className="text-sm text-red-700 dark:text-red-400 mt-1">{error}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ErrorDisplay;