// components/query/SqlDisplay.jsx
import React, { useState, useEffect, useRef } from 'react';
import { Code, Copy, CheckCircle, Maximize2, Minimize2 } from 'lucide-react';
import { formatSQL } from '../../utils/helpers';

const SqlDisplay = ({ sqlQuery }) => {
  const [copySuccess, setCopySuccess] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [panelHeight, setPanelHeight] = useState(200); // Default height in pixels
  const [isResizingVertical, setIsResizingVertical] = useState(false);
  const panelRef = useRef(null);
  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    });
  };

  const formattedSQL = formatSQL(sqlQuery);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizingVertical) return;
      
      const panelTop = panelRef.current?.getBoundingClientRect().top || 0;
      const newHeight = Math.max(100, Math.min(400, e.clientY - panelTop));
      setPanelHeight(newHeight);
    };

    const handleMouseUp = () => {
      setIsResizingVertical(false);
      document.body.style.cursor = 'default';
      document.body.style.userSelect = 'auto';
    };

    if (isResizingVertical) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'ns-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizingVertical]);

  return (
    <div 
      ref={panelRef}
      className="relative bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700"
      style={{ height: `${panelHeight}px` }}
    >
      {/* Add resize handle at the top */}
      <div
        className="absolute top-0 left-0 right-0 h-1 hover:h-2 bg-transparent hover:bg-blue-500/20 cursor-ns-resize z-10 transition-all"
        onMouseDown={() => setIsResizingVertical(true)}
      />
      
      <div className="p-4 h-full flex flex-col">
        {/* Keep existing header content */}
        <div className="flex items-center justify-between mb-3">
          {/* ... existing header code ... */}
        </div>
        
        {/* Update the content div to be scrollable within the fixed height */}
        <div className="flex-1 overflow-auto">
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
    </div>
  );
};

export default SqlDisplay;