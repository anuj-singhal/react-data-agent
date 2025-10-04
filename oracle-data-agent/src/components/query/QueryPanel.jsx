// components/query/QueryPanel.jsx
import React, { useState, useRef, useEffect } from 'react';
import { ChevronLeft, ChevronRight, GripVertical, MessageSquare, History } from 'lucide-react';
import ChatInterface from '../chat/ChatInterface';
import SqlDisplay from './SqlDisplay';
import QueryHistory from './QueryHistory';

const QueryPanel = ({ 
  sqlQuery,
  queryHistory,
  onQueryExecute,
  collapsed,
  setCollapsed,
  width,
  onWidthChange
}) => {
  const [isResizing, setIsResizing] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [activeTab, setActiveTab] = useState('chat'); // 'chat' or 'history'
  const panelRef = useRef(null);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizing) return;
      
      const newWidth = e.clientX;
      if (newWidth > 250 && newWidth < window.innerWidth - 400) {
        onWidthChange(newWidth);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      document.body.style.cursor = 'default';
      document.body.style.userSelect = 'auto';
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing, onWidthChange]);

  const handleQueryExecute = (sql, result) => {
    onQueryExecute(sql, result);
  };

  return (
    <div 
      ref={panelRef}
      className={`${collapsed ? 'w-12' : ''} transition-all duration-300 flex flex-col bg-gray-50 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 relative`}
      style={{ width: collapsed ? '48px' : `${width}px` }}
    >
      
      {/* Collapse Toggle Button */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="absolute -right-3 top-1/2 transform -translate-y-1/2 z-20 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-full p-1 shadow-md hover:bg-gray-50 dark:hover:bg-gray-700"
        title={collapsed ? "Expand panel" : "Collapse panel"}
      >
        {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
      </button>

      {/* Resize Handle */}
      {!collapsed && (
        <div
          className="absolute right-0 top-0 bottom-0 w-1 hover:w-2 bg-transparent hover:bg-blue-500/20 cursor-col-resize z-10 transition-all group"
          onMouseDown={() => setIsResizing(true)}
        >
          <div className="absolute right-0 top-1/2 transform -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
            <GripVertical className="w-4 h-4 text-gray-400" />
          </div>
        </div>
      )}

      {!collapsed && (
        <>
          {/* Tab Navigation */}
          <div className="flex border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
            <button
              onClick={() => setActiveTab('chat')}
              className={`flex-1 px-4 py-3 text-sm font-medium flex items-center justify-center gap-2 transition-colors ${
                activeTab === 'chat'
                  ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
              }`}
            >
              <MessageSquare className="w-4 h-4" />
              Chat
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`flex-1 px-4 py-3 text-sm font-medium flex items-center justify-center gap-2 transition-colors ${
                activeTab === 'history'
                  ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
              }`}
            >
              <History className="w-4 h-4" />
              History
              {queryHistory.length > 0 && (
                <span className="ml-1 px-2 py-0.5 bg-gray-200 dark:bg-gray-700 rounded-full text-xs">
                  {queryHistory.length}
                </span>
              )}
            </button>
          </div>
          
          {/* Tab Content */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {activeTab === 'chat' ? (
              <>
                <div className="flex-1 overflow-hidden">
                  <ChatInterface 
                    onQueryExecute={handleQueryExecute}
                    sessionId={sessionId}
                    onSessionStart={setSessionId}
                  />
                </div>
                
                {sqlQuery && (
                  <div className="border-t border-gray-200 dark:border-gray-700">
                    <SqlDisplay sqlQuery={sqlQuery} />
                  </div>
                )}
              </>
            ) : (
              <QueryHistory 
                queryHistory={queryHistory}
                onSelectQuery={(query) => {
                  // Switch to chat tab and send the query
                  setActiveTab('chat');
                  // You might want to send this query to chat
                }}
              />
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default QueryPanel;