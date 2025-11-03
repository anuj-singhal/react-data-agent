// components/chat/ChatInterface.jsx
import React, { useState, useEffect, useRef } from 'react';
import { Send, Loader2, Bot, User, Database, Code, CheckCircle, X, Copy } from 'lucide-react';

import { formatSQL } from '../../utils/helpers';

const ChatInterface = ({ onQueryExecute, sessionId, onSessionStart }) => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [pendingQuery, setPendingQuery] = useState(null);
  const messagesEndRef = useRef(null);
  const [copiedMessageId, setCopiedMessageId] = useState(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (!sessionId && !onSessionStart.called) {
      onSessionStart.called = true;
      startNewSession();
    }
    return () => {
      onSessionStart.called = false;
    };
  }, [sessionId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const startNewSession = async () => {
    try {
      const response = await fetch('http://localhost:8000/chat/start', {
        method: 'POST'
      });
      const data = await response.json();
      onSessionStart(data.session_id);
      
      // Add welcome message
      addMessage('assistant', data.message);
    } catch (error) {
      console.error('Failed to start chat session:', error);
    }
  };

  const addMessage = (role, content, metadata = {}) => {
    const newMessage = {
      id: Date.now(),
      role,
      content,
      timestamp: new Date().toISOString(),
      ...metadata
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const sendMessage = async (message = inputMessage) => {
    if (!message.trim() || !sessionId) return;

    // Add user message to chat
    addMessage('user', message);
    setInputMessage('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          message: message,
          include_history: true
        })
      });

      const data = await response.json();
      
      // Handle response based on action type
      if (data.action_type === 'query_confirmation') {
        // Show query confirmation
        setPendingQuery(data.sql_query);
        addMessage('assistant', data.message, { 
          type: 'confirmation',
          sql_query: data.sql_query 
        });
      } else if (data.action_type === 'query_execution') {
        // Query was executed
        if (data.result) {
          onQueryExecute(data.sql_query, data.result);
        }
        addMessage('assistant', data.message, {
          type: 'execution',
          sql_query: data.sql_query,
          has_result: !!data.result
        });
      } else if (data.action_type === 'follow_up') {
        // Follow-up question answered
        addMessage('assistant', data.message, { type: 'follow_up' });
      } else {
        // General response
        addMessage('assistant', data.message);
      }

      if (data.error) {
        addMessage('system', `Error: ${data.error}`, { type: 'error' });
      }

    } catch (error) {
      console.error('Failed to send message:', error);
      addMessage('system', 'Failed to send message. Please try again.', { type: 'error' });
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleCopySQL = (sql, messageId) => {
    navigator.clipboard.writeText(sql).then(() => {
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000);
    });
  };

  const handleConfirmQuery = async (confirmed, feedback = '') => {
    if (!sessionId || !pendingQuery) return;

    setLoading(true);
    
    try {
      const response = await fetch('http://localhost:8000/chat/confirm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          sql_query: pendingQuery,
          confirmed: confirmed,
          user_feedback: feedback
        })
      });

      const data = await response.json();
      
      if (data.success) {
        if (data.result) {
          // Query executed
          onQueryExecute(pendingQuery, data.result);
          addMessage('assistant', data.message, {
            type: 'execution',
            has_result: true
          });
          setPendingQuery(null);
        } else if (data.requires_confirmation) {
          // Modified query needs confirmation
          setPendingQuery(data.sql_query);
          addMessage('assistant', data.message, {
            type: 'confirmation',
            sql_query: data.sql_query
          });
        } else {
          addMessage('assistant', data.message);
          setPendingQuery(null);
        }
      } else {
        addMessage('system', data.message || 'Operation failed', { type: 'error' });
      }
    } catch (error) {
      console.error('Failed to confirm query:', error);
      addMessage('system', 'Failed to process confirmation', { type: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const renderMessage = (message) => {
    const isUser = message.role === 'user';
    const isSystem = message.role === 'system';
    
    return (
      <div
        key={message.id}
        className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'} mb-4`}
      >
        {!isUser && (
          <div className="flex-shrink-0">
            {isSystem ? (
              <div className="w-8 h-8 rounded-full bg-red-100 dark:bg-red-900 flex items-center justify-center">
                <Database className="w-4 h-4 text-red-600 dark:text-red-400" />
              </div>
            ) : (
              <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center">
                <Bot className="w-4 h-4 text-blue-600 dark:text-blue-400" />
              </div>
            )}
          </div>
        )}
        
        <div className={`max-w-[70%] ${isUser ? 'order-1' : ''}`}>
          <div
            className={`rounded-lg px-4 py-2 ${
              isUser
                ? 'bg-blue-600 text-white'
                : isSystem
                ? 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100'
            }`}
          >
            {message.type === 'confirmation' && message.sql_query ? (
              <div>
                <p className="mb-3 whitespace-pre-wrap">{message.content.split('```')[0]}</p>
                <div className="bg-gray-900 dark:bg-gray-800 rounded p-4 my-3 max-w-full">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-400 flex items-center gap-1">
                      <Code className="w-4 h-4" />
                      SQL Query
                    </span>
                    <button
                      onClick={() => handleCopySQL(message.sql_query, message.id)}
                      className="p-1.5 hover:bg-gray-700 rounded transition-colors group"
                      title="Copy SQL"
                    >
                      {copiedMessageId === message.id ? (
                        <CheckCircle className="w-4 h-4 text-green-400" />
                      ) : (
                        <Copy className="w-4 h-4 text-gray-400 group-hover:text-gray-300" />
                      )}
                    </button>
                  </div>
                  <div className="overflow-x-auto max-h-64 overflow-y-auto">
                    <pre className="text-sm text-gray-300 min-w-0">
                      <code className="block whitespace-pre">{formatSQL(message.sql_query)}</code>
                    </pre>
                  </div>
                </div>
                <div className="flex gap-2 mt-3">
                  <button
                    onClick={() => handleConfirmQuery(true)}
                    disabled={loading}
                    className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white rounded text-sm flex items-center gap-1"
                  >
                    <CheckCircle className="w-3 h-3" />
                    Execute
                  </button>
                  {/* <button
                    onClick={() => {
                      const feedback = prompt('What would you like to change?');
                      if (feedback) handleConfirmQuery(false, feedback);
                    }}
                    disabled={loading}
                    className="px-3 py-1 bg-yellow-600 hover:bg-yellow-700 text-white rounded text-sm flex items-center gap-1"
                  >
                    <Edit className="w-3 h-3" />
                    Modify
                  </button> */}
                  <button
                    onClick={() => handleConfirmQuery(false)}
                    disabled={loading}
                    className="px-3 py-1 bg-gray-600 hover:bg-gray-700 text-white rounded text-sm flex items-center gap-1"
                  >
                    <X className="w-3 h-3" />
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <p className="whitespace-pre-wrap">{message.content}</p>
            )}
            
            {message.has_result && (
              <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-600">
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  ✓ Results displayed in the table
                </span>
              </div>
            )}
          </div>
          
          <span className="text-xs text-gray-500 dark:text-gray-400 mt-1 block">
            {new Date(message.timestamp).toLocaleTimeString()}
          </span>
        </div>
        
        {isUser && (
          <div className="flex-shrink-0 order-2">
            <div className="w-8 h-8 rounded-full bg-green-100 dark:bg-green-900 flex items-center justify-center">
              <User className="w-4 h-4 text-green-600 dark:text-green-400" />
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full bg-white dark:bg-gray-800">
      {/* Chat Header */}
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
        <h3 className="font-semibold text-gray-800 dark:text-white flex items-center gap-2">
          <Bot className="w-4 h-4" />
          Query Assistant
        </h3>
        <button
          onClick={startNewSession}
          className="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
        >
          New Session
        </button>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 dark:text-gray-400 mt-8">
            <Bot className="w-12 h-12 mx-auto mb-3 text-gray-400" />
            <p>Start by asking a question about your database!</p>
            <div className="mt-4 space-y-2">
              <p className="text-sm">Try asking:</p>
              <div className="flex flex-wrap gap-2 justify-center mt-2">
                {[
                  "Show me all tables",
                  "Get top 5 banks by profit",
                  "What's the average deposits?"
                ].map((suggestion, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setInputMessage(suggestion);
                      inputRef.current?.focus();
                    }}
                    className="px-3 py-1 bg-gray-100 dark:bg-gray-700 rounded-full text-xs hover:bg-gray-200 dark:hover:bg-gray-600"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <>
            {messages.map(renderMessage)}
            {loading && (
              <div className="flex gap-3 mb-4">
                <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center">
                  <Bot className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                </div>
                <div className="bg-gray-100 dark:bg-gray-700 rounded-lg px-4 py-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                </div>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 dark:border-gray-700 p-4">
        <div className="flex gap-2">
          <textarea
            ref={inputRef}
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Ask about your data or request modifications..."
            className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                      bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                      focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            rows={2}
            disabled={loading}
          />
          <button
            onClick={() => sendMessage()}
            disabled={loading || !inputMessage.trim()}
            className={`px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2
              ${loading || !inputMessage.trim()
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 active:scale-95'} 
              text-white`}
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>
        <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
          Press Enter to send • Shift+Enter for new line
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;