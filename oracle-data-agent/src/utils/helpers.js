// utils/helpers.js

export const exportToCSV = (columns, rows) => {
  if (!columns || !rows) return;

  const csvContent = [
    columns.join(','),
    ...rows.map(row => 
      row.map(cell => 
        typeof cell === 'string' && cell.includes(',') 
          ? `"${cell}"` 
          : cell ?? ''
      ).join(',')
    )
  ].join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `query_result_${new Date().toISOString()}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
};

export const filterRows = (rows, searchTerm) => {
  if (!rows || !searchTerm) return rows || [];
  
  return rows.filter(row =>
    row.some(cell =>
      String(cell ?? '').toLowerCase().includes(searchTerm.toLowerCase())
    )
  );
};

export const paginateRows = (rows, currentPage, rowsPerPage) => {
  const start = (currentPage - 1) * rowsPerPage;
  const end = start + rowsPerPage;
  return rows.slice(start, end);
};

export const calculateTotalPages = (totalRows, rowsPerPage) => {
  return Math.ceil(totalRows / rowsPerPage);
};

export const formatCellValue = (value) => {
  if (value === null || value === undefined) {
    return { text: 'NULL', isNull: true };
  }
  
  if (typeof value === 'number') {
    return {
      text: Number.isInteger(value) ? value.toString() : parseFloat(value).toFixed(2),
      isNumber: true
    };
  }
  
  const stringValue = String(value);
  // Don't truncate in the formatter, let CSS handle it
  return {
    text: stringValue,
    fullText: stringValue,
    isTruncated: false
  };
};

export const getOptimalColumnWidth = (columnName, rows = [], columnIndex = 0) => {
  const minWidth = 100; // Minimum width in pixels
  const maxWidth = 300; // Maximum width in pixels
  const charWidth = 7; // Approximate width per character
  const padding = 24; // Extra padding for cell
  
  // Start with column header width
  let maxLength = columnName.length;
  
  // Sample first 10 rows to determine content width
  const sampleSize = Math.min(10, rows.length);
  for (let i = 0; i < sampleSize; i++) {
    if (rows[i] && rows[i][columnIndex]) {
      const cellLength = String(rows[i][columnIndex]).length;
      maxLength = Math.max(maxLength, cellLength);
    }
  }
  
  const calculatedWidth = maxLength * charWidth + padding;
  return Math.min(maxWidth, Math.max(minWidth, calculatedWidth));
};

// Format SQL query for pretty display
export const formatSQL = (sql) => {
  if (!sql) return '';
  
  // SQL keywords that should start a new line
  const keywords = [
    'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'GROUP BY', 'ORDER BY', 
    'HAVING', 'LIMIT', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN',
    'OUTER JOIN', 'ON', 'UNION', 'INSERT', 'UPDATE', 'DELETE', 'CREATE',
    'ALTER', 'DROP', 'WITH', 'AS', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END'
  ];
  
  let formatted = sql;
  
  // Add line breaks before major keywords
  keywords.forEach(keyword => {
    const regex = new RegExp(`\\b(${keyword})\\b`, 'gi');
    formatted = formatted.replace(regex, '\n$1');
  });
  
  // Clean up multiple line breaks
  formatted = formatted.replace(/\n+/g, '\n');
  
  // Remove leading newline if present
  formatted = formatted.trim();
  
  // Add indentation for better readability
  const lines = formatted.split('\n');
  let indentLevel = 0;
  const formattedLines = lines.map(line => {
    const trimmedLine = line.trim();
    
    // Decrease indent for closing parenthesis
    if (trimmedLine.startsWith(')')) {
      indentLevel = Math.max(0, indentLevel - 1);
    }
    
    const indentedLine = '  '.repeat(indentLevel) + trimmedLine;
    
    // Increase indent after opening parenthesis
    if (trimmedLine.endsWith('(')) {
      indentLevel++;
    }
    
    return indentedLine;
  });
  
  return formattedLines.join('\n');
};