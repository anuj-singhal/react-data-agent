// components/charts/ChartPanel.jsx
import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  LineChart as LineChartIcon, 
  PieChart as PieChartIcon,
  ScatterChart as ScatterChartIcon,
  TrendingUp,
  X,
  Download,
  Settings
} from 'lucide-react';
import ChartRenderer from './ChartRenderer';
import ChartConfiguration from './ChartConfiguration';

const ChartPanel = ({ result, visible, onClose }) => {
  const [chartConfig, setChartConfig] = useState(null);
  const [showConfig, setShowConfig] = useState(true);
  const [chartData, setChartData] = useState(null);

  useEffect(() => {
    if (result && result.columns && result.rows) {
      // Auto-suggest initial chart configuration
      const numericColumns = getNumericColumns(result.columns, result.rows);
      const textColumns = getTextColumns(result.columns, result.rows);
      
      if (numericColumns.length > 0 && textColumns.length > 0) {
        setChartConfig({
          type: 'bar',
          xAxis: textColumns[0],
          yAxis: numericColumns[0],
          title: 'Data Visualization'
        });
        setShowConfig(false);
      }
    }
  }, [result]);

  useEffect(() => {
    if (chartConfig && result) {
      const data = prepareChartData(result, chartConfig);
      setChartData(data);
    }
  }, [chartConfig, result]);

  const getNumericColumns = (columns, rows) => {
    if (!rows || rows.length === 0) return [];
    
    return columns.filter((col, idx) => {
      // Check first few rows to determine if column is numeric
      const sample = rows.slice(0, 10).map(row => row[idx]);
      return sample.every(val => 
        val === null || val === undefined || !isNaN(parseFloat(val))
      );
    });
  };

  const getTextColumns = (columns, rows) => {
    if (!rows || rows.length === 0) return [];
    
    return columns.filter((col, idx) => {
      const sample = rows.slice(0, 10).map(row => row[idx]);
      return sample.some(val => 
        typeof val === 'string' && isNaN(parseFloat(val))
      );
    });
  };

  const prepareChartData = (result, config) => {
    if (!result || !config) return null;

    const xIndex = result.columns.indexOf(config.xAxis);
    const yIndex = result.columns.indexOf(config.yAxis);
    
    if (xIndex === -1 || yIndex === -1) return null;

    const data = result.rows.map(row => ({
      x: row[xIndex],
      y: parseFloat(row[yIndex]) || 0,
      label: row[xIndex]
    }));

    // Handle additional series for multi-series charts
    if (config.series && config.series.length > 0) {
      const seriesData = config.series.map(seriesCol => {
        const seriesIndex = result.columns.indexOf(seriesCol);
        return {
          name: seriesCol,
          data: result.rows.map(row => parseFloat(row[seriesIndex]) || 0)
        };
      });
      
      return {
        labels: result.rows.map(row => row[xIndex]),
        datasets: seriesData
      };
    }

    return data;
  };

  const handleDownloadChart = () => {
    // Download chart as image
    const canvas = document.querySelector('canvas');
    if (canvas) {
      const url = canvas.toDataURL('image/png');
      const link = document.createElement('a');
      link.download = `chart_${new Date().getTime()}.png`;
      link.href = url;
      link.click();
    }
  };

  if (!visible || !result) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-2xl w-full max-w-6xl h-5/6 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-blue-600" />
            <h2 className="text-lg font-semibold text-gray-800 dark:text-white">
              Data Visualization
            </h2>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowConfig(!showConfig)}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Chart Settings"
            >
              <Settings className="w-4 h-4 text-gray-600 dark:text-gray-400" />
            </button>
            
            <button
              onClick={handleDownloadChart}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Download Chart"
            >
              <Download className="w-4 h-4 text-gray-600 dark:text-gray-400" />
            </button>
            
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Close"
            >
              <X className="w-4 h-4 text-gray-600 dark:text-gray-400" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 flex overflow-hidden">
          {/* Chart Configuration Panel */}
          {showConfig && (
            <div className="w-80 border-r border-gray-200 dark:border-gray-700 overflow-y-auto">
              <ChartConfiguration
                result={result}
                config={chartConfig}
                onChange={setChartConfig}
              />
            </div>
          )}

          {/* Chart Display */}
          <div className="flex-1 p-6 overflow-auto">
            {chartConfig && chartData ? (
              <ChartRenderer
                data={chartData}
                config={chartConfig}
              />
            ) : (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <BarChart3 className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                  <p className="text-gray-600 dark:text-gray-400 mb-2">
                    No chart configuration
                  </p>
                  <button
                    onClick={() => setShowConfig(true)}
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                  >
                    Configure Chart
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer with quick stats */}
        {result && (
          <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
            <div className="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400">
              <span>
                {result.rows.length} rows × {result.columns.length} columns
              </span>
              {chartConfig && (
                <span className="flex items-center gap-1">
                  <TrendingUp className="w-4 h-4" />
                  {chartConfig.type} chart
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChartPanel;