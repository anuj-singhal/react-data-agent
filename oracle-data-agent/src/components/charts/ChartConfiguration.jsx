// components/charts/ChartConfiguration.jsx
import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  LineChart as LineChartIcon, 
  PieChart as PieChartIcon,
  Activity,
  Layers
} from 'lucide-react';

const CHART_TYPES = [
  { id: 'bar', name: 'Bar Chart', icon: BarChart3, description: 'Compare values across categories' },
  { id: 'line', name: 'Line Chart', icon: LineChartIcon, description: 'Show trends over time' },
  { id: 'pie', name: 'Pie Chart', icon: PieChartIcon, description: 'Show parts of a whole' },
  { id: 'area', name: 'Area Chart', icon: Activity, description: 'Show cumulative trends' },
  { id: 'multibar', name: 'Grouped Bar', icon: Layers, description: 'Compare multiple series' },
];

const ChartConfiguration = ({ result, config, onChange }) => {
  const [selectedType, setSelectedType] = useState(config?.type || 'bar');
  const [xAxis, setXAxis] = useState(config?.xAxis || '');
  const [yAxis, setYAxis] = useState(config?.yAxis || '');
  const [series, setSeries] = useState(config?.series || []);
  const [title, setTitle] = useState(config?.title || 'Data Visualization');
  const [showLegend, setShowLegend] = useState(config?.showLegend !== false);
  const [showGrid, setShowGrid] = useState(config?.showGrid !== false);

  const { columns, rows } = result || {};

  useEffect(() => {
    // Update parent config
    onChange({
      type: selectedType,
      xAxis,
      yAxis,
      series,
      title,
      showLegend,
      showGrid
    });
  }, [selectedType, xAxis, yAxis, series, title, showLegend, showGrid]);

  const getNumericColumns = () => {
    if (!rows || rows.length === 0) return [];
    
    return columns.filter((col, idx) => {
      const sample = rows.slice(0, 10).map(row => row[idx]);
      return sample.every(val => 
        val === null || val === undefined || !isNaN(parseFloat(val))
      );
    });
  };

  const getTextColumns = () => {
    if (!rows || rows.length === 0) return [];
    
    return columns.filter((col, idx) => {
      const sample = rows.slice(0, 10).map(row => row[idx]);
      return sample.some(val => 
        typeof val === 'string' && isNaN(parseFloat(val))
      );
    });
  };

  const numericColumns = getNumericColumns();
  const textColumns = getTextColumns();

  const handleSeriesToggle = (column) => {
    if (series.includes(column)) {
      setSeries(series.filter(s => s !== column));
    } else {
      setSeries([...series, column]);
    }
  };

  const supportsMultipleSeries = ['multibar', 'line', 'area'].includes(selectedType);
  const requiresNumericY = ['bar', 'line', 'area', 'multibar'].includes(selectedType);

  return (
    <div className="p-4 space-y-6">
      {/* Chart Type Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
          Chart Type
        </label>
        <div className="space-y-2">
          {CHART_TYPES.map(type => {
            const Icon = type.icon;
            return (
              <button
                key={type.id}
                onClick={() => setSelectedType(type.id)}
                className={`w-full p-3 rounded-lg border-2 transition-all text-left ${
                  selectedType === type.id
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                }`}
              >
                <div className="flex items-start gap-3">
                  <Icon className={`w-5 h-5 mt-0.5 ${
                    selectedType === type.id ? 'text-blue-600' : 'text-gray-400'
                  }`} />
                  <div>
                    <div className={`font-medium ${
                      selectedType === type.id 
                        ? 'text-blue-700 dark:text-blue-400' 
                        : 'text-gray-700 dark:text-gray-300'
                    }`}>
                      {type.name}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                      {type.description}
                    </div>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Chart Title */}
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Chart Title
        </label>
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                    bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                    focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          placeholder="Enter chart title"
        />
      </div>

      {/* X-Axis Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          X-Axis (Category)
        </label>
        <select
          value={xAxis}
          onChange={(e) => setXAxis(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                    bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                    focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          <option value="">Select column...</option>
          {selectedType === 'pie' ? (
            // For pie chart, allow any column
            columns.map(col => (
              <option key={col} value={col}>{col}</option>
            ))
          ) : (
            // For other charts, prefer text columns
            (textColumns.length > 0 ? textColumns : columns).map(col => (
              <option key={col} value={col}>{col}</option>
            ))
          )}
        </select>
        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
          {selectedType === 'pie' ? 'Labels for pie slices' : 'Categories for the horizontal axis'}
        </p>
      </div>

      {/* Y-Axis Selection */}
      {requiresNumericY && (
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Y-Axis (Value)
          </label>
          <select
            value={yAxis}
            onChange={(e) => setYAxis(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                      bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                      focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">Select column...</option>
            {numericColumns.map(col => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Numeric values for the vertical axis
          </p>
        </div>
      )}

      {/* Pie Chart Value Selection */}
      {selectedType === 'pie' && (
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Values
          </label>
          <select
            value={yAxis}
            onChange={(e) => setYAxis(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                      bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                      focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">Select column...</option>
            {numericColumns.map(col => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Numeric values for pie slice sizes
          </p>
        </div>
      )}

      {/* Multiple Series Selection */}
      {supportsMultipleSeries && numericColumns.length > 1 && (
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Additional Series (Optional)
          </label>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {numericColumns.filter(col => col !== yAxis).map(col => (
              <label
                key={col}
                className="flex items-center gap-2 p-2 hover:bg-gray-50 dark:hover:bg-gray-700 rounded cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={series.includes(col)}
                  onChange={() => handleSeriesToggle(col)}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">{col}</span>
              </label>
            ))}
          </div>
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Select additional columns to compare
          </p>
        </div>
      )}

      {/* Chart Options */}
      <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
          Chart Options
        </label>
        <div className="space-y-2">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showLegend}
              onChange={(e) => setShowLegend(e.target.checked)}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">Show Legend</span>
          </label>
          
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showGrid}
              onChange={(e) => setShowGrid(e.target.checked)}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">Show Grid Lines</span>
          </label>
        </div>
      </div>

      {/* Preview Info */}
      {xAxis && yAxis && (
        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <p className="text-xs text-blue-800 dark:text-blue-300 font-medium mb-1">
            Configuration Ready
          </p>
          <p className="text-xs text-blue-600 dark:text-blue-400">
            {selectedType === 'pie' 
              ? `${xAxis} by ${yAxis}`
              : `${yAxis} by ${xAxis}`
            }
            {series.length > 0 && ` (+ ${series.length} series)`}
          </p>
        </div>
      )}
    </div>
  );
};

export default ChartConfiguration;