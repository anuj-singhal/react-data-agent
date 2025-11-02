// components/charts/ChartRenderer.jsx
import React, { useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Bar, Line, Pie } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const ChartRenderer = ({ data, config }) => {
  const chartRef = useRef(null);

  // Generate colors for chart
  const generateColors = (count) => {
    const colors = [
      'rgba(59, 130, 246, 0.8)',   // Blue
      'rgba(16, 185, 129, 0.8)',   // Green
      'rgba(245, 158, 11, 0.8)',   // Orange
      'rgba(239, 68, 68, 0.8)',    // Red
      'rgba(139, 92, 246, 0.8)',   // Purple
      'rgba(236, 72, 153, 0.8)',   // Pink
      'rgba(14, 165, 233, 0.8)',   // Cyan
      'rgba(251, 146, 60, 0.8)',   // Orange-red
    ];
    
    const result = [];
    for (let i = 0; i < count; i++) {
      result.push(colors[i % colors.length]);
    }
    return result;
  };

  const preparePieData = () => {
    if (!data || !Array.isArray(data)) return null;

    const labels = data.map(item => item.label || item.x);
    const values = data.map(item => item.y);
    const colors = generateColors(data.length);

    return {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderColor: colors.map(color => color.replace('0.8', '1')),
        borderWidth: 1
      }]
    };
  };

  const prepareBarLineData = () => {
    if (!data) return null;

    // Handle multiple series
    if (data.datasets && Array.isArray(data.datasets)) {
      const colors = generateColors(data.datasets.length);
      
      return {
        labels: data.labels,
        datasets: data.datasets.map((dataset, index) => ({
          label: dataset.name,
          data: dataset.data,
          backgroundColor: colors[index],
          borderColor: colors[index].replace('0.8', '1'),
          borderWidth: 2,
          fill: config.type === 'area'
        }))
      };
    }

    // Single series
    if (Array.isArray(data)) {
      const labels = data.map(item => item.label || item.x);
      const values = data.map(item => item.y);
      
      return {
        labels,
        datasets: [{
          label: config.yAxis || 'Value',
          data: values,
          backgroundColor: 'rgba(59, 130, 246, 0.8)',
          borderColor: 'rgba(59, 130, 246, 1)',
          borderWidth: 2,
          fill: config.type === 'area'
        }]
      };
    }

    return null;
  };

  const getChartOptions = () => {
    const baseOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: config.showLegend !== false,
          position: 'top',
        },
        title: {
          display: !!config.title,
          text: config.title || '',
          font: {
            size: 16,
            weight: 'bold'
          }
        },
        tooltip: {
          mode: 'index',
          intersect: false,
        }
      }
    };

    // Add scales for bar/line/area charts
    if (['bar', 'line', 'area', 'multibar'].includes(config.type)) {
      baseOptions.scales = {
        x: {
          grid: {
            display: config.showGrid !== false,
            color: 'rgba(0, 0, 0, 0.05)'
          },
          ticks: {
            maxRotation: 45,
            minRotation: 0
          }
        },
        y: {
          beginAtZero: true,
          grid: {
            display: config.showGrid !== false,
            color: 'rgba(0, 0, 0, 0.05)'
          }
        }
      };
    }

    return baseOptions;
  };

  const renderChart = () => {
    const options = getChartOptions();

    switch (config.type) {
      case 'pie':
        const pieData = preparePieData();
        if (!pieData) return <NoDataMessage />;
        return <Pie ref={chartRef} data={pieData} options={options} />;

      case 'line':
      case 'area':
        const lineData = prepareBarLineData();
        if (!lineData) return <NoDataMessage />;
        return <Line ref={chartRef} data={lineData} options={options} />;

      case 'bar':
      case 'multibar':
        const barData = prepareBarLineData();
        if (!barData) return <NoDataMessage />;
        return <Bar ref={chartRef} data={barData} options={options} />;

      default:
        return <NoDataMessage message="Unsupported chart type" />;
    }
  };

  if (!data || !config) {
    return <NoDataMessage />;
  }

  return (
    <div className="w-full h-full min-h-[400px] relative">
      {renderChart()}
    </div>
  );
};

const NoDataMessage = ({ message = "No data to display" }) => (
  <div className="flex items-center justify-center h-full">
    <p className="text-gray-500 dark:text-gray-400">{message}</p>
  </div>
);

export default ChartRenderer;