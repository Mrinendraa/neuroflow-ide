import React, { useMemo, useState } from 'react';
import { Handle, Position, useStore } from 'reactflow';
import './DataVisualizerNode.css';
import { parseFullTabularFile } from '../../utils/parseTabularFile';

const width = 500;
const height = 350;
const padding = 60;
const bottomPadding = 80; // Extra space for rotated labels

// Helper function to format numbers intelligently
function formatNumber(num) {
  if (Math.abs(num) >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M';
  } else if (Math.abs(num) >= 1000) {
    return (num / 1000).toFixed(1) + 'K';
  } else if (Math.abs(num) < 1 && num !== 0) {
    return num.toFixed(2);
  }
  return num.toFixed(1);
}

// Helper function to truncate text
function truncateText(text, maxLength = 12) {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - 2) + '..';
}

function createAxes(xDomain, yDomain, xLabel, yLabel, isNumericX = true) {
  const [xMin, xMax] = xDomain;
  const [yMin, yMax] = yDomain;
  const scaleX = (x) => padding + ((x - xMin) / (xMax - xMin)) * (width - 2 * padding);
  const scaleY = (y) => height - bottomPadding - ((y - yMin) / (yMax - yMin)) * (height - bottomPadding - padding);

  // Grid lines
  const gridLines = [];
  for (let i = 1; i <= 4; i++) {
    const y = yMin + i * ((yMax - yMin) / 5);
    const yPos = scaleY(y);
    gridLines.push(
      <line
        key={`grid-${i}`}
        x1={padding}
        y1={yPos}
        x2={width - padding}
        y2={yPos}
        stroke="#e0e0e0"
        strokeWidth="1"
        strokeDasharray="3,3"
      />
    );
  }

  // X-axis
  const xAxis = (
    <line
      x1={padding}
      y1={height - bottomPadding}
      x2={width - padding}
      y2={height - bottomPadding}
      stroke="#555"
      strokeWidth="2"
    />
  );

  // Y-axis
  const yAxis = (
    <line
      x1={padding}
      y1={padding}
      x2={padding}
      y2={height - bottomPadding}
      stroke="#555"
      strokeWidth="2"
    />
  );

  // X-axis ticks and labels
  const xTicks = [];
  const numXTicks = isNumericX ? 5 : Math.min(xMax - xMin, 8);
  const xStep = (xMax - xMin) / numXTicks;

  for (let i = 0; i <= numXTicks; i++) {
    const x = xMin + i * xStep;
    const xPos = scaleX(x);
    xTicks.push(
      <g key={`x-tick-${i}`}>
        <line x1={xPos} y1={height - bottomPadding} x2={xPos} y2={height - bottomPadding + 5} stroke="#555" strokeWidth="1.5" />
        <text
          x={xPos}
          y={height - bottomPadding + 15}
          textAnchor="middle"
          fontSize="11"
          fill="#444"
          fontWeight="500"
        >
          {formatNumber(x)}
        </text>
      </g>
    );
  }

  // Y-axis ticks and labels
  const yTicks = [];
  const yStep = (yMax - yMin) / 5;
  for (let i = 0; i <= 5; i++) {
    const y = yMin + i * yStep;
    const yPos = scaleY(y);
    yTicks.push(
      <g key={`y-tick-${i}`}>
        <line x1={padding} y1={yPos} x2={padding - 5} y2={yPos} stroke="#555" strokeWidth="1.5" />
        <text
          x={padding - 10}
          y={yPos + 4}
          textAnchor="end"
          fontSize="11"
          fill="#444"
          fontWeight="500"
        >
          {formatNumber(y)}
        </text>
      </g>
    );
  }

  return (
    <g>
      {gridLines}
      {xAxis}
      {yAxis}
      {xTicks}
      {yTicks}
      <text
        x={width / 2}
        y={height - 10}
        textAnchor="middle"
        fontSize="13"
        fill="#222"
        fontWeight="600"
      >
        {xLabel}
      </text>
      <text
        x={15}
        y={height / 2}
        textAnchor="middle"
        fontSize="13"
        fill="#222"
        fontWeight="600"
        transform={`rotate(-90, 15, ${height / 2})`}
      >
        {yLabel}
      </text>
    </g>
  );
}

function createScatterPlot(points, xDomain, yDomain, xLabel, yLabel) {
  const [xMin, xMax] = xDomain;
  const [yMin, yMax] = yDomain;
  const scaleX = (x) => padding + ((x - xMin) / (xMax - xMin)) * (width - 2 * padding);
  const scaleY = (y) => height - bottomPadding - ((y - yMin) / (yMax - yMin)) * (height - bottomPadding - padding);

  const dots = points.map((p, i) => (
    <g key={i}>
      <circle
        cx={scaleX(p[0])}
        cy={scaleY(p[1])}
        r={5}
        fill="url(#scatterGradient)"
        className="scatter-point"
        opacity="0.85"
      >
        <title>{`(${p[0].toFixed(2)}, ${p[1].toFixed(2)})`}</title>
      </circle>
    </g>
  ));

  return (
    <svg width={width} height={height} className="chart-svg">
      <defs>
        <linearGradient id="scatterGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#667eea" />
          <stop offset="100%" stopColor="#764ba2" />
        </linearGradient>
        <filter id="shadow">
          <feDropShadow dx="0" dy="2" stdDeviation="3" floodOpacity="0.3" />
        </filter>
      </defs>
      <rect x="0" y="0" width={width} height={height} fill="#fafafa" rx="8" />
      {createAxes(xDomain, yDomain, xLabel, yLabel)}
      {dots}
    </svg>
  );
}

function createHistogram(values, bins, label) {
  const maxCount = Math.max(...bins.map(bin => bin.count));
  const binWidth = (width - 2 * padding) / bins.length;
  const shouldRotateLabels = bins.length > 6 || binWidth < 50;

  const bars = bins.map((bin, i) => {
    const barHeight = maxCount > 0 ? (bin.count / maxCount) * (height - bottomPadding - padding) : 0;
    const x = padding + i * binWidth;
    const y = height - bottomPadding - barHeight;
    const displayRange = truncateText(bin.range, 10);

    return (
      <g key={i}>
        <defs>
          <linearGradient id={`barGradient-${i}`} x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#4facfe" />
            <stop offset="100%" stopColor="#00f2fe" />
          </linearGradient>
        </defs>
        <rect
          x={x + 2}
          y={y}
          width={binWidth - 4}
          height={barHeight}
          fill={`url(#barGradient-${i})`}
          stroke="#0288d1"
          strokeWidth="1.5"
          className="bar-element"
          rx="4"
          filter="url(#shadow)"
        >
          <title>{`${bin.range}: ${bin.count}`}</title>
        </rect>
        <text
          x={x + binWidth / 2}
          y={height - bottomPadding + (shouldRotateLabels ? 20 : 15)}
          textAnchor={shouldRotateLabels ? "end" : "middle"}
          fontSize="10"
          fill="#444"
          fontWeight="500"
          transform={shouldRotateLabels ? `rotate(-45, ${x + binWidth / 2}, ${height - bottomPadding + 20})` : ''}
        >
          {displayRange}
        </text>
        {bin.count > 0 && (
          <text
            x={x + binWidth / 2}
            y={y - 8}
            textAnchor="middle"
            fontSize="11"
            fill="#222"
            fontWeight="600"
          >
            {bin.count}
          </text>
        )}
      </g>
    );
  });

  return (
    <svg width={width} height={height} className="chart-svg">
      <defs>
        <filter id="shadow">
          <feDropShadow dx="0" dy="2" stdDeviation="3" floodOpacity="0.3" />
        </filter>
      </defs>
      <rect x="0" y="0" width={width} height={height} fill="#fafafa" rx="8" />
      {createAxes([0, bins.length], [0, maxCount], label, "Frequency", false)}
      {bars}
    </svg>
  );
}

function createBarChart(categories, values, label) {
  const maxValue = Math.max(...values);
  const barWidth = (width - 2 * padding) / categories.length;
  const shouldRotateLabels = categories.length > 6 || barWidth < 50;

  const bars = categories.map((category, i) => {
    const barHeight = maxValue > 0 ? (values[i] / maxValue) * (height - bottomPadding - padding) : 0;
    const x = padding + i * barWidth;
    const y = height - bottomPadding - barHeight;
    const displayCategory = truncateText(String(category), 10);

    return (
      <g key={i}>
        <defs>
          <linearGradient id={`barChartGradient-${i}`} x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#f093fb" />
            <stop offset="100%" stopColor="#f5576c" />
          </linearGradient>
        </defs>
        <rect
          x={x + 3}
          y={y}
          width={barWidth - 6}
          height={barHeight}
          fill={`url(#barChartGradient-${i})`}
          stroke="#d81b60"
          strokeWidth="1.5"
          className="bar-element"
          rx="4"
          filter="url(#shadow)"
        >
          <title>{`${category}: ${values[i]}`}</title>
        </rect>
        <text
          x={x + barWidth / 2}
          y={height - bottomPadding + (shouldRotateLabels ? 20 : 15)}
          textAnchor={shouldRotateLabels ? "end" : "middle"}
          fontSize="10"
          fill="#444"
          fontWeight="500"
          transform={shouldRotateLabels ? `rotate(-45, ${x + barWidth / 2}, ${height - bottomPadding + 20})` : ''}
        >
          {displayCategory}
        </text>
        {values[i] > 0 && (
          <text
            x={x + barWidth / 2}
            y={y - 8}
            textAnchor="middle"
            fontSize="11"
            fill="#222"
            fontWeight="600"
          >
            {values[i]}
          </text>
        )}
      </g>
    );
  });

  return (
    <svg width={width} height={height} className="chart-svg">
      <defs>
        <filter id="shadow">
          <feDropShadow dx="0" dy="2" stdDeviation="3" floodOpacity="0.3" />
        </filter>
      </defs>
      <rect x="0" y="0" width={width} height={height} fill="#fafafa" rx="8" />
      {createAxes([0, categories.length], [0, maxValue], label, "Count", false)}
      {bars}
    </svg>
  );
}

function DataVisualizerNode({ id, data, isConnectable }) {
  const [chartType, setChartType] = useState('scatter');
  const [xColumn, setXColumn] = useState('');
  const [yColumn, setYColumn] = useState('');
  const [histogramColumn, setHistogramColumn] = useState('');
  const [barChartColumn, setBarChartColumn] = useState('');

  // Find upstream data source
  const upstreamData = useStore((store) => {
    const incoming = Array.from(store.edges.values()).filter((e) => e.target === id);
    for (const e of incoming) {
      const src = store.nodeInternals.get(e.source);
      if (src?.type === 'csvReader') {
        return {
          type: 'csv',
          headers: src.data?.headers || [],
          file: src.data?.file
        };
      }
      if (src?.type === 'dataCleaner') {
        return {
          type: 'cleaned',
          headers: src.data?.headers || [],
          cleanedRows: src.data?.cleanedRows || []
        };
      }
      if (src?.type === 'encoder') {
        return {
          type: 'encoded',
          headers: src.data?.headers || [],
          encodedRows: src.data?.encodedRows || []
        };
      }
      if (src?.type === 'normalizer') {
        return {
          type: 'normalized',
          headers: src.data?.headers || [],
          normalizedRows: src.data?.normalizedRows || []
        };
      }
      if (src?.type === 'featureSelector') {
        return {
          type: 'featureSelector',
          headers: src.data?.selectedHeaders || [],
          selectedRows: src.data?.selectedRows || []
        };
      }
    }
    return null;
  });

  const [vizData, setVizData] = useState({
    rows: [],
    headers: [],
    scatterPoints: [],
    histogramBins: [],
    barCategories: [],
    barValues: []
  });

  React.useEffect(() => {
    let cancelled = false;
    async function loadData() {
      if (!upstreamData) return;

      let rows = [];
      let headers = upstreamData.headers;

      if (upstreamData.type === 'csv' && upstreamData.file) {
        const parsed = await parseFullTabularFile(upstreamData.file);
        rows = parsed.rows;
        headers = parsed.headers;
      } else if (upstreamData.type === 'cleaned') {
        rows = upstreamData.cleanedRows;
      } else if (upstreamData.type === 'encoded') {
        rows = upstreamData.encodedRows;
      } else if (upstreamData.type === 'normalized') {
        rows = upstreamData.normalizedRows;
      } else if (upstreamData.type === 'featureSelector') {
        rows = upstreamData.selectedRows;
      }

      if (!cancelled && rows.length > 0) {
        setVizData({ rows, headers, scatterPoints: [], histogramBins: [], barCategories: [], barValues: [] });
      }
    }
    loadData();
    return () => { cancelled = true; };
  }, [upstreamData]);

  const updateVisualization = useMemo(() => {
    if (!vizData.rows.length || !vizData.headers.length) return null;

    if (chartType === 'scatter' && xColumn && yColumn) {
      const xIndex = vizData.headers.indexOf(xColumn);
      const yIndex = vizData.headers.indexOf(yColumn);

      if (xIndex === -1 || yIndex === -1) return null;

      const points = [];
      for (const row of vizData.rows) {
        const x = parseFloat(row[xIndex]);
        const y = parseFloat(row[yIndex]);
        if (Number.isFinite(x) && Number.isFinite(y)) {
          points.push([x, y]);
        }
      }

      if (points.length === 0) return null;

      const xs = points.map(p => p[0]);
      const ys = points.map(p => p[1]);
      const xDomain = [Math.min(...xs), Math.max(...xs)];
      const yDomain = [Math.min(...ys), Math.max(...ys)];

      return createScatterPlot(points, xDomain, yDomain, xColumn, yColumn);
    }

    if (chartType === 'histogram' && histogramColumn) {
      const colIndex = vizData.headers.indexOf(histogramColumn);
      if (colIndex === -1) return null;

      const values = [];
      for (const row of vizData.rows) {
        const val = parseFloat(row[colIndex]);
        if (Number.isFinite(val)) {
          values.push(val);
        }
      }

      if (values.length === 0) return null;

      const min = Math.min(...values);
      const max = Math.max(...values);
      const numBins = Math.min(10, Math.ceil(Math.sqrt(values.length)));
      const binSize = (max - min) / numBins;

      const bins = Array(numBins).fill().map((_, i) => ({
        range: `${(min + i * binSize).toFixed(1)}-${(min + (i + 1) * binSize).toFixed(1)}`,
        count: 0,
        min: min + i * binSize,
        max: min + (i + 1) * binSize
      }));

      values.forEach(val => {
        const binIndex = Math.min(Math.floor((val - min) / binSize), numBins - 1);
        bins[binIndex].count++;
      });

      return createHistogram(values, bins, histogramColumn);
    }

    if (chartType === 'bar' && barChartColumn) {
      const colIndex = vizData.headers.indexOf(barChartColumn);
      if (colIndex === -1) return null;

      const categoryCount = {};
      for (const row of vizData.rows) {
        const category = String(row[colIndex]).trim();
        if (category) {
          categoryCount[category] = (categoryCount[category] || 0) + 1;
        }
      }

      const categories = Object.keys(categoryCount);
      const values = Object.values(categoryCount);

      if (categories.length === 0) return null;

      return createBarChart(categories, values, barChartColumn);
    }

    return null;
  }, [chartType, xColumn, yColumn, histogramColumn, barChartColumn, vizData]);

  const hasData = upstreamData && vizData.headers.length > 0;

  return (
    <div className="data-visualizer-node">
      <div className="dv-title">Data Visualizer</div>

      {hasData && (
        <div className="dv-controls">
          <div className="dv-chart-type">
            <label>Chart Type:</label>
            <select value={chartType} onChange={(e) => setChartType(e.target.value)}>
              <option value="scatter">Scatter Plot</option>
              <option value="histogram">Histogram</option>
              <option value="bar">Bar Chart</option>
            </select>
          </div>

          {chartType === 'scatter' && (
            <>
              <div className="dv-column-select">
                <label>X Axis:</label>
                <select value={xColumn} onChange={(e) => setXColumn(e.target.value)}>
                  <option value="">Select column</option>
                  {vizData.headers.map((h) => (
                    <option key={h} value={h}>{h}</option>
                  ))}
                </select>
              </div>
              <div className="dv-column-select">
                <label>Y Axis:</label>
                <select value={yColumn} onChange={(e) => setYColumn(e.target.value)}>
                  <option value="">Select column</option>
                  {vizData.headers.map((h) => (
                    <option key={h} value={h}>{h}</option>
                  ))}
                </select>
              </div>
            </>
          )}

          {chartType === 'histogram' && (
            <div className="dv-column-select">
              <label>Column:</label>
              <select value={histogramColumn} onChange={(e) => setHistogramColumn(e.target.value)}>
                <option value="">Select column</option>
                {vizData.headers.map((h) => (
                  <option key={h} value={h}>{h}</option>
                ))}
              </select>
            </div>
          )}

          {chartType === 'bar' && (
            <div className="dv-column-select">
              <label>Column:</label>
              <select value={barChartColumn} onChange={(e) => setBarChartColumn(e.target.value)}>
                <option value="">Select column</option>
                {vizData.headers.map((h) => (
                  <option key={h} value={h}>{h}</option>
                ))}
              </select>
            </div>
          )}
        </div>
      )}

      <div className="dv-content">
        {updateVisualization || (
          <div className="dv-placeholder">
            {hasData ? 'Select chart type and columns to visualize' : 'Connect a data source (CSV, Cleaner, Encoder, Normalizer) to visualize data'}
          </div>
        )}
      </div>

      <Handle
        type="target"
        position={Position.Top}
        isConnectable={isConnectable}
        className="custom-handle"
        id="target-top"
      />
      <Handle
        type="target"
        position={Position.Left}
        isConnectable={isConnectable}
        className="custom-handle"
        id="target-left"
      />
    </div>
  );
}

export default DataVisualizerNode;
