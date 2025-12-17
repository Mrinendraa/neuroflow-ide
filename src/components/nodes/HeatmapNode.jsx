import React, { useMemo, useState, useEffect } from 'react';
import { Handle, Position, useStore } from 'reactflow';
import './HeatmapNode.css';

// Utility to calculate Pearson correlation matrix
const calculateCorrelationMatrix = (rows, headers) => {
  if (!rows || rows.length === 0 || !headers) return null;

  // 1. Identify numeric columns
  const numericIndices = headers.map((h, i) => {
    const isNum = rows.every(r => {
      const val = r[i];
      return typeof val === 'number' || (typeof val === 'string' && !isNaN(parseFloat(val)) && val.trim() !== '');
    });
    return isNum ? i : -1;
  }).filter(i => i !== -1);

  if (numericIndices.length < 2) return null; // Need at least 2 columns for correlation

  const numericHeaders = numericIndices.map(i => headers[i]);
  const n = rows.length;
  const numCols = numericIndices.length;

  // 2. Extract numeric data
  const data = rows.map(row => numericIndices.map(i => parseFloat(row[i])));

  // 3. Calculate means
  const means = Array(numCols).fill(0);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < numCols; c++) {
      means[c] += data[r][c];
    }
  }
  for (let c = 0; c < numCols; c++) {
    means[c] /= n;
  }

  // 4. Calculate covariance and variances
  const matrix = Array.from({ length: numCols }, () => Array(numCols).fill(0));
  const stdDevs = Array(numCols).fill(0);

  for (let r = 0; r < n; r++) {
    for (let c = 0; c < numCols; c++) {
      const diff = data[r][c] - means[c];
      stdDevs[c] += diff * diff;
      for (let k = 0; k < numCols; k++) { // Optimization: Symmetric
        if (k >= c) {
          const diffK = data[r][k] - means[k];
          matrix[c][k] += diff * diffK;
        }
      }
    }
  }

  // Finalize std devs
  for (let c = 0; c < numCols; c++) {
    stdDevs[c] = Math.sqrt(stdDevs[c]);
  }

  // Finalize correlation matrix
  for (let c = 0; c < numCols; c++) {
    for (let k = c; k < numCols; k++) {
      const denom = stdDevs[c] * stdDevs[k];
      const val = denom === 0 ? 0 : matrix[c][k] / denom;
      matrix[c][k] = val;
      matrix[k][c] = val; // Symmetric
    }
  }

  return { headers: numericHeaders, matrix };
};

const HeatmapGrid = ({ title, data }) => {
  if (!data) return <div className="heatmap-placeholder">No numeric data for {title}</div>;

  const { headers, matrix } = data;
  const size = headers.length;
  const cellSize = 30; // px

  // Color scale: -1 (Blue) -> 0 (White) -> 1 (Red)
  const getColor = (val) => {
    if (val >= 0) {
      // White to Red
      const intensity = Math.round(val * 255);
      return `rgb(255, ${255 - intensity}, ${255 - intensity})`;
    } else {
      // White to Blue
      const intensity = Math.round(Math.abs(val) * 255);
      return `rgb(${255 - intensity}, ${255 - intensity}, 255)`;
    }
  };

  return (
    <div className="heatmap-container">
      <div className="heatmap-title">{title}</div>
      <div className="heatmap-grid-wrapper">
        <div
          className="heatmap-grid"
          style={{
            gridTemplateColumns: `repeat(${size}, ${cellSize}px)`,
            width: 'fit-content'
          }}
        >
          {matrix.map((row, r) =>
            row.map((val, c) => (
              <div
                key={`${r}-${c}`}
                className="heatmap-cell"
                style={{
                  width: cellSize,
                  height: cellSize,
                  backgroundColor: getColor(val),
                }}
                title={`${headers[r]} vs ${headers[c]}: ${val.toFixed(2)}`}
              >
                {/* Optional: Show value if cell is large enough, or just on hover */}
              </div>
            ))
          )}
        </div>
      </div>
      <div className="heatmap-legend">
        <span>-1</span>
        <div className="legend-gradient"></div>
        <span>1</span>
      </div>
    </div>
  );
};

const HeatmapNode = ({ id, data, isConnectable }) => {
  const [originalData, setOriginalData] = useState(null);
  const [cleanedData, setCleanedData] = useState(null);

  // Find upstream data sources
  const upstreamNodes = useStore((store) => {
    const edges = Array.from(store.edges.values()).filter((e) => e.target === id);
    return edges.map(e => store.nodeInternals.get(e.source)).filter(Boolean);
  });

  useEffect(() => {
    let original = null;
    let cleaned = null;

    upstreamNodes.forEach(node => {
      if (node.type === 'csvReader' || node.type === 'excelReader') {
        if (node.data?.rows) {
          original = { rows: node.data.rows, headers: node.data.headers };
        }
      } else if (['dataCleaner', 'normalizer', 'encoder'].includes(node.type)) {
        // Try to find the processed data
        const rows = node.data?.cleanedRows || node.data?.normalizedRows || node.data?.encodedRows;
        const headers = node.data?.headers;
        if (rows && headers) {
          cleaned = { rows, headers };
        }
      }
    });

    setOriginalData(original);
    setCleanedData(cleaned);
  }, [upstreamNodes]);

  const originalCorr = useMemo(() =>
    originalData ? calculateCorrelationMatrix(originalData.rows, originalData.headers) : null,
    [originalData]
  );

  const cleanedCorr = useMemo(() =>
    cleanedData ? calculateCorrelationMatrix(cleanedData.rows, cleanedData.headers) : null,
    [cleanedData]
  );

  return (
    <div className="heatmap-node">
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} />

      <div className="heatmap-header">
        <span>Correlation Heatmap</span>
      </div>

      <div className="heatmap-content">
        {!originalData && !cleanedData && (
          <div className="heatmap-placeholder">
            Connect a Data Source (Original) and/or a Processor (Cleaned) to view correlation comparison.
          </div>
        )}

        {(originalData || cleanedData) && (
          <div className="heatmap-comparison">
            {originalData && <HeatmapGrid title="Original Data" data={originalCorr} />}
            {cleanedData && <HeatmapGrid title="Cleaned Data" data={cleanedCorr} />}
          </div>
        )}
      </div>

      <Handle type="source" position={Position.Bottom} isConnectable={isConnectable} />
    </div>
  );
};

export default HeatmapNode;
