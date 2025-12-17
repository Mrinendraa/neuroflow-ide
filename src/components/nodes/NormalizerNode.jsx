import React, { useMemo, useState } from 'react';
import { Handle, Position, useStore, useReactFlow } from 'reactflow';
import './NormalizerNode.css';
import { parseFullTabularFile } from '../../utils/parseTabularFile';
import { normalizeDataset } from '../../utils/normalizationUtils';

const NormalizerNode = ({ id, data, isConnectable }) => {
  const [selectedColumns, setSelectedColumns] = useState([]);
  const [normalizationType, setNormalizationType] = useState('minmax');
  const [isProcessing, setIsProcessing] = useState(false);
  const [normalizedData, setNormalizedData] = useState(null);
  const [error, setError] = useState('');
  const { setNodes } = useReactFlow();

  // Find upstream CSV or Encoder node
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
          encodedRows: src.data?.encodedRows || [],
          encodingInfo: src.data?.encodingInfo || {}
        };
      }
      if (src?.type === 'normalizer') {
        return {
          type: 'normalized',
          headers: src.data?.headers || [],
          normalizedRows: src.data?.normalizedRows || [],
          normalizationInfo: src.data?.normalizationInfo || {}
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

  const headers = useMemo(() => upstreamData?.headers || [], [upstreamData]);

  const toggleColumn = (header) => {
    setSelectedColumns(prev =>
      prev.includes(header)
        ? prev.filter(col => col !== header)
        : [...prev, header]
    );
  };

  const onNormalize = async () => {
    if (!upstreamData) {
      setError('Please connect a CSV/Excel node, Data Cleaner, Encoder node, Feature Selector, or Normalizer node.');
      return;
    }
    if (selectedColumns.length === 0) {
      setError('Please select at least one column to normalize.');
      return;
    }

    setIsProcessing(true);
    setError('');

    try {
      let rows;

      if (upstreamData.type === 'csv') {
        // Parse from CSV file
        const parsed = await parseFullTabularFile(upstreamData.file);
        rows = parsed.rows;
      } else if (upstreamData.type === 'cleaned') {
        // Use cleaned data
        rows = upstreamData.cleanedRows;
      } else if (upstreamData.type === 'encoded') {
        // Use pre-encoded data
        rows = upstreamData.encodedRows;
      } else if (upstreamData.type === 'normalized') {
        // Use pre-normalized data
        rows = upstreamData.normalizedRows;
      } else if (upstreamData.type === 'featureSelector') {
        // Use feature-selected data
        rows = upstreamData.selectedRows;
      } else {
        throw new Error('Unknown data source type.');
      }

      // Create normalization configuration
      const normalizationConfig = {};
      selectedColumns.forEach(col => {
        normalizationConfig[col] = { type: normalizationType };
      });

      // Apply normalization
      const result = normalizeDataset(rows, headers, normalizationConfig);

      // Get first 5 rows for preview
      const previewRows = result.normalizedRows.slice(0, 5);

      setNormalizedData({
        headers: result.headers,
        rows: result.normalizedRows,
        previewRows,
        normalizationInfo: result.normalizationInfo
      });

      // Store normalized data in node for downstream nodes
      setNodes((nds) => nds.map((n) => {
        if (n.id !== id) return n;
        return {
          ...n,
          data: {
            ...n.data,
            headers: result.headers,
            normalizedRows: result.normalizedRows,
            normalizationInfo: result.normalizationInfo,
            originalData: upstreamData
          }
        };
      }));

    } catch (err) {
      setError(err?.message || 'Normalization failed.');
    } finally {
      setIsProcessing(false);
    }
  };

  const onClear = () => {
    setNormalizedData(null);
    setSelectedColumns([]);
    setError('');
    setNodes((nds) => nds.map((n) =>
      n.id === id ? { ...n, data: { ...n.data, headers: [], normalizedRows: [], normalizationInfo: {} } } : n
    ));
  };

  return (
    <div className="normalizer-node">
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} />

      <div className="normalizer-header">
        <span className="normalizer-title">{data.label || 'Normalizer'}</span>
      </div>

      {headers.length > 0 && (
        <div className="normalizer-content">
          <div className="normalization-type-section">
            <label>Normalization Type:</label>
            <select value={normalizationType} onChange={(e) => setNormalizationType(e.target.value)}>
              <option value="minmax">Min-Max Normalization (0-1)</option>
              <option value="zscore">Z-Score Normalization (μ=0, σ=1)</option>
            </select>
          </div>

          <div className="columns-section">
            <label>Select Columns to Normalize:</label>
            <div className="column-checkboxes">
              {headers.map((header) => (
                <label key={header} className="column-option">
                  <input
                    type="checkbox"
                    checked={selectedColumns.includes(header)}
                    onChange={() => toggleColumn(header)}
                  />
                  <span>{header}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="normalizer-actions">
            <button
              className="btn primary"
              onClick={onNormalize}
              disabled={isProcessing || selectedColumns.length === 0}
            >
              {isProcessing ? 'Normalizing...' : 'Normalize'}
            </button>
            {normalizedData && (
              <button className="btn secondary" onClick={onClear}>
                Clear
              </button>
            )}
          </div>

          {error && <div className="error-text">{error}</div>}

          {normalizedData && (
            <div className="normalized-preview">
              <div className="preview-title">Normalized Data Preview (first 5 rows)</div>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      {normalizedData.headers.map((header, idx) => (
                        <th key={idx}>
                          {header}
                          {normalizedData.normalizationInfo[header] && (
                            <span className="normalization-badge">
                              {normalizedData.normalizationInfo[header].type}
                            </span>
                          )}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {normalizedData.previewRows.map((row, rIdx) => (
                      <tr key={rIdx}>
                        {normalizedData.headers.map((_, cIdx) => (
                          <td key={cIdx}>
                            {typeof row[cIdx] === 'number'
                              ? row[cIdx].toFixed(4)
                              : String(row[cIdx] ?? '')
                            }
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Show normalization statistics */}
              <div className="normalization-stats">
                <div className="stats-title">Normalization Statistics:</div>
                {Object.entries(normalizedData.normalizationInfo).map(([colName, info]) => (
                  <div key={colName} className="stat-item">
                    <strong>{colName}</strong> ({info.type}):
                    {info.type === 'minmax' && (
                      <span> Min: {info.min.toFixed(2)}, Max: {info.max.toFixed(2)}</span>
                    )}
                    {info.type === 'zscore' && (
                      <span> Mean: {info.mean.toFixed(2)}, Std: {info.stdDev.toFixed(2)}</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      <Handle type="source" position={Position.Bottom} isConnectable={isConnectable} />
    </div>
  );
};

export default NormalizerNode;
