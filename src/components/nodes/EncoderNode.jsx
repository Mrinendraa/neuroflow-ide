import React, { useMemo, useState } from 'react';
import { Handle, Position, useStore, useReactFlow } from 'reactflow';
import './EncoderNode.css';
import { parseFullTabularFile } from '../../utils/parseTabularFile';
import { encodeDataset } from '../../utils/encodingUtils';

const EncoderNode = ({ id, data, isConnectable }) => {
  const [selectedColumns, setSelectedColumns] = useState([]);
  const [encodingType, setEncodingType] = useState('label');
  const [isProcessing, setIsProcessing] = useState(false);
  const [encodedData, setEncodedData] = useState(null);
  const [error, setError] = useState('');
  const { setNodes } = useReactFlow();

  // Find upstream CSV, Encoder, or Normalizer node
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

  const onEncode = async () => {
    if (!upstreamData) {
      setError('Please connect a CSV/Excel node, Data Cleaner, Encoder node, or Normalizer node.');
      return;
    }
    if (selectedColumns.length === 0) {
      setError('Please select at least one column to encode.');
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

      // Create encoding configuration
      const encodingConfig = {};
      selectedColumns.forEach(col => {
        encodingConfig[col] = { type: encodingType };
      });

      // Apply encoding
      const result = encodeDataset(rows, headers, encodingConfig);

      // Get first 5 rows for preview
      const previewRows = result.encodedRows.slice(0, 5);

      setEncodedData({
        headers: result.headers,
        rows: result.encodedRows,
        previewRows,
        encodingInfo: result.encodingInfo
      });

      // Store encoded data in node for downstream nodes
      setNodes((nds) => nds.map((n) => {
        if (n.id !== id) return n;
        return {
          ...n,
          data: {
            ...n.data,
            headers: result.headers,
            encodedRows: result.encodedRows,
            encodingInfo: result.encodingInfo,
            originalData: upstreamData
          }
        };
      }));

    } catch (err) {
      setError(err?.message || 'Encoding failed.');
    } finally {
      setIsProcessing(false);
    }
  };

  const onClear = () => {
    setEncodedData(null);
    setSelectedColumns([]);
    setError('');
    setNodes((nds) => nds.map((n) =>
      n.id === id ? { ...n, data: { ...n.data, headers: [], encodedRows: [], encodingInfo: {}, originalData: null } } : n
    ));
  };

  return (
    <div className="encoder-node">
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} />

      <div className="encoder-header">
        <span className="encoder-title">{data.label || 'Encoder'}</span>
      </div>

      {headers.length > 0 && (
        <div className="encoder-content">
          <div className="encoding-type-section">
            <label>Encoding Type:</label>
            <select value={encodingType} onChange={(e) => setEncodingType(e.target.value)}>
              <option value="label">Label Encoding</option>
              <option value="frequency">Frequency Encoding</option>
            </select>
          </div>

          <div className="columns-section">
            <label>Select Columns to Encode:</label>
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

          <div className="encoder-actions">
            <button
              className="btn primary"
              onClick={onEncode}
              disabled={isProcessing || selectedColumns.length === 0}
            >
              {isProcessing ? 'Encoding...' : 'Encode'}
            </button>
            {encodedData && (
              <button className="btn secondary" onClick={onClear}>
                Clear
              </button>
            )}
          </div>

          {error && <div className="error-text">{error}</div>}

          {encodedData && (
            <div className="encoded-preview">
              <div className="preview-title">Encoded Data Preview (first 5 rows)</div>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      {encodedData.headers.map((header, idx) => (
                        <th key={idx}>
                          {header}
                          {encodedData.encodingInfo[header] && (
                            <span className="encoding-badge">
                              {encodedData.encodingInfo[header].type}
                            </span>
                          )}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {encodedData.previewRows.map((row, rIdx) => (
                      <tr key={rIdx}>
                        {encodedData.headers.map((_, cIdx) => (
                          <td key={cIdx}>{String(row[cIdx] ?? '')}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      <Handle type="source" position={Position.Bottom} isConnectable={isConnectable} />
    </div>
  );
};

export default EncoderNode;


