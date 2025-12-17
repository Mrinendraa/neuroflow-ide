import React, { useMemo, useState, useEffect } from 'react';
import { Handle, Position, useStore, useReactFlow } from 'reactflow';
import './FeatureSelectorNode.css';
import { parseFullTabularFile } from '../../utils/parseTabularFile';

const FeatureSelectorNode = ({ id, data, isConnectable }) => {
    const { setNodes } = useReactFlow();

    const [selectedColumns, setSelectedColumns] = useState([]);
    const [filteredData, setFilteredData] = useState(null);
    const [error, setError] = useState('');

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
        }
        return null;
    });

    const headers = useMemo(() => upstreamData?.headers || [], [upstreamData]);

    // Toggle column selection
    const toggleColumn = (columnName) => {
        setSelectedColumns(prev =>
            prev.includes(columnName)
                ? prev.filter(col => col !== columnName)
                : [...prev, columnName]
        );
    };

    // Select all columns
    const selectAll = () => {
        setSelectedColumns([...headers]);
    };

    // Deselect all columns
    const deselectAll = () => {
        setSelectedColumns([]);
    };

    // Apply feature selection
    const onApply = async () => {
        if (!upstreamData) {
            setError('Please connect a data source (CSV, Data Cleaner, Encoder, or Normalizer).');
            return;
        }

        if (selectedColumns.length === 0) {
            setError('Please select at least one column.');
            return;
        }

        setError('');

        try {
            let rows;

            if (upstreamData.type === 'csv') {
                const parsed = await parseFullTabularFile(upstreamData.file);
                rows = parsed.rows;
            } else if (upstreamData.type === 'cleaned') {
                rows = upstreamData.cleanedRows;
            } else if (upstreamData.type === 'encoded') {
                rows = upstreamData.encodedRows;
            } else if (upstreamData.type === 'normalized') {
                rows = upstreamData.normalizedRows;
            } else {
                throw new Error('Unknown data source type.');
            }

            // Get indices of selected columns
            const selectedIndices = selectedColumns.map(col => headers.indexOf(col));

            // Filter rows to include only selected columns
            const filteredRows = rows.map(row =>
                selectedIndices.map(idx => row[idx])
            );

            // Get first 5 rows for preview
            const previewRows = filteredRows.slice(0, 5);

            setFilteredData({
                selectedHeaders: selectedColumns,
                selectedRows: filteredRows,
                previewRows,
                totalRows: filteredRows.length
            });

            // Store filtered data in node for downstream nodes
            setNodes((nds) => nds.map((n) => {
                if (n.id !== id) return n;
                return {
                    ...n,
                    data: {
                        ...n.data,
                        selectedHeaders: selectedColumns,
                        selectedRows: filteredRows
                    }
                };
            }));

        } catch (err) {
            setError(err?.message || 'Feature selection failed.');
            console.error('Feature selection error:', err);
        }
    };

    const onClear = () => {
        setFilteredData(null);
        setSelectedColumns([]);
        setError('');
        setNodes((nds) => nds.map((n) =>
            n.id === id ? { ...n, data: { ...n.data, selectedHeaders: [], selectedRows: [] } } : n
        ));
    };

    return (
        <div className="feature-selector-node">
            <Handle type="target" position={Position.Top} isConnectable={isConnectable} id="target-top" />
            <Handle type="target" position={Position.Left} isConnectable={isConnectable} id="target-left" />

            <div className="fs-header">
                <span className="fs-title">{data.label || 'Feature Selector'}</span>
            </div>

            {headers.length > 0 && (
                <div className="fs-content">
                    {/* Column Selection */}
                    <div className="fs-section">
                        <div className="section-header">
                            Select Features ({selectedColumns.length}/{headers.length})
                        </div>
                        <div className="fs-actions-top">
                            <button className="btn-small" onClick={selectAll}>Select All</button>
                            <button className="btn-small" onClick={deselectAll}>Clear All</button>
                        </div>
                        <div className="column-checkboxes">
                            {headers.map((header, idx) => (
                                <label key={idx} className="column-option">
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

                    {/* Action Buttons */}
                    <div className="fs-actions">
                        <button
                            className="btn primary"
                            onClick={onApply}
                            disabled={selectedColumns.length === 0}
                            style={{ flex: filteredData ? 0.6 : 1 }}
                        >
                            Apply Selection
                        </button>
                        {filteredData && (
                            <button className="btn secondary" onClick={onClear} style={{ flex: 0.4 }}>
                                Clear
                            </button>
                        )}
                    </div>

                    {error && <div className="error-text">{error}</div>}

                    {/* Preview */}
                    {filteredData && (
                        <div className="fs-preview">
                            <div className="preview-header" >
                                <span className="preview-title">Selected Data Preview</span>
                                <span className="preview-count">({filteredData.totalRows} rows total)</span>
                            </div>
                            <div className="table-scroll">
                                <table>
                                    <thead>
                                        <tr>
                                            {filteredData.selectedHeaders.map((header, idx) => (
                                                <th key={idx}>{header}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {filteredData.previewRows.map((row, rIdx) => (
                                            <tr key={rIdx}>
                                                {row.map((cell, cIdx) => (
                                                    <td key={cIdx}>
                                                        {typeof cell === 'number'
                                                            ? cell.toFixed(4)
                                                            : String(cell ?? '')
                                                        }
                                                    </td>
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

            {!headers.length && (
                <div className="fs-placeholder">
                    Connect a data source (CSV, Data Cleaner, Encoder, or Normalizer) to select features
                </div>
            )}

            <Handle type="source" position={Position.Bottom} isConnectable={isConnectable} id="source-bottom" />
            <Handle type="source" position={Position.Right} isConnectable={isConnectable} id="source-right" />
        </div>
    );
};

export default FeatureSelectorNode;
