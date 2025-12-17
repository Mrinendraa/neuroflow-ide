import React, { useMemo, useRef, useState } from 'react';
import { Handle, Position, useStore, useReactFlow } from 'reactflow';
import './CsvReaderNode.css';
import { parseTabularFile } from '../../utils/parseTabularFile';
// We'll remove the icon for this simpler design or keep it subtle
// import { FaTable } from 'react-icons/fa';

function CsvReaderNode({ id, data }) {
  const inputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [sample, setSample] = useState({ headers: [], rows: [] });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const { setNodes } = useReactFlow();

  const label = useMemo(() => data.label || 'CSV/Excel Reader', [data.label]);

  // Determine if this node has an incoming edge from a Start node
  const allowUpload = useStore((store) => {
    const incoming = Array.from(store.edges.values()).filter((e) => e.target === id);
    if (incoming.length === 0) return false;
    return incoming.some((e) => {
      const src = store.nodeInternals.get(e.source);
      return src?.type === 'start';
    });
  });

  const onPickFile = () => {
    setError('');
    if (inputRef.current) inputRef.current.click();
  };

  const onFileChange = async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    setSelectedFile(file);
    setIsLoading(true);
    setError('');
    try {
      const parsed = await parseTabularFile(file);
      setSample(parsed);
      // persist headers and file on the node data so downstream nodes can use it
      setNodes((nds) => nds.map((n) => {
        if (n.id !== id) return n;
        return { ...n, data: { ...n.data, headers: parsed.headers, file } };
      }));
    } catch (err) {
      setError(err?.message || 'Failed to parse file');
      setSelectedFile(null);
      setSample({ headers: [], rows: [] });
      setNodes((nds) => nds.map((n) => (n.id === id ? { ...n, data: { ...n.data, headers: [], file: undefined } } : n)));
    } finally {
      setIsLoading(false);
    }
  };

  const onDeleteFile = () => {
    setSelectedFile(null);
    setSample({ headers: [], rows: [] });
    setError('');
    if (inputRef.current) {
      inputRef.current.value = '';
    }
    setNodes((nds) => nds.map((n) => (n.id === id ? { ...n, data: { ...n.data, headers: [], file: undefined } } : n)));
  };

  return (
    <div className="csv-reader-node minimal">
      <div className="node-content">
        <div className="node-label">{label}</div>

        {allowUpload && !selectedFile && (
          <button className="btn primary" onClick={onPickFile} disabled={isLoading}>
            {isLoading ? 'Loading…' : 'Upload CSV/Excel'}
          </button>
        )}

        {allowUpload && (
          <input
            ref={inputRef}
            type="file"
            accept=".csv,.tsv,.xls,.xlsx,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,text/csv,text/tab-separated-values"
            style={{ display: 'none' }}
            onChange={onFileChange}
          />
        )}

        {allowUpload && selectedFile && (
          <div className="file-info">
            <div className="file-name" title={selectedFile.name}>{selectedFile.name}</div>
            <div className="file-actions">
              <button className="btn" onClick={onPickFile} disabled={isLoading}>Replace</button>
              <button className="btn danger" onClick={onDeleteFile} disabled={isLoading}>Delete</button>
            </div>
          </div>
        )}

        {allowUpload && error && <div className="error-text">{error}</div>}

        {allowUpload && sample?.headers?.length > 0 && (
          <div className="sample-table">
            <div className="sample-title">Sample (first 5 rows)</div>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    {sample.headers.map((h, idx) => (
                      <th key={idx}>{String(h || '')}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sample.rows.map((row, rIdx) => (
                    <tr key={rIdx}>
                      {sample.headers.map((_, cIdx) => (
                        <td key={cIdx}>{String(row?.[cIdx] ?? '')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Input handles */}
      <Handle type="target" position={Position.Top} className="custom-handle" id="a" />
      <Handle type="target" position={Position.Left} className="custom-handle" id="b" />

      {/* Output handles */}
      <Handle type="source" position={Position.Bottom} className="custom-handle" id="c" />
      <Handle type="source" position={Position.Right} className="custom-handle" id="d" />
    </div>
  );
}

export default CsvReaderNode;