import React, { useMemo, useState, useEffect } from 'react';
import { Handle, Position, useStore, useReactFlow } from 'reactflow';
import './LinearRegressionNode.css';
import { FaChartLine, FaCog } from 'react-icons/fa';
import { parseFullTabularFile } from '../../utils/parseTabularFile';
import { trainLinearRegression, checkApiHealth } from '../../utils/apiClient';

const LinearRegressionNode = ({ id, data, isConnectable }) => {
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [config, setConfig] = useState({
    learningRate: 0.01,
    maxIterations: 1000,
    regularization: 'none',
    alpha: 0.1
  });
  const [xCol, setXCol] = useState('');
  const [yCol, setYCol] = useState('');
  const [trainPercent, setTrainPercent] = useState(80);
  const [trainPercentInput, setTrainPercentInput] = useState('80');
  const [isTraining, setIsTraining] = useState(false);
  const [trainMsg, setTrainMsg] = useState('');
  const [modelResults, setModelResults] = useState(null);
  const [apiStatus, setApiStatus] = useState(null);
  const { setNodes } = useReactFlow();

  // Check API health on mount
  useEffect(() => {
    checkApiHealth().then(status => {
      setApiStatus(status);
    });
  }, []);

  // Inspect incoming edge to find the upstream CSV node or Encoder node
  const upstreamData = useStore((store) => {
    const incoming = Array.from(store.edges.values()).filter((e) => e.target === id);
    if (incoming.length === 0) return null;
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
      if (src?.type === 'pca') {
        return {
          type: 'pca',
          headers: src.data?.pcaHeaders || [],
          pcaRows: src.data?.pcaRows || [],
          pcaInfo: src.data?.pcaInfo || {}
        };
      }
    }
    return null;
  });

  const headers = useMemo(() => upstreamData?.headers || [], [upstreamData]);

  const toggleConfig = () => {
    setIsConfigOpen(!isConfigOpen);
  };

  const onRun = async () => {
    setTrainMsg('');
    if (!upstreamData) {
      alert('Please connect a CSV/Excel node or Encoder node.');
      return;
    }
    if (!xCol || !yCol) {
      alert('Please select both independent (X) and dependent (Y) columns.');
      return;
    }
    setIsTraining(true);
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
      } else if (upstreamData.type === 'pca') {
        // Use PCA-transformed data
        rows = upstreamData.pcaRows;
      } else {
        throw new Error('Unknown data source type.');
      }

      const xi = headers.indexOf(xCol);
      const yi = headers.indexOf(yCol);
      if (xi === -1 || yi === -1) throw new Error('Selected columns not found.');

      const X = [];
      const Y = [];
      for (const r of rows) {
        const xv = parseFloat(r[xi]);
        const yv = parseFloat(r[yi]);
        if (!Number.isFinite(xv) || !Number.isFinite(yv)) continue;
        X.push(xv);
        Y.push(yv);
      }
      if (X.length < 2) throw new Error('Not enough numeric rows for training.');

      // Call Python API
      setTrainMsg('Training linear regression model...');
      const result = await trainLinearRegression(X, Y, trainPercent, xCol, yCol);

      if (result.success) {
        setModelResults(result);
        setTrainMsg(`Training complete! Test R²: ${(result.test_metrics.r2_score * 100).toFixed(2)}%`);

        // Persist model info on this node's data for downstream nodes (visualizer)
        setNodes((nds) => nds.map((n) => {
          if (n.id !== id) return n;
          return {
            ...n,
            data: {
              ...n.data,
              model: {
                slope: result.slope,
                intercept: result.intercept,
                xCol,
                yCol,
                train_metrics: result.train_metrics,
                test_metrics: result.test_metrics
              }
            }
          };
        }));
        alert('Linear Regression training finished.');
      } else {
        throw new Error(result.error || 'Training failed');
      }
    } catch (err) {
      setTrainMsg(err?.message || 'Training failed.');
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="linear-regression-node">
      <Handle
        type="target"
        position={Position.Top}
        isConnectable={isConnectable}
        style={{ background: '#555' }}
      />

      <div className="node-header">
        <FaChartLine className="node-icon" />
        <span className="node-title">{data.label}</span>
        <button className="config-button" onClick={toggleConfig}>
          <FaCog className={`gear-icon ${isConfigOpen ? 'rotating' : ''}`} />
        </button>
      </div>

      {headers.length > 0 && (
        <div className="lr-selects">
          <div className="lr-row">
            <label>Independent (X):</label>
            <select value={xCol} onChange={(e) => setXCol(e.target.value)}>
              <option value="">Select column</option>
              {headers.map((h) => (
                <option key={h} value={h}>{h}</option>
              ))}
            </select>
          </div>
          <div className="lr-row">
            <label>Dependent (Y):</label>
            <select value={yCol} onChange={(e) => setYCol(e.target.value)}>
              <option value="">Select column</option>
              {headers.map((h) => (
                <option key={h} value={h}>{h}</option>
              ))}
            </select>
          </div>
          <div className="lr-row">
            <label>Train Data % (0-99):</label>
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <button
                type="button"
                onClick={() => {
                  if (trainPercent > 0) {
                    const newVal = trainPercent - 1;
                    setTrainPercent(newVal);
                    setTrainPercentInput(newVal.toString());
                  }
                }}
                disabled={trainPercent <= 0}
                style={{
                  width: '24px',
                  height: '24px',
                  padding: '0',
                  border: '1px solid #ccc',
                  borderRadius: '3px',
                  background: '#f8f9fa',
                  cursor: trainPercent <= 0 ? 'not-allowed' : 'pointer'
                }}
              >
                -
              </button>
              <input
                type="number"
                min="0"
                max="99"
                step="1"
                value={trainPercentInput}
                onChange={(e) => {
                  const val = e.target.value;
                  setTrainPercentInput(val);
                  const numVal = parseInt(val, 10);
                  if (!isNaN(numVal) && numVal >= 0 && numVal <= 99) {
                    setTrainPercent(numVal);
                  }
                }}
                onBlur={(e) => {
                  const numVal = parseInt(e.target.value, 10);
                  if (isNaN(numVal) || numVal < 0 || numVal > 99) {
                    setTrainPercentInput(trainPercent.toString());
                  } else {
                    setTrainPercentInput(numVal.toString());
                    setTrainPercent(numVal);
                  }
                }}
                style={{ width: '60px', padding: '4px', textAlign: 'center' }}
              />
              <button
                type="button"
                onClick={() => {
                  if (trainPercent < 99) {
                    const newVal = trainPercent + 1;
                    setTrainPercent(newVal);
                    setTrainPercentInput(newVal.toString());
                  }
                }}
                disabled={trainPercent >= 99}
                style={{
                  width: '24px',
                  height: '24px',
                  padding: '0',
                  border: '1px solid #ccc',
                  borderRadius: '3px',
                  background: '#f8f9fa',
                  cursor: trainPercent >= 99 ? 'not-allowed' : 'pointer'
                }}
              >
                +
              </button>
              <span style={{ marginLeft: '4px', fontSize: '0.85em', color: '#666' }}>
                (Test: {100 - trainPercent}%)
              </span>
            </div>
          </div>
          <div className="lr-actions">
            <button className="btn primary" onClick={onRun} disabled={isTraining || apiStatus === false}>
              {isTraining ? 'Training...' : 'Train Model'}
            </button>
          </div>
          {apiStatus === false && (
            <div style={{
              background: '#fff3cd',
              border: '1px solid #ffc107',
              padding: '4px',
              marginTop: '4px',
              fontSize: '0.8em',
              borderRadius: '3px'
            }}>
              ⚠️ API server not running
            </div>
          )}
          {trainMsg && <div className="lr-msg">{trainMsg}</div>}

          {modelResults && (
            <div className="model-results" style={{
              marginTop: '8px',
              padding: '6px',
              background: '#f8f9fa',
              border: '1px solid #dee2e6',
              borderRadius: '3px',
              fontSize: '0.85em'
            }}>
              <div style={{ fontWeight: '700', color: '#1a202c', marginBottom: '8px', paddingBottom: '4px', borderBottom: '1px solid #e1e8f0' }}>Train Metrics</div>
              <div>MSE: <strong>{modelResults.train_metrics.mse.toFixed(4)}</strong></div>
              <div>RMSE: <strong>{modelResults.train_metrics.rmse.toFixed(4)}</strong></div>
              <div>MAE: <strong>{modelResults.train_metrics.mae.toFixed(4)}</strong></div>
              <div>R² Score: <strong>{(modelResults.train_metrics.r2_score * 100).toFixed(2)}%</strong></div>
              <div>MAPE: <strong>{modelResults.train_metrics.mape.toFixed(2)}%</strong></div>

              <div style={{ fontWeight: '700', color: '#1a202c', marginTop: '12px', marginBottom: '8px', paddingBottom: '4px', borderBottom: '1px solid #e1e8f0' }}>Test Metrics</div>
              <div>MSE: <strong>{modelResults.test_metrics.mse.toFixed(4)}</strong></div>
              <div>RMSE: <strong>{modelResults.test_metrics.rmse.toFixed(4)}</strong></div>
              <div>MAE: <strong>{modelResults.test_metrics.mae.toFixed(4)}</strong></div>
              <div>R² Score: <strong>{(modelResults.test_metrics.r2_score * 100).toFixed(2)}%</strong></div>
              <div>MAPE: <strong>{modelResults.test_metrics.mape.toFixed(2)}%</strong></div>

              <div style={{ marginTop: '12px', paddingTop: '8px', borderTop: '1px solid #e1e8f0', fontSize: '0.8rem', color: '#718096' }}>
                <div>Train Size: <strong>{modelResults.train_size}</strong> | Test Size: <strong>{modelResults.test_size}</strong></div>
              </div>

              {/* Equation with hover tooltip */}
              <div className="equation-container">
                <div className="equation-display">
                  y = {modelResults.slope.toFixed(4)}x + {modelResults.intercept.toFixed(4)}
                </div>
                <div className="equation-tooltip">
                  <div className="equation-tooltip-content">
                    y = {modelResults.slope.toFixed(4)}x + {modelResults.intercept.toFixed(4)}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {isConfigOpen && (
        <div className="config-panel">
          <div className="config-section">
            <label>Learning Rate:</label>
            <input
              type="number"
              value={config.learningRate}
              onChange={(e) => setConfig({ ...config, learningRate: parseFloat(e.target.value) })}
              step="0.001"
              min="0"
              max="1"
            />
          </div>

          <div className="config-section">
            <label>Max Iterations:</label>
            <input
              type="number"
              value={config.maxIterations}
              onChange={(e) => setConfig({ ...config, maxIterations: parseInt(e.target.value) })}
              min="1"
              max="10000"
            />
          </div>

          <div className="config-section">
            <label>Regularization:</label>
            <select
              value={config.regularization}
              onChange={(e) => setConfig({ ...config, regularization: e.target.value })}
            >
              <option value="none">None</option>
              <option value="l1">L1 (Lasso)</option>
              <option value="l2">L2 (Ridge)</option>
              <option value="elastic">Elastic Net</option>
            </select>
          </div>

          {config.regularization !== 'none' && (
            <div className="config-section">
              <label>Alpha:</label>
              <input
                type="number"
                value={config.alpha}
                onChange={(e) => setConfig({ ...config, alpha: parseFloat(e.target.value) })}
                step="0.01"
                min="0"
                max="1"
              />
            </div>
          )}
        </div>
      )}

      <Handle
        type="source"
        position={Position.Bottom}
        isConnectable={isConnectable}
        style={{ background: '#555' }}
      />
    </div>
  );
};

export default LinearRegressionNode;