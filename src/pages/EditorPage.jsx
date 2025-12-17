import React, { useState, useCallback, useRef, useEffect } from 'react';
import ReactFlow, {
  MiniMap,
  Background,
  addEdge,
  applyNodeChanges,
  applyEdgeChanges,
  useReactFlow,
  SmoothStepEdge,
  useViewport,
} from 'reactflow';
import 'reactflow/dist/style.css';

import TopToolbar from '../components/ui/TopToolbar';
import BottomToolbar from '../components/ui/BottomToolbar';
import Sidebar from '../components/ui/Sidebar';
import ContextMenu from '../components/ui/ContextMenu';
import CsvReaderNode from '../components/nodes/CsvReaderNode';
import LinearRegressionNode from '../components/nodes/LinearRegressionNode';
import MultiLinearRegressionNode from '../components/nodes/MultiLinearRegressionNode';
import PolynomialRegressionNode from '../components/nodes/PolynomialRegressionNode';
import KNNRegressionNode from '../components/nodes/KNNRegressionNode';
import KNNClassificationNode from '../components/nodes/KNNClassificationNode';
import DataCleanerNode from '../components/nodes/DataCleanerNode';
import BasicNode from '../components/nodes/BasicNode';
import StartNode from '../components/nodes/StartNode';
import ModelVisualizerNode from '../components/nodes/ModelVisualizerNode';
import EncoderNode from '../components/nodes/EncoderNode';
import NormalizerNode from '../components/nodes/NormalizerNode';
import LogisticRegressionNode from '../components/nodes/LogisticRegressionNode';
import DataVisualizerNode from '../components/nodes/DataVisualizerNode';
import ModelEvaluatorNode from '../components/nodes/ModelEvaluatorNode';
import HeatmapNode from '../components/nodes/HeatmapNode';
import FeatureSelectorNode from '../components/nodes/FeatureSelectorNode';
import PCANode from '../components/nodes/PCANode';
import FloatingEdge from '../components/edges/FloatingEdge';
import './EditorPage.css';

const nodeTypes = {
  // Existing specialized nodes
  start: StartNode,
  csvReader: CsvReaderNode,
  linearRegression: LinearRegressionNode,
  multiLinearRegression: MultiLinearRegressionNode,
  polynomialRegression: PolynomialRegressionNode,
  knnRegression: KNNRegressionNode,
  knnClassification: KNNClassificationNode,
  dataCleaner: DataCleanerNode,
  modelVisualizer: ModelVisualizerNode,
  encoder: EncoderNode,
  normalizer: NormalizerNode,
  logisticRegression: LogisticRegressionNode,
  dataVisualizer: DataVisualizerNode,
  heatmap: HeatmapNode,
  featureSelector: FeatureSelectorNode,
  pca: PCANode,
  // Generic/basic nodes for all other sidebar items
  excelReader: BasicNode,
  databaseReader: BasicNode,
  heatmap: HeatmapNode,
  ridgeRegression: BasicNode,
  lassoRegression: BasicNode,
  kMeans: BasicNode,
  hierarchicalClustering: BasicNode,
  dbscan: BasicNode,
  mlp: BasicNode,
  cnn: BasicNode,
  rnn: BasicNode,
  transformer: BasicNode,
  evaluator: ModelEvaluatorNode,
  visualizer: BasicNode,
  exporter: BasicNode,
};
const edgeTypes = {
  floating: FloatingEdge,
};

let id = 1;
const getId = () => `node_${id++}`; // Corrected this line

const EditorPage = () => {
  const reactFlowWrapper = useRef(null);
  const [nodes, setNodes] = useState([
    {
      id: 'node_0',
      type: 'start',
      position: { x: 300, y: 300 },
      data: { label: 'Start' },
    },
  ]);
  const [edges, setEdges] = useState([]);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);

  const [activeTool, setActiveTool] = useState('select');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarClosing, setSidebarClosing] = useState(false);

  // Context menu state
  const [contextMenu, setContextMenu] = useState(null);
  const [edgeContextMenu, setEdgeContextMenu] = useState(null);

  // Undo/Redo history
  const historyRef = useRef([]);
  const historyIndexRef = useRef(-1);
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const saveTimeoutRef = useRef(null);

  const { zoomIn, zoomOut, fitView } = useReactFlow();
  const { zoom } = useViewport();

  // Initialize history with initial state (only once on mount)
  const initializedRef = useRef(false);
  useEffect(() => {
    if (!initializedRef.current) {
      const initialNodes = [
        {
          id: 'node_0',
          type: 'start',
          position: { x: 300, y: 300 },
          data: { label: 'Start' },
        },
      ];
      historyRef.current = [{
        nodes: JSON.parse(JSON.stringify(initialNodes)),
        edges: JSON.parse(JSON.stringify([])),
      }];
      historyIndexRef.current = 0;
      initializedRef.current = true;
    }
  }, []);

  // Save state to history (with optional debounce)
  const saveToHistory = useCallback((nodesState, edgesState, debounce = false) => {
    // Clear existing timeout if debouncing
    if (debounce && saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }

    const saveHistory = () => {
      const currentIndex = historyIndexRef.current;
      // Remove any history after current index (when undoing and then making new changes)
      historyRef.current = historyRef.current.slice(0, currentIndex + 1);

      // Add new state to history
      historyRef.current.push({
        nodes: JSON.parse(JSON.stringify(nodesState)),
        edges: JSON.parse(JSON.stringify(edgesState)),
      });

      // Limit history to 50 states
      if (historyRef.current.length > 50) {
        historyRef.current.shift();
      } else {
        historyIndexRef.current = historyRef.current.length - 1;
      }

      // Update undo/redo button states
      setCanUndo(historyIndexRef.current > 0);
      setCanRedo(historyIndexRef.current < historyRef.current.length - 1);
    };

    if (debounce) {
      saveTimeoutRef.current = setTimeout(saveHistory, 300);
    } else {
      saveHistory();
    }
  }, []);

  // Undo function
  const handleUndo = useCallback(() => {
    if (historyIndexRef.current > 0) {
      historyIndexRef.current -= 1;
      const previousState = historyRef.current[historyIndexRef.current];
      setNodes(previousState.nodes);
      setEdges(previousState.edges);
      setCanUndo(historyIndexRef.current > 0);
      setCanRedo(true);
    }
  }, []);

  // Redo function
  const handleRedo = useCallback(() => {
    if (historyIndexRef.current < historyRef.current.length - 1) {
      historyIndexRef.current += 1;
      const nextState = historyRef.current[historyIndexRef.current];
      setNodes(nextState.nodes);
      setEdges(nextState.edges);
      setCanUndo(true);
      setCanRedo(historyIndexRef.current < historyRef.current.length - 1);
    }
  }, []);

  // Keyboard shortcuts for undo/redo
  useEffect(() => {
    const handleKeyDown = (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'z' && !event.shiftKey) {
        event.preventDefault();
        handleUndo();
      } else if ((event.ctrlKey || event.metaKey) && (event.key === 'y' || (event.key === 'z' && event.shiftKey))) {
        event.preventDefault();
        handleRedo();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleUndo, handleRedo]);

  const isValidConnection = useCallback((connection) => {
    if (connection.source === connection.target) return false;
    return true;
  }, []);

  const nodeColors = {
    // Input
    csvReader: '#f59e0b',
    excelReader: '#f59e0b',
    databaseReader: '#f59e0b',
    start: '#64748b',

    // Regression
    linearRegression: '#5a67d8',
    multiLinearRegression: '#5a67d8',
    knnRegression: '#5a67d8',
    ridgeRegression: '#5a67d8',
    lassoRegression: '#5a67d8',
    polynomialRegression: '#5a67d8',

    // Classification / Advanced Models
    logisticRegression: '#FF0080',
    knnClassification: '#FF0080',
    kMeans: '#FF0080',
    hierarchicalClustering: '#FF0080',
    dbscan: '#FF0080',
    mlp: '#FF0080',
    cnn: '#FF0080',
    rnn: '#FF0080',
    transformer: '#FF0080',

    // Processing
    dataCleaner: '#00b09b',
    normalizer: '#00b09b',
    encoder: '#00b09b',
    pca: '#8b5cf6',
    featureSelector: '#00b09b',
    heatmap: '#00b09b',

    // Visualization
    modelVisualizer: '#fda085',
    dataVisualizer: '#fda085',
    visualizer: '#fda085',

    // Evaluation
    evaluator: '#f5576c',

    // Default
    default: '#6a1b9a'
  };

  const onConnect = useCallback((params) => {
    setEdges((eds) => {
      // Find the source node to determine its type
      const sourceNode = nodes.find(n => n.id === params.source);
      const nodeType = sourceNode?.type || 'default';
      const edgeColor = nodeColors[nodeType] || nodeColors.default;

      const newEdge = {
        ...params,
        type: 'floating', // Ensure we keep the floating edge type
        style: { stroke: edgeColor, strokeWidth: 2 },
        markerEnd: { type: 'arrowclosed', color: edgeColor },
      };

      const newEdges = addEdge(newEdge, eds);
      // Save to history after a short delay
      setTimeout(() => saveToHistory(nodes, newEdges), 100);
      return newEdges;
    });
  }, [nodes, saveToHistory]);

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();
      const type = event.dataTransfer.getData('application/reactflow');
      const nodeName = event.dataTransfer.getData('application/reactflow-name');
      if (typeof type === 'undefined' || !type) return;

      const position = reactFlowInstance.screenToFlowPosition({ x: event.clientX, y: event.clientY });
      const newNode = {
        id: getId(),
        type,
        position,
        data: {
          label: nodeName || `New Node`,
          nodeType: type
        }
      };
      setNodes((nds) => {
        const newNodes = nds.concat(newNode);
        setTimeout(() => saveToHistory(newNodes, edges), 100);
        return newNodes;
      });
    },
    [reactFlowInstance, edges, saveToHistory]
  );

  const addDefaultNode = useCallback(() => {
    const newNode = {
      id: getId(),
      type: 'csvReader',
      position: { x: 50 + Math.random() * 400, y: 50 + Math.random() * 400 },
      data: { label: `New Node ${id}` },
    };
    setNodes((nds) => {
      const newNodes = nds.concat(newNode);
      setTimeout(() => saveToHistory(newNodes, edges), 100);
      return newNodes;
    });
    fitView();
  }, [fitView, edges, saveToHistory]);

  const handleMenuClick = useCallback(() => {
    if (sidebarOpen) {
      setSidebarClosing(true);
      setTimeout(() => {
        setSidebarOpen(false);
        setSidebarClosing(false);
      }, 300); // Match animation duration
    } else {
      setSidebarOpen(true);
    }
  }, [sidebarOpen]);

  // Handle node context menu (right-click)
  const onNodeContextMenu = useCallback((event, node) => {
    event.preventDefault();
    // Don't show context menu for start node
    if (node.type === 'start') {
      return;
    }

    setContextMenu({
      x: event.clientX,
      y: event.clientY,
      nodeId: node.id,
      nodeType: node.type,
    });
  }, []);

  // Handle edge context menu (right-click)
  const onEdgeContextMenu = useCallback((event, edge) => {
    event.preventDefault();
    setEdgeContextMenu({
      x: event.clientX,
      y: event.clientY,
      edgeId: edge.id,
    });
  }, []);

  // Close context menu when clicking outside
  useEffect(() => {
    const handleClick = () => {
      setContextMenu(null);
      setEdgeContextMenu(null);
    };

    if (contextMenu || edgeContextMenu) {
      document.addEventListener('click', handleClick);
      return () => document.removeEventListener('click', handleClick);
    }
  }, [contextMenu, edgeContextMenu]);

  // Delete node function
  const handleDeleteNode = useCallback((nodeId) => {
    setNodes((nds) => {
      const newNodes = nds.filter((node) => node.id !== nodeId);
      // Also remove connected edges
      setEdges((eds) => {
        const newEdges = eds.filter(
          (edge) => edge.source !== nodeId && edge.target !== nodeId
        );
        setTimeout(() => saveToHistory(newNodes, newEdges), 100);
        return newEdges;
      });
      return newNodes;
    });
    setContextMenu(null);
  }, [saveToHistory]);

  // Delete edge function
  const handleDeleteEdge = useCallback((edgeId) => {
    setEdges((eds) => {
      const newEdges = eds.filter((edge) => edge.id !== edgeId);
      setTimeout(() => saveToHistory(nodes, newEdges), 100);
      return newEdges;
    });
    setEdgeContextMenu(null);
  }, [nodes, saveToHistory]);

  return (
    <div className={`editor-container-new ${activeTool === 'pan' ? 'pan-active' : ''}`}>
      <TopToolbar activeTool={activeTool} setActiveTool={setActiveTool} onMenuClick={handleMenuClick} />
      {sidebarOpen && <Sidebar className={sidebarClosing ? 'slide-out' : ''} />}

      <div className="canvas-area">
        <div className="reactflow-wrapper-new" ref={reactFlowWrapper}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={(changes) => {
              setNodes((nds) => {
                // Filter out remove changes for start node
                const filteredChanges = changes.filter(change => {
                  if (change.type === 'remove') {
                    const node = nds.find(n => n.id === change.id);
                    if (node && node.type === 'start') {
                      return false;
                    }
                  }
                  return true;
                });

                const newNodes = applyNodeChanges(filteredChanges, nds);
                // Save to history after node changes
                const isDragEnd = filteredChanges.some(change => change.type === 'position' && change.dragging === false);
                const isRemove = filteredChanges.some(change => change.type === 'remove');
                const isAdd = filteredChanges.some(change => change.type === 'add');

                if (isDragEnd) {
                  // Debounce drag end to avoid too many saves
                  saveToHistory(newNodes, edges, true);
                } else if (isRemove || isAdd) {
                  // Save immediately for add/remove operations
                  saveToHistory(newNodes, edges, false);
                }
                return newNodes;
              });
            }}
            onEdgesChange={(changes) => {
              setEdges((eds) => {
                const newEdges = applyEdgeChanges(changes, eds);
                const isRemove = changes.some(change => change.type === 'remove');
                if (isRemove) {
                  setTimeout(() => saveToHistory(nodes, newEdges), 100);
                }
                return newEdges;
              });
            }}
            onConnect={onConnect}
            onInit={setReactFlowInstance}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onNodeContextMenu={onNodeContextMenu}
            onEdgeContextMenu={onEdgeContextMenu}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            isValidConnection={isValidConnection}
            fitView
            fitViewOptions={{ maxZoom: 0.75 }}
            proOptions={{ hideAttribution: true }}
            connectionMode="loose"
            connectionLineComponent={SmoothStepEdge}
            defaultEdgeOptions={{
              type: 'floating',
              markerEnd: { type: 'arrowclosed', color: '#6a1b9a' },
              style: { stroke: '#6a1b9a', strokeWidth: 2 },
            }}
            panOnDrag={activeTool === 'pan'}
            selectionOnDrag={activeTool === 'select'}
          >
            <MiniMap
              style={{
                position: 'absolute',
                bottom: '60px',
                right: '15px',
                boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)'
              }}
              nodeColor="#888"
              maskColor="rgba(255, 255, 255, 0.7)"
            />
          </ReactFlow>

          <button className="add-node-button" onClick={addDefaultNode}>+</button>
          {contextMenu && (
            <ContextMenu
              x={contextMenu.x}
              y={contextMenu.y}
              nodeId={contextMenu.nodeId}
              nodeType={contextMenu.nodeType}
              onDelete={handleDeleteNode}
              onClose={() => setContextMenu(null)}
            />
          )}
          {edgeContextMenu && (
            <div
              className="edge-context-menu"
              style={{
                position: 'fixed',
                top: edgeContextMenu.y,
                left: edgeContextMenu.x,
                background: 'white',
                border: '1px solid #ccc',
                borderRadius: '4px',
                boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
                zIndex: 1000,
                minWidth: '120px',
              }}
            >
              <button
                onClick={() => handleDeleteEdge(edgeContextMenu.edgeId)}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  border: 'none',
                  background: 'transparent',
                  textAlign: 'left',
                  cursor: 'pointer',
                  color: '#e53e3e',
                  fontSize: '14px',
                }}
                onMouseEnter={(e) => e.target.style.background = '#f7fafc'}
                onMouseLeave={(e) => e.target.style.background = 'transparent'}
              >
                🗑️ Delete Edge
              </button>
            </div>
          )}
        </div>
      </div>
      <BottomToolbar
        zoomIn={zoomIn}
        zoomOut={zoomOut}
        fitView={fitView}
        zoomLevel={zoom}
        onUndo={handleUndo}
        onRedo={handleRedo}
        canUndo={canUndo}
        canRedo={canRedo}
      />
    </div>
  );
};

export default EditorPage;