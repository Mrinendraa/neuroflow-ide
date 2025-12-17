import React from 'react';
import { Handle, Position } from 'reactflow';
import './StartNode.css';

function StartNode({ data }) {
  const label = data?.label || 'Start';
  return (
    <div className="start-node">
      <div className="start-node__label">{label}</div>
      <Handle type="source" position={Position.Bottom} className="custom-handle" id="out" />
      <Handle type="source" position={Position.Right} className="custom-handle" id="out_r" />
    </div>
  );
}

export default StartNode;



