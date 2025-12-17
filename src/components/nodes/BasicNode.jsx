import React from 'react';
import { Handle, Position } from 'reactflow';
import './BasicNode.css';

const BasicNode = ({ data, isConnectable }) => {
  const label = data?.label || 'Node';

  return (
    <div className="basic-node">
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} />
      <div className="basic-node__body">
        <span className="basic-node__label">{label}</span>
      </div>
      <Handle type="source" position={Position.Bottom} isConnectable={isConnectable} />
    </div>
  );
};

export default BasicNode;
