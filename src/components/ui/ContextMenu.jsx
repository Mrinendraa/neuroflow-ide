import React from 'react';
import './ContextMenu.css';
import { MdDelete } from 'react-icons/md';

const ContextMenu = ({ x, y, nodeId, nodeType, onDelete, onClose }) => {
  if (!nodeId) return null;

  // Don't show delete option for start node
  const showDelete = nodeType !== 'start';

  const handleDelete = () => {
    if (showDelete && onDelete) {
      onDelete(nodeId);
    }
    onClose();
  };

  if (!showDelete) return null;

  return (
    <div 
      className="context-menu" 
      style={{ left: `${x}px`, top: `${y}px` }}
      onClick={(e) => e.stopPropagation()}
    >
      <button className="context-menu-item delete" onClick={handleDelete}>
        <MdDelete />
        <span>Delete</span>
      </button>
    </div>
  );
};

export default ContextMenu;

