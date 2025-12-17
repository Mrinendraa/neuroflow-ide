import React from 'react';
import './BottomToolbar.css';
import { MdAdd, MdRemove, MdZoomOutMap, MdUndo, MdRedo } from 'react-icons/md';

// The component now accepts props for zoom functionality and undo/redo
const BottomToolbar = ({ zoomIn, zoomOut, fitView, zoomLevel, onUndo, onRedo, canUndo, canRedo }) => {
  // Format the zoom level to a percentage
  const zoomPercentage = `${Math.round(zoomLevel * 100)}%`;

  return (
    <div className="bottom-toolbar">
      <div className="toolbar-left-bottom">
        {/* Attach the zoomOut function to the minus button's onClick handler */}
        <button className="icon-button" onClick={zoomOut} title="Zoom Out">
          <MdRemove />
        </button>

        <span className="zoom-level">{zoomPercentage}</span>

        {/* Attach the zoomIn function to the plus button's onClick handler */}
        <button className="icon-button" onClick={zoomIn} title="Zoom In">
          <MdAdd />
        </button>
        
        {/* Attach the fitView function to a "fit view" button */}
        <button className="icon-button" onClick={fitView} title="Fit View">
          <MdZoomOutMap />
        </button>
      </div>
      <div className="toolbar-right-bottom">
        <button 
          className={`icon-button ${!canUndo ? 'disabled' : ''}`} 
          onClick={onUndo}
          disabled={!canUndo}
          title="Undo (Ctrl+Z)"
        >
          <MdUndo />
        </button>
        <button 
          className={`icon-button ${!canRedo ? 'disabled' : ''}`} 
          onClick={onRedo}
          disabled={!canRedo}
          title="Redo (Ctrl+Y)"
        >
          <MdRedo />
        </button>
      </div>
    </div>
  );
};

export default BottomToolbar;