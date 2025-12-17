import React from 'react';
import './TopToolbar.css';
import {
  MdMenu, MdOutlineMouse, MdOutlinePanTool, MdOutlineShare, MdOutlineLibraryBooks,
  MdLightMode, MdDarkMode
} from 'react-icons/md';
import { useTheme } from '../../context/ThemeContext';

// Component now accepts props to manage tool state
const TopToolbar = ({ activeTool, setActiveTool, onMenuClick }) => {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="top-toolbar">
      {/* Hamburger menu in separate circle */}
      <div className="hamburger-container">
        <button className="hamburger-button" onClick={onMenuClick}>
          <MdMenu />
        </button>
      </div>

      {/* Main floating oval toolbar */}
      <div className="main-toolbar">
        <div className="toolbar-center">
          {/* Select Tool Button */}
          <button
            className={`icon-button ${activeTool === 'select' ? 'selected' : ''}`}
            onClick={() => setActiveTool('select')}
            title="Select Tool (V)"
          >
            <MdOutlineMouse />
          </button>

          {/* Hand/Pan Tool Button */}
          <button
            className={`icon-button ${activeTool === 'pan' ? 'selected' : ''}`}
            onClick={() => setActiveTool('pan')}
            title="Pan Tool (H)"
          >
            <MdOutlinePanTool />
          </button>

          {/* Theme Toggle Button */}
          <button
            className="icon-button"
            onClick={toggleTheme}
            title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
          >
            {theme === 'dark' ? <MdLightMode /> : <MdDarkMode />}
          </button>
        </div>
      </div>

      {/* Share and Library in separate oval */}
      <div className="actions-toolbar">
        <button className="action-button">
          <MdOutlineShare />
          <span>Share</span>
        </button>
        <button className="action-button">
          <MdOutlineLibraryBooks />
          <span>Library</span>
        </button>
      </div>
    </div>
  );
};

export default TopToolbar;