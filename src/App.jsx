import { ReactFlowProvider } from 'reactflow';
import EditorPage from './pages/EditorPage';

import { ThemeProvider } from './context/ThemeContext';



function App() {
  return (
    <ThemeProvider>

      <ReactFlowProvider>
        <EditorPage />
      </ReactFlowProvider>
    </ThemeProvider>
  );
}

export default App;