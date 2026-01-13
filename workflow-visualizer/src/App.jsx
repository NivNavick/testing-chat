import { useState } from 'react';
import WorkflowLoader from './components/WorkflowLoader';
import WorkflowVisualization from './components/WorkflowVisualization';
import './App.css';

function App() {
  const [workflowData, setWorkflowData] = useState(null);

  const handleWorkflowLoad = (data) => {
    setWorkflowData(data);
  };

  const handleReset = () => {
    setWorkflowData(null);
  };

  return (
    <div className="app">
      {!workflowData ? (
        <WorkflowLoader onWorkflowLoad={handleWorkflowLoad} />
      ) : (
        <div className="visualization-container">
          <button className="reset-button" onClick={handleReset}>
            ← Load Different Workflow
          </button>
          <WorkflowVisualization
            nodes={workflowData.nodes}
            edges={workflowData.edges}
            workflow={workflowData.workflow}
          />
        </div>
      )}
    </div>
  );
}

export default App;
