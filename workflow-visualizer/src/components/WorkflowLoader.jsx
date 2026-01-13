import { useState } from 'react';
import { parseWorkflowYAML } from '../utils/workflowParser';
import './WorkflowLoader.css';

export default function WorkflowLoader({ onWorkflowLoad }) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState(null);
  const [yamlText, setYamlText] = useState('');
  const [showTextInput, setShowTextInput] = useState(false);

  const handleFileUpload = async (file) => {
    setError(null);
    
    if (!file.name.endsWith('.yaml') && !file.name.endsWith('.yml')) {
      setError('Please upload a YAML file (.yaml or .yml)');
      return;
    }

    try {
      const text = await file.text();
      const result = parseWorkflowYAML(text);
      onWorkflowLoad(result);
    } catch (err) {
      setError(`Error parsing workflow: ${err.message}`);
      console.error(err);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleFileInput = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleTextSubmit = () => {
    setError(null);
    
    if (!yamlText.trim()) {
      setError('Please paste YAML content');
      return;
    }

    try {
      const result = parseWorkflowYAML(yamlText);
      onWorkflowLoad(result);
      setYamlText('');
      setShowTextInput(false);
    } catch (err) {
      setError(`Error parsing workflow: ${err.message}`);
      console.error(err);
    }
  };

  const loadSampleWorkflow = async (filename) => {
    setError(null);
    
    try {
      const response = await fetch(`/workflows/${filename}`);
      if (!response.ok) {
        throw new Error(`Failed to load workflow: ${response.statusText}`);
      }
      const text = await response.text();
      const result = parseWorkflowYAML(text);
      onWorkflowLoad(result);
    } catch (err) {
      setError(`Error loading sample: ${err.message}`);
      console.error(err);
    }
  };

  const sampleWorkflows = [
    'example-workflow.yaml',
    'medical_insights.yaml',
    'medical_early_arrival.yaml',
    'salary_analysis.yaml',
    'early_arrival_simple.yaml',
    'medical_insights_dual_entry.yaml',
  ];

  return (
    <div className="workflow-loader">
      <div className="loader-header">
        <h1>Workflow Visualizer</h1>
        <p>Upload or paste a workflow YAML file to visualize its structure</p>
      </div>

      {error && (
        <div className="error-message">
          <span className="error-icon">⚠️</span>
          {error}
        </div>
      )}

      <div
        className={`drop-zone ${isDragging ? 'dragging' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <div className="drop-zone-content">
          <div className="upload-icon">📁</div>
          <p className="drop-zone-text">
            Drag and drop a YAML file here
          </p>
          <p className="drop-zone-subtext">or</p>
          <label className="file-input-label">
            <input
              type="file"
              accept=".yaml,.yml"
              onChange={handleFileInput}
              className="file-input"
            />
            Choose File
          </label>
        </div>
      </div>

      <div className="divider">
        <span>OR</span>
      </div>

      <button
        className="text-input-toggle"
        onClick={() => setShowTextInput(!showTextInput)}
      >
        {showTextInput ? '📁 Upload File Instead' : '📝 Paste YAML Text'}
      </button>

      {showTextInput && (
        <div className="text-input-section">
          <textarea
            className="yaml-textarea"
            placeholder="Paste your workflow YAML here..."
            value={yamlText}
            onChange={(e) => setYamlText(e.target.value)}
            rows={15}
          />
          <button
            className="submit-button"
            onClick={handleTextSubmit}
          >
            Visualize Workflow
          </button>
        </div>
      )}

      <div className="sample-workflows">
        <h3>Available Workflows</h3>
        <p className="sample-note">
          Load these files from: <code>csv_analyzer/workflows/definitions/</code>
        </p>
        <div className="sample-list">
          {sampleWorkflows.map((workflow) => (
            <button
              key={workflow}
              className="sample-button"
              onClick={() => loadSampleWorkflow(workflow)}
            >
              {workflow}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

