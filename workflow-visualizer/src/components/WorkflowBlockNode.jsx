import { Handle, Position } from '@xyflow/react';
import './WorkflowBlockNode.css';

/**
 * Custom node component for workflow blocks
 */
export default function WorkflowBlockNode({ data }) {
  const { label, handler, parameters, inputs, description } = data;
  
  const paramCount = parameters ? Object.keys(parameters).length : 0;
  const inputCount = inputs ? inputs.length : 0;

  return (
    <div className="workflow-block-node">
      <Handle 
        type="target" 
        position={Position.Left} 
        className="workflow-handle"
      />
      
      <div className="node-header">
        <div className="node-label">{label}</div>
        <div className="node-handler">{handler}</div>
      </div>
      
      {description && (
        <div className="node-description">{description}</div>
      )}
      
      <div className="node-stats">
        {inputCount > 0 && (
          <span className="stat-badge inputs">
            📥 {inputCount}
          </span>
        )}
        {paramCount > 0 && (
          <span className="stat-badge params">
            ⚙️ {paramCount}
          </span>
        )}
      </div>
      
      <Handle 
        type="source" 
        position={Position.Right} 
        className="workflow-handle"
      />
    </div>
  );
}

