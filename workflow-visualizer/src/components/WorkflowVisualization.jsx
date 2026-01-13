import { useCallback, useState } from 'react';
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Panel,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import WorkflowBlockNode from './WorkflowBlockNode';
import './WorkflowVisualization.css';

const nodeTypes = {
  workflowBlock: WorkflowBlockNode,
};

export default function WorkflowVisualization({ nodes: initialNodes, edges: initialEdges, workflow }) {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = useState(null);

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const onNodeClick = useCallback((event, node) => {
    setSelectedNode(node);
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  return (
    <div className="workflow-visualization">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-left"
        elevateEdgesOnSelect={true}
        elevateNodesOnSelect={false}
      >
        <Background variant="dots" gap={16} size={1} />
        <Controls />
        <MiniMap 
          nodeColor={(node) => {
            return '#d0d9f7';
          }}
          maskColor="rgba(0, 0, 0, 0.1)"
        />
        
        <Panel position="top-left" className="workflow-info-panel">
          <div className="workflow-header">
            <h2>{workflow?.name || 'Workflow'}</h2>
            {workflow?.version && (
              <span className="version-badge">v{workflow.version}</span>
            )}
          </div>
          {workflow?.description && (
            <p className="workflow-description">{workflow.description}</p>
          )}
          <div className="workflow-stats">
            <div className="stat-item">
              <span className="stat-label">Blocks:</span>
              <span className="stat-value">{nodes.length}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Connections:</span>
              <span className="stat-value">{edges.length}</span>
            </div>
          </div>
        </Panel>

        {selectedNode && (
          <Panel position="top-right" className="node-details-panel">
            <div className="panel-header">
              <h3>Block Details</h3>
              <button 
                className="close-button"
                onClick={() => setSelectedNode(null)}
              >
                ×
              </button>
            </div>
            <div className="detail-section">
              <strong>ID:</strong> {selectedNode.data.label}
            </div>
            <div className="detail-section">
              <strong>Handler:</strong> {selectedNode.data.handler}
            </div>
            
            {selectedNode.data.inputs && selectedNode.data.inputs.length > 0 && (
              <div className="detail-section">
                <strong>Inputs:</strong>
                <ul className="detail-list">
                  {selectedNode.data.inputs.map((input, idx) => (
                    <li key={idx}>
                      <code>{input.field || input.name}</code>
                      {input.source && (
                        <span className="input-source"> ← {input.source}</span>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            {selectedNode.data.parameters && Object.keys(selectedNode.data.parameters).length > 0 && (
              <div className="detail-section">
                <strong>Parameters:</strong>
                <ul className="detail-list">
                  {Object.entries(selectedNode.data.parameters).map(([key, value]) => (
                    <li key={key}>
                      <code>{key}</code>: {String(value)}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </Panel>
        )}
      </ReactFlow>
    </div>
  );
}

