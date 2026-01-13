import yaml from 'js-yaml';

/**
 * Parses a workflow YAML and converts it to React Flow nodes and edges
 * @param {string} yamlContent - The YAML content as a string
 * @returns {Object} - { nodes, edges, workflow }
 */
export function parseWorkflowYAML(yamlContent) {
  try {
    const workflow = yaml.load(yamlContent);
    
    if (!workflow || !workflow.blocks) {
      throw new Error('Invalid workflow structure: missing blocks');
    }

    const nodes = [];
    const edges = [];
    const blockPositions = calculateBlockPositions(workflow.blocks);

    // Create nodes for each block
    workflow.blocks.forEach((block, index) => {
      const position = blockPositions[block.id] || { x: 250, y: index * 150 };
      
      nodes.push({
        id: block.id,
        type: 'workflowBlock',
        position,
        data: {
          label: block.id,
          handler: block.handler,
          parameters: block.parameters || {},
          inputs: block.inputs || [],
          description: getBlockDescription(block)
        }
      });

      // Create edges based on inputs
      if (block.inputs && Array.isArray(block.inputs)) {
        block.inputs.forEach((input) => {
          const sourceInfo = parseSourceReference(input.source || input.name);
          if (sourceInfo) {
            edges.push({
              id: `${sourceInfo.blockId}-${block.id}-${input.name || input.field || 'data'}`,
              source: sourceInfo.blockId,
              target: block.id,
              label: input.field || input.name || '',
              type: 'smoothstep',
              animated: false,
              data: {
                sourceField: sourceInfo.field,
                targetField: input.field || input.name
              }
            });
          }
        });
      }
    });

    return {
      nodes,
      edges,
      workflow: {
        name: workflow.name,
        description: workflow.description,
        version: workflow.version,
        parameters: workflow.parameters || {}
      }
    };
  } catch (error) {
    console.error('Error parsing workflow YAML:', error);
    throw error;
  }
}

/**
 * Parse a source reference like "upload.uploaded_files" or "classify.classified_data.employee_name"
 * @param {string} source - The source reference
 * @returns {Object|null} - { blockId, field }
 */
function parseSourceReference(source) {
  if (!source || typeof source !== 'string') {
    return null;
  }

  const parts = source.split('.');
  if (parts.length < 2) {
    return null;
  }

  return {
    blockId: parts[0],
    field: parts.slice(1).join('.')
  };
}

/**
 * Calculate positions for blocks in a hierarchical layout
 * @param {Array} blocks - Array of workflow blocks
 * @returns {Object} - Map of block ID to position { x, y }
 */
function calculateBlockPositions(blocks) {
  const positions = {};
  const dependencies = new Map();
  const levels = new Map();

  // Build dependency graph
  blocks.forEach(block => {
    dependencies.set(block.id, new Set());
    
    if (block.inputs && Array.isArray(block.inputs)) {
      block.inputs.forEach(input => {
        const sourceInfo = parseSourceReference(input.source || input.name);
        if (sourceInfo) {
          dependencies.get(block.id).add(sourceInfo.blockId);
        }
      });
    }
  });

  // Calculate levels using topological sort
  const calculateLevel = (blockId, visited = new Set()) => {
    if (levels.has(blockId)) {
      return levels.get(blockId);
    }
    
    if (visited.has(blockId)) {
      return 0; // Circular dependency, assign level 0
    }
    
    visited.add(blockId);
    const deps = dependencies.get(blockId);
    
    if (!deps || deps.size === 0) {
      levels.set(blockId, 0);
      return 0;
    }
    
    let maxLevel = 0;
    deps.forEach(depId => {
      const depLevel = calculateLevel(depId, visited);
      maxLevel = Math.max(maxLevel, depLevel + 1);
    });
    
    levels.set(blockId, maxLevel);
    return maxLevel;
  };

  // Calculate levels for all blocks
  blocks.forEach(block => {
    calculateLevel(block.id);
  });

  // Group blocks by level
  const levelGroups = new Map();
  levels.forEach((level, blockId) => {
    if (!levelGroups.has(level)) {
      levelGroups.set(level, []);
    }
    levelGroups.get(level).push(blockId);
  });

  // Position blocks
  const horizontalSpacing = 420;
  const verticalSpacing = 180;
  const startX = 100;
  const startY = 100;

  levelGroups.forEach((blockIds, level) => {
    blockIds.forEach((blockId, indexInLevel) => {
      positions[blockId] = {
        x: startX + (level * horizontalSpacing),
        y: startY + (indexInLevel * verticalSpacing)
      };
    });
  });

  return positions;
}

/**
 * Generate a description for a block
 * @param {Object} block - The workflow block
 * @returns {string} - Description text
 */
function getBlockDescription(block) {
  const parts = [];
  
  if (block.handler) {
    parts.push(`Handler: ${block.handler}`);
  }
  
  if (block.inputs && block.inputs.length > 0) {
    const inputCount = block.inputs.length;
    parts.push(`${inputCount} input${inputCount > 1 ? 's' : ''}`);
  }
  
  if (block.parameters && Object.keys(block.parameters).length > 0) {
    const paramCount = Object.keys(block.parameters).length;
    parts.push(`${paramCount} parameter${paramCount > 1 ? 's' : ''}`);
  }
  
  return parts.join(' • ');
}

/**
 * Load workflow YAML files from the project
 * @returns {Array} - List of available workflow files
 */
export function getAvailableWorkflows() {
  // This would be populated from the backend or file system
  return [
    'medical_insights.yaml',
    'medical_early_arrival.yaml',
    'salary_analysis.yaml',
    'early_arrival_simple.yaml',
    'medical_insights_dual_entry.yaml'
  ];
}

