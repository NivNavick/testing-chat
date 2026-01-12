-- Configurable Insights Schema
-- Migration: 002_insight_config
-- Description: Creates tables for the generic AI-driven insight framework
-- 
-- This enables:
-- - Defining insights as data (no code changes to add new ones)
-- - Configurable rules with AI-interpretable descriptions
-- - Severity mappings for flagging results
-- - Execution audit logging

-- ============================================================================
-- 1. Insight Definitions (generic - works for ANY insight)
-- ============================================================================
CREATE TABLE IF NOT EXISTS insight_definitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    display_name VARCHAR(200),
    description TEXT,
    goal TEXT NOT NULL,                  -- Natural language goal for AI to interpret
    required_tables TEXT[] NOT NULL,     -- Prerequisites: ["employee_shifts", "medical_actions"]
    category VARCHAR(100),               -- For grouping: "compliance", "payroll", "operations"
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Index for listing by category
CREATE INDEX IF NOT EXISTS idx_insight_definitions_category ON insight_definitions(category);
CREATE INDEX IF NOT EXISTS idx_insight_definitions_active ON insight_definitions(is_active);

-- ============================================================================
-- 2. Configurable Rules (AI reads descriptions to understand how to apply)
-- ============================================================================
CREATE TABLE IF NOT EXISTS insight_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    insight_id UUID REFERENCES insight_definitions(id) ON DELETE CASCADE,
    rule_name VARCHAR(100) NOT NULL,
    value_type VARCHAR(50) NOT NULL,     -- 'integer', 'float', 'string', 'boolean'
    current_value TEXT NOT NULL,         -- Can be changed via API
    default_value TEXT NOT NULL,         -- For reset functionality
    min_value TEXT,                      -- Optional: for numeric validation
    max_value TEXT,                      -- Optional: for numeric validation
    description TEXT NOT NULL,           -- KEY: AI reads this to understand the rule
    ai_hint TEXT,                        -- Optional: extra guidance for AI
    display_order INTEGER DEFAULT 0,     -- For UI ordering
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(insight_id, rule_name)
);

-- Index for fetching rules by insight
CREATE INDEX IF NOT EXISTS idx_insight_rules_insight_id ON insight_rules(insight_id);

-- ============================================================================
-- 3. Severity Conditions (AI applies these to flag results)
-- ============================================================================
CREATE TABLE IF NOT EXISTS insight_severities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    insight_id UUID REFERENCES insight_definitions(id) ON DELETE CASCADE,
    condition_name VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('INFO', 'WARNING', 'CRITICAL')),
    condition_description TEXT NOT NULL, -- AI reads this to know when to apply
    display_order INTEGER DEFAULT 0,     -- For UI ordering
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(insight_id, condition_name)
);

-- Index for fetching severities by insight
CREATE INDEX IF NOT EXISTS idx_insight_severities_insight_id ON insight_severities(insight_id);

-- ============================================================================
-- 4. Execution Audit Log
-- ============================================================================
CREATE TABLE IF NOT EXISTS insight_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    insight_id UUID REFERENCES insight_definitions(id) ON DELETE SET NULL,
    session_id UUID,                     -- References the analysis session
    executed_at TIMESTAMP DEFAULT NOW(),
    
    -- Snapshot of configuration at execution time
    rules_snapshot JSONB,                -- Rules used at execution time
    
    -- AI-generated artifacts
    correlations_found JSONB,            -- How AI correlated the tables
    generated_sql TEXT,                  -- The AI-generated query
    ai_explanation TEXT,                 -- AI's explanation of the query
    
    -- Results summary
    total_records INTEGER,
    flagged_records INTEGER,
    flags_by_severity JSONB,             -- {"CRITICAL": 2, "WARNING": 5, "INFO": 1}
    
    -- Performance
    execution_time_ms FLOAT,
    
    -- Error handling
    success BOOLEAN DEFAULT true,
    error_message TEXT
);

-- Indexes for querying execution history
CREATE INDEX IF NOT EXISTS idx_insight_executions_insight_id ON insight_executions(insight_id);
CREATE INDEX IF NOT EXISTS idx_insight_executions_session_id ON insight_executions(session_id);
CREATE INDEX IF NOT EXISTS idx_insight_executions_executed_at ON insight_executions(executed_at DESC);

-- ============================================================================
-- 5. Trigger to update updated_at timestamp
-- ============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to insight_definitions
DROP TRIGGER IF EXISTS update_insight_definitions_updated_at ON insight_definitions;
CREATE TRIGGER update_insight_definitions_updated_at
    BEFORE UPDATE ON insight_definitions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Apply trigger to insight_rules
DROP TRIGGER IF EXISTS update_insight_rules_updated_at ON insight_rules;
CREATE TRIGGER update_insight_rules_updated_at
    BEFORE UPDATE ON insight_rules
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- 6. Helper view for getting insights with rule/severity counts
-- ============================================================================
CREATE OR REPLACE VIEW insight_summary AS
SELECT 
    d.id,
    d.name,
    d.display_name,
    d.category,
    d.is_active,
    d.required_tables,
    COUNT(DISTINCT r.id) AS rule_count,
    COUNT(DISTINCT s.id) AS severity_count,
    d.created_at,
    d.updated_at
FROM insight_definitions d
LEFT JOIN insight_rules r ON d.id = r.insight_id
LEFT JOIN insight_severities s ON d.id = s.insight_id
GROUP BY d.id, d.name, d.display_name, d.category, d.is_active, 
         d.required_tables, d.created_at, d.updated_at;

