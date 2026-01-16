-- API Sessions and Workflow Executions Schema
-- Migration for FastAPI async wrapper
-- Run this after the initial schema migrations

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. API Sessions Table
-- Tracks user sessions with metadata stored in PostgreSQL
-- Actual data files stored in S3
CREATE TABLE IF NOT EXISTS api_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    vertical VARCHAR(100) NOT NULL DEFAULT 'medical',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    s3_bucket VARCHAR(255),
    s3_prefix VARCHAR(500)  -- S3 path prefix for session files
);

-- Index for listing sessions
CREATE INDEX IF NOT EXISTS idx_api_sessions_vertical ON api_sessions(vertical);
CREATE INDEX IF NOT EXISTS idx_api_sessions_created_at ON api_sessions(created_at DESC);

-- 2. Session Documents Table
-- Tracks documents uploaded to a session
CREATE TABLE IF NOT EXISTS session_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES api_sessions(id) ON DELETE CASCADE,
    document_id VARCHAR(100) NOT NULL,
    document_type VARCHAR(100),
    filename VARCHAR(500) NOT NULL,
    s3_uri VARCHAR(1000),  -- S3 URI for the uploaded file
    row_count INTEGER DEFAULT 0,
    column_count INTEGER DEFAULT 0,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    
    -- Classification results (if classified)
    classification_result JSONB,
    classification_confidence FLOAT,
    
    UNIQUE(session_id, document_id)
);

-- Indexes for querying documents
CREATE INDEX IF NOT EXISTS idx_session_docs_session_id ON session_documents(session_id);
CREATE INDEX IF NOT EXISTS idx_session_docs_document_type ON session_documents(document_type);
CREATE INDEX IF NOT EXISTS idx_session_docs_uploaded_at ON session_documents(uploaded_at DESC);

-- 3. Workflow Executions Table
-- Tracks workflow runs and their results
CREATE TABLE IF NOT EXISTS workflow_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_name VARCHAR(255) NOT NULL,
    session_id UUID REFERENCES api_sessions(id) ON DELETE SET NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed, cancelled
    parameters JSONB DEFAULT '{}',
    results JSONB DEFAULT '{}',  -- block_id -> output_name -> s3_uri
    error TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    execution_time_ms FLOAT
);

-- Indexes for workflow executions
CREATE INDEX IF NOT EXISTS idx_workflow_exec_session ON workflow_executions(session_id);
CREATE INDEX IF NOT EXISTS idx_workflow_exec_status ON workflow_executions(status);
CREATE INDEX IF NOT EXISTS idx_workflow_exec_workflow_name ON workflow_executions(workflow_name);
CREATE INDEX IF NOT EXISTS idx_workflow_exec_started_at ON workflow_executions(started_at DESC);

-- 4. Block Executions Table (optional - for detailed block tracking)
CREATE TABLE IF NOT EXISTS block_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_execution_id UUID NOT NULL REFERENCES workflow_executions(id) ON DELETE CASCADE,
    block_id VARCHAR(100) NOT NULL,
    handler VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed, skipped
    inputs JSONB DEFAULT '{}',
    outputs JSONB DEFAULT '{}',
    error TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    execution_time_ms FLOAT,
    
    UNIQUE(workflow_execution_id, block_id)
);

CREATE INDEX IF NOT EXISTS idx_block_exec_workflow ON block_executions(workflow_execution_id);
CREATE INDEX IF NOT EXISTS idx_block_exec_status ON block_executions(status);

-- 5. Update trigger for api_sessions
CREATE OR REPLACE FUNCTION update_api_session_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_api_session_timestamp ON api_sessions;
CREATE TRIGGER trigger_update_api_session_timestamp
    BEFORE UPDATE ON api_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_api_session_timestamp();

-- 6. Classification History Table (optional - for tracking classification attempts)
CREATE TABLE IF NOT EXISTS classification_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES api_sessions(id) ON DELETE SET NULL,
    document_id VARCHAR(100),
    filename VARCHAR(500),
    vertical VARCHAR(100),
    detected_document_type VARCHAR(100),
    confidence FLOAT,
    final_score FLOAT,
    document_score FLOAT,
    column_score FLOAT,
    coverage_score FLOAT,
    suggested_mappings JSONB,
    similar_examples JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_classification_history_session ON classification_history(session_id);
CREATE INDEX IF NOT EXISTS idx_classification_history_doc_type ON classification_history(detected_document_type);
CREATE INDEX IF NOT EXISTS idx_classification_history_created_at ON classification_history(created_at DESC);

