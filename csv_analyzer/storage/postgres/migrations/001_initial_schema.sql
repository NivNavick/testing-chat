-- Analytics Platform Initial Schema
-- Migration 001: Core tables for sessions, files, tables, classifications, and insights

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vertical VARCHAR(50) NOT NULL,
    mode VARCHAR(20) DEFAULT 'AUTO',
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    closed_at TIMESTAMP
);

-- Uploaded files (metadata, actual file in S3)
CREATE TABLE IF NOT EXISTS uploaded_files (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(10) NOT NULL,
    file_size_bytes BIGINT,
    s3_key VARCHAR(500) NOT NULL,
    original_structure VARCHAR(20),
    processing_status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT,
    uploaded_at TIMESTAMP DEFAULT NOW()
);

-- Extracted tables (metadata, data loaded to DuckDB on demand)
CREATE TABLE IF NOT EXISTS extracted_tables (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_id UUID REFERENCES uploaded_files(id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    table_name VARCHAR(100),
    location VARCHAR(100),
    row_count INT,
    column_count INT,
    columns JSONB,
    structure_type VARCHAR(20),
    transforms_applied JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Classifications
CREATE TABLE IF NOT EXISTS classifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_id UUID REFERENCES extracted_tables(id) ON DELETE CASCADE,
    document_type VARCHAR(100),
    confidence FLOAT,
    method VARCHAR(20),
    column_mappings JSONB,
    unmapped_columns JSONB,
    llm_reasoning TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Relationships between tables
CREATE TABLE IF NOT EXISTS table_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    from_table_id UUID REFERENCES extracted_tables(id),
    from_column VARCHAR(100),
    to_table_id UUID REFERENCES extracted_tables(id),
    to_column VARCHAR(100),
    relationship_type VARCHAR(20),
    confidence FLOAT,
    transform_sql TEXT,
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Business rules
CREATE TABLE IF NOT EXISTS business_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    rule_type VARCHAR(20),
    description TEXT,
    applies_to_tables UUID[],
    condition_sql TEXT,
    formula_sql TEXT,
    confidence FLOAT,
    source VARCHAR(20),
    evidence TEXT,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Insight results (metadata)
CREATE TABLE IF NOT EXISTS insight_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    insight_name VARCHAR(100),
    insight_type VARCHAR(20),
    executed_sql TEXT,
    parameters JSONB,
    row_count INT,
    column_names JSONB,
    summary_stats JSONB,
    s3_result_key VARCHAR(500),
    execution_time_ms FLOAT,
    success BOOLEAN,
    error_message TEXT,
    executed_at TIMESTAMP DEFAULT NOW()
);

-- Insight result data (actual rows stored in PostgreSQL)
CREATE TABLE IF NOT EXISTS insight_result_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    result_id UUID REFERENCES insight_results(id) ON DELETE CASCADE,
    row_number INT,
    row_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Custom insights
CREATE TABLE IF NOT EXISTS custom_insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id),
    name VARCHAR(100),
    description TEXT,
    requires_tables JSONB,
    sql_template TEXT,
    parameters JSONB,
    generated_by VARCHAR(20),
    validated BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_files_session ON uploaded_files(session_id);
CREATE INDEX IF NOT EXISTS idx_tables_session ON extracted_tables(session_id);
CREATE INDEX IF NOT EXISTS idx_tables_file ON extracted_tables(file_id);
CREATE INDEX IF NOT EXISTS idx_classifications_table ON classifications(table_id);
CREATE INDEX IF NOT EXISTS idx_relationships_session ON table_relationships(session_id);
CREATE INDEX IF NOT EXISTS idx_rules_session ON business_rules(session_id);
CREATE INDEX IF NOT EXISTS idx_results_session ON insight_results(session_id);
CREATE INDEX IF NOT EXISTS idx_result_data_result_id ON insight_result_data(result_id);
CREATE INDEX IF NOT EXISTS idx_result_data_row_number ON insight_result_data(result_id, row_number);
CREATE INDEX IF NOT EXISTS idx_custom_insights_session ON custom_insights(session_id);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for sessions updated_at
DROP TRIGGER IF EXISTS update_sessions_updated_at ON sessions;
CREATE TRIGGER update_sessions_updated_at
    BEFORE UPDATE ON sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

