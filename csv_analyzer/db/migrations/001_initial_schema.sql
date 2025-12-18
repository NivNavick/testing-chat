-- CSV Analyzer Schema
-- Database: csv_mapping
-- Run this migration to set up the tables

-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Verticals (medical, banking, etc.)
CREATE TABLE IF NOT EXISTS verticals (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 2. Document Types (employee_shifts, medical_actions, etc.)
CREATE TABLE IF NOT EXISTS document_types (
    id SERIAL PRIMARY KEY,
    vertical_id INTEGER REFERENCES verticals(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(vertical_id, name)
);

-- 3. Target Schemas (what we transform TO)
CREATE TABLE IF NOT EXISTS target_schemas (
    id SERIAL PRIMARY KEY,
    document_type_id INTEGER REFERENCES document_types(id) ON DELETE CASCADE,
    version VARCHAR(20) DEFAULT '1.0',
    schema_definition JSONB NOT NULL,  -- the actual schema fields
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(document_type_id, version)
);

-- 4. Ground Truth Records (labeled CSV examples)
-- This is the main table for storing ground truth embeddings
CREATE TABLE IF NOT EXISTS ground_truth (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(100) UNIQUE,  -- e.g., "gt_medical_shifts_001"
    document_type_id INTEGER REFERENCES document_types(id) ON DELETE CASCADE,
    
    -- CSV metadata (NOT actual data)
    source_description TEXT,           -- "Hospital HR export", etc.
    row_count INTEGER,
    column_count INTEGER,
    
    -- Column profiles as JSON (structure only, no actual data)
    column_profiles JSONB NOT NULL,
    
    -- Text representation (what we embed)
    text_representation TEXT NOT NULL,
    
    -- THE EMBEDDING (e5-large = 1024 dimensions)
    embedding vector(1024) NOT NULL,
    
    -- Column mappings (source → target)
    column_mappings JSONB NOT NULL,
    
    -- Metadata
    labeler VARCHAR(255),
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 5. Column Mapping Knowledge Base (individual column mappings)
-- Stores learned column name → target field associations
CREATE TABLE IF NOT EXISTS column_mappings_kb (
    id SERIAL PRIMARY KEY,
    vertical_id INTEGER REFERENCES verticals(id) ON DELETE CASCADE,
    document_type_id INTEGER REFERENCES document_types(id) ON DELETE CASCADE,
    
    -- Source column info
    source_column_name VARCHAR(255) NOT NULL,
    source_column_type VARCHAR(50),
    sample_values JSONB,
    
    -- What it maps to
    target_field VARCHAR(255) NOT NULL,
    
    -- Text representation for this column
    column_text_representation TEXT NOT NULL,
    
    -- Embedding for this column
    embedding vector(1024) NOT NULL,
    
    -- How many times we've seen this mapping
    occurrence_count INTEGER DEFAULT 1,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(vertical_id, document_type_id, source_column_name, target_field)
);

-- Vector similarity indexes
-- IMPORTANT: IVFFlat indexes require sufficient data (100+ rows) to work correctly.
-- For small datasets (<100 rows), exact search is faster and more accurate.
-- Uncomment these when you have enough ground truth records:
--
-- CREATE INDEX IF NOT EXISTS idx_ground_truth_embedding 
--     ON ground_truth USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
--
-- CREATE INDEX IF NOT EXISTS idx_column_mappings_embedding 
--     ON column_mappings_kb USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
--
-- Alternative: HNSW indexes work better for small datasets but use more memory:
-- CREATE INDEX IF NOT EXISTS idx_ground_truth_embedding 
--     ON ground_truth USING hnsw (embedding vector_cosine_ops);

-- Standard indexes for filtering
CREATE INDEX IF NOT EXISTS idx_ground_truth_doc_type ON ground_truth(document_type_id);
CREATE INDEX IF NOT EXISTS idx_column_mappings_vertical ON column_mappings_kb(vertical_id);
CREATE INDEX IF NOT EXISTS idx_column_mappings_doc_type ON column_mappings_kb(document_type_id);

-- Insert default verticals
INSERT INTO verticals (name, description) VALUES 
    ('medical', 'Healthcare and medical industry')
ON CONFLICT (name) DO NOTHING;

-- Insert default document types for medical vertical
INSERT INTO document_types (vertical_id, name, description)
SELECT v.id, 'employee_shifts', 'Employee work schedules and shift assignments'
FROM verticals v WHERE v.name = 'medical'
ON CONFLICT (vertical_id, name) DO NOTHING;

INSERT INTO document_types (vertical_id, name, description)
SELECT v.id, 'medical_actions', 'Medical procedures and treatments performed'
FROM verticals v WHERE v.name = 'medical'
ON CONFLICT (vertical_id, name) DO NOTHING;

INSERT INTO document_types (vertical_id, name, description)
SELECT v.id, 'patient_records', 'Patient demographic and medical history'
FROM verticals v WHERE v.name = 'medical'
ON CONFLICT (vertical_id, name) DO NOTHING;

INSERT INTO document_types (vertical_id, name, description)
SELECT v.id, 'lab_results', 'Laboratory test results'
FROM verticals v WHERE v.name = 'medical'
ON CONFLICT (vertical_id, name) DO NOTHING;

-- Insert target schema for employee_shifts
INSERT INTO target_schemas (document_type_id, version, schema_definition)
SELECT dt.id, '1.0', '{
    "name": "employee_shifts",
    "description": "Standardized employee shift schedule",
    "fields": [
        {"name": "employee_id", "type": "string", "required": true, "description": "Unique employee identifier"},
        {"name": "shift_date", "type": "date", "required": true, "description": "Date of the shift"},
        {"name": "shift_start", "type": "datetime", "required": true, "description": "Shift start time"},
        {"name": "shift_end", "type": "datetime", "required": true, "description": "Shift end time"},
        {"name": "duration_minutes", "type": "integer", "required": false, "description": "Shift duration in minutes"},
        {"name": "department_code", "type": "string", "required": false, "description": "Department identifier"},
        {"name": "shift_type", "type": "string", "required": false, "description": "Type of shift (morning/evening/night)"}
    ]
}'::jsonb
FROM document_types dt
JOIN verticals v ON dt.vertical_id = v.id
WHERE v.name = 'medical' AND dt.name = 'employee_shifts'
ON CONFLICT (document_type_id, version) DO NOTHING;

-- Insert target schema for medical_actions
INSERT INTO target_schemas (document_type_id, version, schema_definition)
SELECT dt.id, '1.0', '{
    "name": "medical_actions",
    "description": "Standardized medical procedures and treatments",
    "fields": [
        {"name": "action_id", "type": "string", "required": true, "description": "Unique action identifier"},
        {"name": "patient_id", "type": "string", "required": true, "description": "Patient identifier"},
        {"name": "performer_id", "type": "string", "required": true, "description": "Healthcare worker who performed"},
        {"name": "action_code", "type": "string", "required": true, "description": "Procedure/action code"},
        {"name": "action_name", "type": "string", "required": false, "description": "Readable action name"},
        {"name": "performed_at", "type": "datetime", "required": true, "description": "When action was performed"},
        {"name": "department_code", "type": "string", "required": false, "description": "Department identifier"},
        {"name": "diagnosis_code", "type": "string", "required": false, "description": "Related diagnosis code"},
        {"name": "notes", "type": "string", "required": false, "description": "Additional notes"}
    ]
}'::jsonb
FROM document_types dt
JOIN verticals v ON dt.vertical_id = v.id
WHERE v.name = 'medical' AND dt.name = 'medical_actions'
ON CONFLICT (document_type_id, version) DO NOTHING;
