-- Optional Tables Support for Multi-File Insights
-- Migration: 003_optional_tables
-- Description: Adds optional_tables column to support context enrichment tables
-- 
-- This enables:
-- - Defining optional tables (like employee metadata) that enhance insights
-- - Multi-file correlation with employee validation
-- - Role-based filtering using employee context

-- ============================================================================
-- 1. Add optional_tables column to insight_definitions
-- ============================================================================
ALTER TABLE insight_definitions 
ADD COLUMN IF NOT EXISTS optional_tables TEXT[] DEFAULT '{}';

-- Comment explaining the column
COMMENT ON COLUMN insight_definitions.optional_tables IS 
'Optional tables that enhance the insight (e.g., employee_salary for role validation). 
Unlike required_tables, insights can run without these but will use them if available.';

-- ============================================================================
-- 2. Update the insight_summary view to include optional_tables
-- ============================================================================
-- Drop and recreate view since we're adding new columns
DROP VIEW IF EXISTS insight_summary;
CREATE VIEW insight_summary AS
SELECT 
    d.id,
    d.name,
    d.display_name,
    d.category,
    d.is_active,
    d.required_tables,
    d.optional_tables,
    COUNT(DISTINCT r.id) AS rule_count,
    COUNT(DISTINCT s.id) AS severity_count,
    d.created_at,
    d.updated_at
FROM insight_definitions d
LEFT JOIN insight_rules r ON d.id = r.insight_id
LEFT JOIN insight_severities s ON d.id = s.insight_id
GROUP BY d.id, d.name, d.display_name, d.category, d.is_active, 
         d.required_tables, d.optional_tables, d.created_at, d.updated_at;

