"""
PostgreSQL Repository for Configurable Insights.

Provides CRUD operations for:
- Insight definitions
- Insight rules
- Severity mappings
- Execution audit logs
"""

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from csv_analyzer.storage.postgres_db import get_insights_pool, clean_for_json
from csv_analyzer.insights.configurable.models import (
    ConfigurableInsightDefinition,
    InsightRule,
    SeverityMapping,
    InsightExecution,
    InsightSummary,
    ValueType,
    Severity,
)

logger = logging.getLogger(__name__)


class ConfigurableInsightRepository:
    """
    Repository for managing configurable insight definitions.
    
    All CRUD operations are async using asyncpg.
    """
    
    # =========================================================================
    # Insight Definition Operations
    # =========================================================================
    
    @staticmethod
    async def get_insight(name: str) -> Optional[ConfigurableInsightDefinition]:
        """
        Get an insight definition by name, including all rules and severities.
        
        Args:
            name: Unique insight name
            
        Returns:
            Complete InsightDefinition with rules and severities, or None
        """
        pool = await get_insights_pool()
        async with pool.acquire() as conn:
            # Get the insight definition
            row = await conn.fetchrow(
                """
                SELECT id, name, display_name, description, goal, 
                       required_tables, optional_tables, category, is_active, 
                       created_at, updated_at
                FROM insight_definitions
                WHERE name = $1
                """,
                name
            )
            
            if not row:
                return None
            
            insight_id = row['id']
            
            # Get rules
            rule_rows = await conn.fetch(
                """
                SELECT id, rule_name, value_type, current_value, default_value,
                       min_value, max_value, description, ai_hint, display_order
                FROM insight_rules
                WHERE insight_id = $1
                ORDER BY display_order, rule_name
                """,
                insight_id
            )
            
            # Get severities
            severity_rows = await conn.fetch(
                """
                SELECT id, condition_name, severity, condition_description, display_order
                FROM insight_severities
                WHERE insight_id = $1
                ORDER BY display_order, condition_name
                """,
                insight_id
            )
        
        # Build the model
        rules = [
            InsightRule(
                id=r['id'],
                rule_name=r['rule_name'],
                value_type=ValueType(r['value_type']),
                current_value=r['current_value'],
                default_value=r['default_value'],
                min_value=r['min_value'],
                max_value=r['max_value'],
                description=r['description'],
                ai_hint=r['ai_hint'],
                display_order=r['display_order'] or 0
            )
            for r in rule_rows
        ]
        
        severities = [
            SeverityMapping(
                id=s['id'],
                condition_name=s['condition_name'],
                severity=Severity(s['severity']),
                condition_description=s['condition_description'],
                display_order=s['display_order'] or 0
            )
            for s in severity_rows
        ]
        
        return ConfigurableInsightDefinition(
            id=row['id'],
            name=row['name'],
            display_name=row['display_name'],
            description=row['description'],
            goal=row['goal'],
            required_tables=row['required_tables'],
            optional_tables=row['optional_tables'] or [],
            category=row['category'],
            is_active=row['is_active'],
            rules=rules,
            severities=severities,
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )
    
    @staticmethod
    async def get_insight_by_id(insight_id: UUID) -> Optional[ConfigurableInsightDefinition]:
        """Get insight by UUID."""
        pool = await get_insights_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT name FROM insight_definitions WHERE id = $1",
                insight_id
            )
        if row:
            return await ConfigurableInsightRepository.get_insight(row['name'])
        return None
    
    @staticmethod
    async def list_insights(
        category: Optional[str] = None,
        active_only: bool = True
    ) -> List[InsightSummary]:
        """
        List all insight definitions with summary info.
        
        Args:
            category: Optional filter by category
            active_only: If True, only return active insights
            
        Returns:
            List of InsightSummary objects
        """
        pool = await get_insights_pool()
        
        query = """
            SELECT 
                d.id, d.name, d.display_name, d.category, d.is_active,
                d.required_tables, d.optional_tables, d.created_at, d.updated_at,
                COUNT(DISTINCT r.id) AS rule_count,
                COUNT(DISTINCT s.id) AS severity_count
            FROM insight_definitions d
            LEFT JOIN insight_rules r ON d.id = r.insight_id
            LEFT JOIN insight_severities s ON d.id = s.insight_id
            WHERE 1=1
        """
        params = []
        param_idx = 1
        
        if active_only:
            query += f" AND d.is_active = ${param_idx}"
            params.append(True)
            param_idx += 1
        
        if category:
            query += f" AND d.category = ${param_idx}"
            params.append(category)
            param_idx += 1
        
        query += """
            GROUP BY d.id, d.name, d.display_name, d.category, d.is_active,
                     d.required_tables, d.optional_tables, d.created_at, d.updated_at
            ORDER BY d.category, d.name
        """
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        return [
            InsightSummary(
                id=r['id'],
                name=r['name'],
                display_name=r['display_name'],
                category=r['category'],
                is_active=r['is_active'],
                required_tables=r['required_tables'],
                optional_tables=r['optional_tables'] or [],
                rule_count=r['rule_count'],
                severity_count=r['severity_count'],
                created_at=r['created_at'],
                updated_at=r['updated_at']
            )
            for r in rows
        ]
    
    @staticmethod
    async def list_insights_for_tables(
        available_tables: List[str]
    ) -> List[InsightSummary]:
        """
        List insights whose required_tables are all available.
        
        Args:
            available_tables: Tables that have been uploaded to the session
            
        Returns:
            List of insights that can run with these tables
        """
        pool = await get_insights_pool()
        
        # PostgreSQL array containment: required_tables <@ available_tables
        # means all elements in required_tables are in available_tables
        query = """
            SELECT 
                d.id, d.name, d.display_name, d.category, d.is_active,
                d.required_tables, d.optional_tables, d.created_at, d.updated_at,
                COUNT(DISTINCT r.id) AS rule_count,
                COUNT(DISTINCT s.id) AS severity_count
            FROM insight_definitions d
            LEFT JOIN insight_rules r ON d.id = r.insight_id
            LEFT JOIN insight_severities s ON d.id = s.insight_id
            WHERE d.is_active = true
              AND d.required_tables <@ $1
            GROUP BY d.id, d.name, d.display_name, d.category, d.is_active,
                     d.required_tables, d.optional_tables, d.created_at, d.updated_at
            ORDER BY d.category, d.name
        """
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, available_tables)
        
        return [
            InsightSummary(
                id=r['id'],
                name=r['name'],
                display_name=r['display_name'],
                category=r['category'],
                is_active=r['is_active'],
                required_tables=r['required_tables'],
                optional_tables=r['optional_tables'] or [],
                rule_count=r['rule_count'],
                severity_count=r['severity_count'],
                created_at=r['created_at'],
                updated_at=r['updated_at']
            )
            for r in rows
        ]
    
    # =========================================================================
    # Rule Operations
    # =========================================================================
    
    @staticmethod
    async def update_rule(
        insight_name: str,
        rule_name: str,
        value: Any
    ) -> InsightRule:
        """
        Update a rule's current_value.
        
        Args:
            insight_name: Name of the insight
            rule_name: Name of the rule to update
            value: New value (will be validated and coerced)
            
        Returns:
            Updated InsightRule
            
        Raises:
            ValueError: If insight or rule not found, or value is invalid
        """
        # First get the current rule to validate
        insight = await ConfigurableInsightRepository.get_insight(insight_name)
        if not insight:
            raise ValueError(f"Insight '{insight_name}' not found")
        
        rule = insight.get_rule(rule_name)
        if not rule:
            raise ValueError(f"Rule '{rule_name}' not found in insight '{insight_name}'")
        
        # Validate the new value
        validated_value = rule.validate_new_value(value)
        
        # Update in database
        pool = await get_insights_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE insight_rules 
                SET current_value = $1, updated_at = NOW()
                WHERE insight_id = $2 AND rule_name = $3
                """,
                str(validated_value),
                insight.id,
                rule_name
            )
        
        # Return updated rule
        rule.current_value = str(validated_value)
        return rule
    
    @staticmethod
    async def reset_rule(insight_name: str, rule_name: str) -> InsightRule:
        """
        Reset a rule to its default value.
        
        Args:
            insight_name: Name of the insight
            rule_name: Name of the rule to reset
            
        Returns:
            Reset InsightRule
        """
        insight = await ConfigurableInsightRepository.get_insight(insight_name)
        if not insight:
            raise ValueError(f"Insight '{insight_name}' not found")
        
        rule = insight.get_rule(rule_name)
        if not rule:
            raise ValueError(f"Rule '{rule_name}' not found in insight '{insight_name}'")
        
        pool = await get_insights_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE insight_rules 
                SET current_value = default_value, updated_at = NOW()
                WHERE insight_id = $1 AND rule_name = $2
                """,
                insight.id,
                rule_name
            )
        
        rule.current_value = rule.default_value
        return rule
    
    @staticmethod
    async def reset_all_rules(insight_name: str) -> List[InsightRule]:
        """Reset all rules for an insight to their defaults."""
        insight = await ConfigurableInsightRepository.get_insight(insight_name)
        if not insight:
            raise ValueError(f"Insight '{insight_name}' not found")
        
        pool = await get_insights_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE insight_rules 
                SET current_value = default_value, updated_at = NOW()
                WHERE insight_id = $1
                """,
                insight.id
            )
        
        # Reload and return
        updated = await ConfigurableInsightRepository.get_insight(insight_name)
        return updated.rules if updated else []
    
    # =========================================================================
    # Execution Logging
    # =========================================================================
    
    @staticmethod
    async def log_execution(execution: InsightExecution) -> UUID:
        """
        Log an insight execution to the audit table.
        
        Args:
            execution: Execution details to log
            
        Returns:
            UUID of the created log entry
        """
        pool = await get_insights_pool()
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO insight_executions (
                    insight_id, session_id, rules_snapshot, correlations_found,
                    generated_sql, ai_explanation, total_records, flagged_records,
                    flags_by_severity, execution_time_ms, success, error_message
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING id
                """,
                execution.insight_id,
                execution.session_id,
                json.dumps(clean_for_json(execution.rules_snapshot)),
                json.dumps(clean_for_json(execution.correlations_found)),
                execution.generated_sql,
                execution.ai_explanation,
                execution.total_records,
                execution.flagged_records,
                json.dumps(clean_for_json(execution.flags_by_severity)),
                execution.execution_time_ms,
                execution.success,
                execution.error_message
            )
        
        return row['id']
    
    @staticmethod
    async def get_executions(
        insight_name: str,
        limit: int = 20
    ) -> List[InsightExecution]:
        """
        Get execution history for an insight.
        
        Args:
            insight_name: Name of the insight
            limit: Maximum number of executions to return
            
        Returns:
            List of InsightExecution, most recent first
        """
        # Get insight ID first
        pool = await get_insights_pool()
        async with pool.acquire() as conn:
            insight_row = await conn.fetchrow(
                "SELECT id FROM insight_definitions WHERE name = $1",
                insight_name
            )
            
            if not insight_row:
                return []
            
            rows = await conn.fetch(
                """
                SELECT id, insight_id, session_id, executed_at, rules_snapshot,
                       correlations_found, generated_sql, ai_explanation,
                       total_records, flagged_records, flags_by_severity,
                       execution_time_ms, success, error_message
                FROM insight_executions
                WHERE insight_id = $1
                ORDER BY executed_at DESC
                LIMIT $2
                """,
                insight_row['id'],
                limit
            )
        
        return [
            InsightExecution(
                id=r['id'],
                insight_id=r['insight_id'],
                session_id=r['session_id'],
                executed_at=r['executed_at'],
                rules_snapshot=json.loads(r['rules_snapshot']) if r['rules_snapshot'] else {},
                correlations_found=json.loads(r['correlations_found']) if r['correlations_found'] else [],
                generated_sql=r['generated_sql'],
                ai_explanation=r['ai_explanation'],
                total_records=r['total_records'],
                flagged_records=r['flagged_records'],
                flags_by_severity=json.loads(r['flags_by_severity']) if r['flags_by_severity'] else {},
                execution_time_ms=r['execution_time_ms'],
                success=r['success'],
                error_message=r['error_message']
            )
            for r in rows
        ]
    
    @staticmethod
    async def get_execution_by_session(
        session_id: UUID,
        insight_name: str
    ) -> Optional[InsightExecution]:
        """Get the most recent execution for a session and insight."""
        pool = await get_insights_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT e.id, e.insight_id, e.session_id, e.executed_at, e.rules_snapshot,
                       e.correlations_found, e.generated_sql, e.ai_explanation,
                       e.total_records, e.flagged_records, e.flags_by_severity,
                       e.execution_time_ms, e.success, e.error_message
                FROM insight_executions e
                JOIN insight_definitions d ON e.insight_id = d.id
                WHERE e.session_id = $1 AND d.name = $2
                ORDER BY e.executed_at DESC
                LIMIT 1
                """,
                session_id,
                insight_name
            )
        
        if not row:
            return None
        
        return InsightExecution(
            id=row['id'],
            insight_id=row['insight_id'],
            session_id=row['session_id'],
            executed_at=row['executed_at'],
            rules_snapshot=json.loads(row['rules_snapshot']) if row['rules_snapshot'] else {},
            correlations_found=json.loads(row['correlations_found']) if row['correlations_found'] else [],
            generated_sql=row['generated_sql'],
            ai_explanation=row['ai_explanation'],
            total_records=row['total_records'],
            flagged_records=row['flagged_records'],
            flags_by_severity=json.loads(row['flags_by_severity']) if row['flags_by_severity'] else {},
            execution_time_ms=row['execution_time_ms'],
            success=row['success'],
            error_message=row['error_message']
        )
    
    # =========================================================================
    # Activation/Deactivation
    # =========================================================================
    
    @staticmethod
    async def set_active(insight_name: str, is_active: bool) -> bool:
        """
        Activate or deactivate an insight.
        
        Args:
            insight_name: Name of the insight
            is_active: Whether to activate (True) or deactivate (False)
            
        Returns:
            True if successful
        """
        pool = await get_insights_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE insight_definitions 
                SET is_active = $1, updated_at = NOW()
                WHERE name = $2
                """,
                is_active,
                insight_name
            )
        
        return result == "UPDATE 1"

