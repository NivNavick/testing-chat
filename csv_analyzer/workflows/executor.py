"""
Block Executor.

Dispatches and executes blocks based on their type:
- Code blocks: Python function handlers
- SQL blocks: DuckDB query execution
- Class-based blocks: BaseBlock subclasses
"""

import logging
from typing import Any, Dict, Optional

import duckdb
import pandas as pd

from csv_analyzer.workflows.block import BlockRegistry, BlockDefinition
from csv_analyzer.workflows.base_block import BlockContext

logger = logging.getLogger(__name__)


class BlockExecutor:
    """
    Executes blocks based on their type.
    
    Supports:
    - Code blocks: Call registered Python functions
    - SQL blocks: Execute SQL queries via DuckDB
    - Class-based blocks: Instantiate and run BaseBlock subclasses
    """
    
    def __init__(self, duckdb_connection: Optional[duckdb.DuckDBPyConnection] = None):
        """
        Initialize the executor.
        
        Args:
            duckdb_connection: Optional DuckDB connection for SQL blocks
        """
        self._connection = duckdb_connection
    
    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create DuckDB connection."""
        if self._connection is None:
            self._connection = duckdb.connect(":memory:")
        return self._connection
    
    def execute(
        self,
        block_def: BlockDefinition,
        ctx: BlockContext,
    ) -> Dict[str, str]:
        """
        Execute a block and return its outputs.
        
        Args:
            block_def: Block definition
            ctx: Block context with inputs and params
            
        Returns:
            Dict mapping output_name to S3 URI
        """
        logger.info(f"Executing block: {block_def.name} (type={block_def.type})")
        
        if block_def.type == "sql":
            return self._execute_sql_block(block_def, ctx)
        else:
            return self._execute_code_block(block_def, ctx)
    
    def _execute_code_block(
        self,
        block_def: BlockDefinition,
        ctx: BlockContext,
    ) -> Dict[str, str]:
        """
        Execute a code or builtin block.
        
        Args:
            block_def: Block definition
            ctx: Block context
            
        Returns:
            Dict mapping output_name to S3 URI
        """
        # Check for class-based block
        block_class = BlockRegistry.get_class(block_def.name)
        if block_class is not None:
            logger.debug(f"Using class-based block: {block_class.__name__}")
            instance = block_class(ctx)
            return instance.run()
        
        # Fall back to function handler
        handler = BlockRegistry.get(block_def.name)
        if handler is None:
            raise ValueError(f"No handler found for block: {block_def.name}")
        
        logger.debug(f"Using function handler: {handler.__name__}")
        return handler(ctx)
    
    def _execute_sql_block(
        self,
        block_def: BlockDefinition,
        ctx: BlockContext,
    ) -> Dict[str, str]:
        """
        Execute a SQL block using DuckDB.
        
        Args:
            block_def: Block definition with SQL template
            ctx: Block context with inputs and params
            
        Returns:
            Dict mapping output_name to S3 URI
        """
        if not block_def.sql:
            raise ValueError(f"SQL block {block_def.name} has no SQL defined")
        
        # Prepare SQL with parameter substitution
        sql = block_def.sql
        
        # Substitute parameters using {{param_name}} syntax
        for param_name, param_value in ctx.params.items():
            placeholder = f"{{{{{param_name}}}}}"
            if placeholder in sql:
                sql = sql.replace(placeholder, str(param_value))
        
        # Register input DataFrames as tables
        for input_name, s3_uri in ctx.inputs.items():
            # Load DataFrame from S3
            from csv_analyzer.workflows.base_block import BaseBlock
            
            class TempBlock(BaseBlock):
                def run(self):
                    pass
            
            temp = TempBlock(ctx)
            df = temp.load_from_s3(s3_uri)
            
            if isinstance(df, pd.DataFrame):
                table_name = f"{{{{input_table}}}}" if input_name == "input_data" else input_name
                self.connection.register(input_name, df)
                
                # Also register as {{input_table}} for convenience
                if input_name == "input_data":
                    sql = sql.replace("{{input_table}}", input_name)
        
        logger.debug(f"Executing SQL: {sql[:200]}...")
        
        # Execute SQL
        result_df = self.connection.execute(sql).fetchdf()
        
        logger.info(f"SQL result: {len(result_df)} rows")
        
        # Get output name from block definition
        output_name = block_def.outputs[0].name if block_def.outputs else "result"
        
        # Save result to S3
        from csv_analyzer.workflows.base_block import BaseBlock
        
        class ResultSaver(BaseBlock):
            def run(self):
                pass
        
        saver = ResultSaver(ctx)
        result_uri = saver.save_to_s3(output_name, result_df)
        
        return {output_name: result_uri}
    
    def register_dataframe(self, name: str, df: pd.DataFrame) -> None:
        """
        Register a DataFrame as a table in DuckDB.
        
        Args:
            name: Table name
            df: DataFrame to register
        """
        self.connection.register(name, df)
        logger.debug(f"Registered table: {name} ({len(df)} rows)")
    
    def execute_sql(self, sql: str) -> pd.DataFrame:
        """
        Execute raw SQL and return result.
        
        Args:
            sql: SQL query
            
        Returns:
            Result DataFrame
        """
        return self.connection.execute(sql).fetchdf()
    
    def close(self) -> None:
        """Close DuckDB connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

