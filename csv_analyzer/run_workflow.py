#!/usr/bin/env python3
"""
Workflow Runner - Execute DAG-based workflows.

Usage:
    # Run a workflow
    python -m csv_analyzer.run_workflow medical_early_arrival \\
        --files shifts.csv actions.csv \\
        --param max_early_minutes=45 \\
        --output results/

    # List available workflows
    python -m csv_analyzer.run_workflow --list

    # List available blocks
    python -m csv_analyzer.run_workflow --list-blocks

    # Show workflow details
    python -m csv_analyzer.run_workflow medical_early_arrival --show
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_param(param_str: str) -> tuple:
    """Parse key=value parameter string."""
    if "=" not in param_str:
        raise ValueError(f"Invalid parameter format: {param_str}. Expected key=value")
    
    key, value = param_str.split("=", 1)
    
    # Try to convert to appropriate type
    if value.lower() == "true":
        return key, True
    elif value.lower() == "false":
        return key, False
    elif value.isdigit():
        return key, int(value)
    else:
        try:
            return key, float(value)
        except ValueError:
            return key, value


def list_workflows(definitions_dir: Path) -> None:
    """List available workflows."""
    print("\nAvailable Workflows:")
    print("=" * 60)
    
    import yaml
    
    for yaml_file in sorted(definitions_dir.glob("*.yaml")):
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            name = data.get("name", yaml_file.stem)
            desc = data.get("description", "")
            version = data.get("version", "1.0")
            blocks = data.get("blocks", [])
            
            print(f"\n{name} (v{version})")
            print(f"  Description: {desc}")
            print(f"  Blocks: {len(blocks)}")
            print(f"  Pipeline: {' → '.join(b['id'] for b in blocks)}")
            
        except Exception as e:
            print(f"\n{yaml_file.stem}: Error - {e}")
    
    print()


def list_blocks() -> None:
    """List available blocks."""
    from csv_analyzer.workflows.block import BlockRegistry
    
    # Ensure blocks are loaded
    BlockRegistry._ensure_initialized()
    
    print("\nAvailable Blocks:")
    print("=" * 60)
    
    for name in sorted(BlockRegistry.list_blocks()):
        definition = BlockRegistry.get_definition(name)
        if definition:
            print(f"\n{name}")
            print(f"  Type: {definition.type}")
            if definition.description:
                print(f"  Description: {definition.description}")
            
            if definition.inputs:
                print(f"  Inputs:")
                for inp in definition.inputs:
                    req = " (required)" if inp.required else ""
                    print(f"    - {inp.name}: {inp.ontology.value}{req}")
            
            if definition.outputs:
                print(f"  Outputs:")
                for out in definition.outputs:
                    opt = " (optional)" if out.optional else ""
                    print(f"    - {out.name}: {out.ontology.value}{opt}")
            
            if definition.parameters:
                print(f"  Parameters:")
                for param in definition.parameters:
                    default = f" (default: {param.default})" if param.default is not None else ""
                    req = " (required)" if param.required else ""
                    print(f"    - {param.name}: {param.type}{req}{default}")
    
    print()


def show_workflow(workflow_path: Path) -> None:
    """Show workflow details."""
    import yaml
    
    with open(workflow_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    name = data.get("name", workflow_path.stem)
    desc = data.get("description", "")
    version = data.get("version", "1.0")
    params = data.get("parameters", {})
    blocks = data.get("blocks", [])
    
    print(f"\nWorkflow: {name}")
    print("=" * 60)
    print(f"Description: {desc}")
    print(f"Version: {version}")
    
    if params:
        print(f"\nGlobal Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    print(f"\nBlocks ({len(blocks)}):")
    for block in blocks:
        block_id = block.get("id", "?")
        handler = block.get("handler", "?")
        inputs = block.get("inputs", [])
        block_params = block.get("parameters", {})
        condition = block.get("condition")
        
        print(f"\n  [{block_id}] → {handler}")
        
        if inputs:
            print(f"    Inputs:")
            for inp in inputs:
                print(f"      - {inp['name']} ← {inp.get('source', 'N/A')}")
        
        if block_params:
            print(f"    Parameters:")
            for key, value in block_params.items():
                print(f"      - {key}: {value}")
        
        if condition:
            print(f"    Condition: {condition}")
    
    print()


def save_results(
    results: Dict[str, Dict[str, str]],
    output_dir: Path,
    workflow_name: str,
) -> None:
    """Save workflow results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results manifest
    manifest_path = output_dir / f"{workflow_name}_{timestamp}_results.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run DAG-based workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run a workflow with input files
    python -m csv_analyzer.run_workflow medical_early_arrival \\
        --files shifts.csv actions.csv \\
        --param max_early_minutes=45

    # List available workflows
    python -m csv_analyzer.run_workflow --list

    # List available blocks
    python -m csv_analyzer.run_workflow --list-blocks

    # Show workflow details
    python -m csv_analyzer.run_workflow medical_early_arrival --show
        """
    )
    
    parser.add_argument("workflow", nargs="?", help="Workflow name to run")
    parser.add_argument("--files", "-f", nargs="+", help="Input files")
    parser.add_argument("--param", "-p", action="append", help="key=value parameters")
    parser.add_argument("--output", "-o", default="./results", help="Output directory")
    parser.add_argument("--bucket", "-b", help="S3 bucket (overrides S3_BUCKET env var)")
    parser.add_argument("--local", "-l", action="store_true", help="Use local storage instead of S3")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Max parallel workers")
    parser.add_argument("--list", action="store_true", help="List available workflows")
    parser.add_argument("--list-blocks", action="store_true", help="List available blocks")
    parser.add_argument("--show", action="store_true", help="Show workflow details")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get definitions directory
    definitions_dir = Path(__file__).parent / "workflows" / "definitions"
    
    # Handle --list
    if args.list:
        list_workflows(definitions_dir)
        return
    
    # Handle --list-blocks
    if args.list_blocks:
        list_blocks()
        return
    
    # Require workflow name for other operations
    if not args.workflow:
        parser.error("Workflow name is required")
    
    # Find workflow file
    workflow_path = definitions_dir / f"{args.workflow}.yaml"
    if not workflow_path.exists():
        # Try without .yaml
        workflow_path = definitions_dir / args.workflow
        if not workflow_path.exists():
            print(f"Error: Workflow not found: {args.workflow}")
            print(f"Looked in: {definitions_dir}")
            print("\nUse --list to see available workflows")
            sys.exit(1)
    
    # Handle --show
    if args.show:
        show_workflow(workflow_path)
        return
    
    # Build parameters
    params: Dict[str, Any] = {}
    
    # Add files parameter
    if args.files:
        params["files"] = [str(Path(f).absolute()) for f in args.files]
    
    # Parse additional parameters
    for p in (args.param or []):
        try:
            key, value = parse_param(p)
            params[key] = value
        except ValueError as e:
            parser.error(str(e))
    
    # Set bucket
    if args.bucket:
        os.environ["S3_BUCKET"] = args.bucket
    
    print()
    print("=" * 60)
    print(f"Running Workflow: {args.workflow}")
    print("=" * 60)
    print(f"Input files: {args.files or 'None'}")
    print(f"Parameters: {params}")
    print(f"Output: {args.output}")
    print()
    
    # Import and run
    from csv_analyzer.workflows.engine import WorkflowEngine
    
    try:
        # Determine storage mode
        local_storage = str(Path(args.output).absolute()) if args.local else ""
        bucket = "" if args.local else (args.bucket or os.environ.get("S3_BUCKET", ""))
        
        engine = WorkflowEngine.from_yaml(
            str(workflow_path),
            bucket=bucket,
            max_workers=args.workers,
            local_storage_path=local_storage,
        )
        
        results = engine.run(params)
        
        # Save results
        output_dir = Path(args.output)
        save_results(results, output_dir, args.workflow)
        
        print()
        print("=" * 60)
        print("Workflow Completed Successfully")
        print("=" * 60)
        
        # Print summary
        for block_id, outputs in results.items():
            print(f"\n{block_id}:")
            for output_name, s3_uri in outputs.items():
                print(f"  {output_name}: {s3_uri}")
        
        print()
        
    except Exception as e:
        logger.exception("Workflow failed")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

