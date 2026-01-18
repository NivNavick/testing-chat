"""
Insight Blocks.

Analysis blocks that produce insight reports:
- sql_insight: Generic block to execute YAML-defined SQL insights (RECOMMENDED)
- early_arrival: Early arrival detection (DEPRECATED - use sql_insight)
- expensive_employees: Employee cost analysis (DEPRECATED - use sql_insight)
- manpower_cost: Manpower cost per shift (DEPRECATED - use sql_insight)
- gross_profit: Gross profit per shift (DEPRECATED - use sql_insight)

Migration Note:
  The pandas-based blocks have been replaced by SQL versions defined in YAML files.
  
  Use the sql_insight block with the appropriate insight_name parameter:
    - early_arrival
    - expensive_employees
    - manpower_cost
    - gross_profit
"""

# Import all insight blocks to trigger registration
from csv_analyzer.blocks.insights import sql_insight  # Generic SQL insight block (RECOMMENDED)
from csv_analyzer.blocks.insights import early_arrival  # DEPRECATED
from csv_analyzer.blocks.insights import expensive_employees  # DEPRECATED
from csv_analyzer.blocks.insights import manpower_cost  # DEPRECATED
from csv_analyzer.blocks.insights import gross_profit  # DEPRECATED

