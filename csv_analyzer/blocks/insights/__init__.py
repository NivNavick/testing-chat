"""
Insight Blocks.

Analysis blocks that produce insight reports:
- early_arrival: Early arrival detection
- expensive_employees: Employee cost analysis
- manpower_cost: Manpower cost per shift and avg cost per procedure
- gross_profit: Gross profit per shift (revenue - costs)
- staff_coverage: Staff coverage validation against schedule and business rules
"""

# Import all insight blocks to trigger registration
from csv_analyzer.blocks.insights import early_arrival
from csv_analyzer.blocks.insights import expensive_employees
from csv_analyzer.blocks.insights import manpower_cost
from csv_analyzer.blocks.insights import gross_profit
from csv_analyzer.blocks.insights import staff_coverage

