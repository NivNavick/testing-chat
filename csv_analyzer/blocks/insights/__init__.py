"""
Insight Blocks.

Analysis blocks that produce insight reports:
- early_arrival: Early arrival detection
- expensive_employees: Employee cost analysis
"""

# Import all insight blocks to trigger registration
from csv_analyzer.blocks.insights import early_arrival
from csv_analyzer.blocks.insights import expensive_employees

