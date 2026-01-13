"""
Builtin Blocks.

Core blocks for workflow operations:
- upload: Upload files to S3
- preprocess: Data preprocessing and transformation
- classification: CSV classification and column canonization
- simple_categorize: Fast filename-based categorization (no ML)
- router: Route data by document type
- ses_email: Send data via AWS SES email
"""

# Import all builtin blocks to trigger registration
from csv_analyzer.blocks.builtin import upload
from csv_analyzer.blocks.builtin import preprocess
from csv_analyzer.blocks.builtin import classification
from csv_analyzer.blocks.builtin import simple_categorize
from csv_analyzer.blocks.builtin import router
from csv_analyzer.blocks.builtin import ses_email

