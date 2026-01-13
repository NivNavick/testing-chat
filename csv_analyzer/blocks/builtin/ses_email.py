"""
SES Email Block - Send workflow data via AWS SES.

This is a "sink" block that accepts data inputs and sends them
as CSV attachments via AWS Simple Email Service (SES).
"""

import io
import logging
import os
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Union

import boto3
import pandas as pd

from csv_analyzer.workflows.block import BlockRegistry
from csv_analyzer.workflows.ontology import DataType
from csv_analyzer.workflows.base_block import BaseBlock, BlockContext

logger = logging.getLogger(__name__)


class SESEmailBlock(BaseBlock):
    """
    Send workflow data via AWS SES email.
    
    This block accepts data inputs (single or list) and sends them
    as CSV attachments in a single email.
    
    The sender email must be verified in AWS SES.
    For testing in SES Sandbox mode, recipient emails must also be verified.
    
    Inputs:
    - data: Single URI or list of URIs from previous blocks
    
    The block has NO outputs - it's a terminal/sink block that sends email.
    """
    
    def run(self) -> Dict[str, str]:
        """
        Send data as email attachments via AWS SES.
        
        Returns:
            Dict with email status and message ID
        """
        # Get required parameters
        sender_email = self.require_param("sender_email")
        recipient_email = self.require_param("recipient_email")
        
        # Get optional parameters
        subject = self.get_param("subject", "Workflow Data Export")
        body_text = self.get_param("body_text", None)
        body_html = self.get_param("body_html", None)
        attachment_prefix = self.get_param("attachment_filename_prefix", "data")
        include_summary = self.get_param("include_summary", True)
        aws_region = self.get_param("aws_region") or os.environ.get("AWS_REGION", "us-east-1")
        
        # Normalize recipient to list
        if isinstance(recipient_email, str):
            recipients = [recipient_email]
        else:
            recipients = recipient_email
        
        self.logger.info(f"Preparing email from {sender_email} to {recipients}")
        
        # Load input data
        data_input = self.ctx.inputs.get("data")
        if not data_input:
            raise ValueError("No 'data' input provided to email block")
        
        # Normalize to list
        if isinstance(data_input, list):
            data_uris = data_input
        else:
            data_uris = [data_input]
        
        self.logger.info(f"Loading {len(data_uris)} data source(s) for email attachments...")
        
        # Load all data sources
        dataframes = []
        for idx, uri in enumerate(data_uris, start=1):
            try:
                data = self._load_input_data(uri)
                
                if isinstance(data, pd.DataFrame):
                    df = data
                elif isinstance(data, list) and data and isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    self.logger.warning(f"Skipping unsupported data type: {type(data)}")
                    continue
                
                # Extract block ID from URI path for meaningful filename
                # URI format: .../workflow_run_id/block_id/output_name.json
                block_id = self._extract_block_id_from_uri(uri)
                filename = f"{block_id}.csv" if block_id else f"{attachment_prefix}_{idx}.csv"
                
                dataframes.append((filename, df))
                self.logger.info(f"  Loaded data source {idx}: {len(df)} rows, {len(df.columns)} columns")
                
            except Exception as e:
                self.logger.error(f"  Error loading data source {idx} from {uri}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No valid data to send via email")
        
        # Generate email body if not provided
        if body_text is None and include_summary:
            body_text = self._generate_summary_text(dataframes)
        elif body_text is None:
            body_text = "Please find the attached data files."
        
        # Create MIME email
        msg = self._create_mime_message(
            sender=sender_email,
            recipients=recipients,
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            attachments=dataframes,
        )
        
        # Send via SES
        try:
            ses_client = boto3.client('ses', region_name=aws_region)
            
            self.logger.info(f"Sending email via AWS SES (region: {aws_region})...")
            response = ses_client.send_raw_email(
                Source=sender_email,
                Destinations=recipients,
                RawMessage={'Data': msg.as_string()}
            )
            
            message_id = response['MessageId']
            self.logger.info(f"✅ Email sent successfully! MessageId: {message_id}")
            self.logger.info(f"   From: {sender_email}")
            self.logger.info(f"   To: {', '.join(recipients)}")
            self.logger.info(f"   Attachments: {len(dataframes)} CSV file(s)")
            
            return {
                "status": "success",
                "message_id": message_id,
                "recipient_count": len(recipients),
                "attachment_count": len(dataframes),
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send email: {e}")
            raise
    
    def _load_input_data(self, uri: str) -> Any:
        """Load data from URI (S3 or local)."""
        return self.load_from_s3(uri)
    
    def _extract_block_id_from_uri(self, uri: str) -> str:
        """
        Extract block ID from URI path.
        
        URI format examples:
        - /path/to/results/workflow_run_id/block_id/output_name.json
        - s3://bucket/workflows/workflow_run_id/block_id/output_name.json
        
        Returns the block_id or empty string if not found.
        """
        from pathlib import Path
        
        try:
            # Remove s3:// prefix if present
            path_str = uri.replace("s3://", "")
            
            # Get path parts
            parts = Path(path_str).parts
            
            # Look for the part before the filename (output_name.json)
            # The block_id is typically 2 levels up from the file
            if len(parts) >= 2:
                # parts[-1] is filename (e.g., result.json)
                # parts[-2] is block_id (e.g., early_arrival)
                block_id = parts[-2]
                return block_id
            
        except Exception as e:
            self.logger.warning(f"Could not extract block ID from URI {uri}: {e}")
        
        return ""
    
    def _generate_summary_text(self, dataframes: List[tuple]) -> str:
        """Generate a summary of the attached data."""
        total_rows = sum(df.shape[0] for _, df in dataframes)
        
        summary = f"Workflow Data Export\n"
        summary += f"{'=' * 50}\n\n"
        summary += f"Attached {len(dataframes)} data file(s) with {total_rows} total rows.\n\n"
        
        for filename, df in dataframes:
            summary += f"📎 {filename}\n"
            summary += f"   - Rows: {len(df):,}\n"
            summary += f"   - Columns: {len(df.columns)}\n"
            summary += f"   - Fields: {', '.join(df.columns[:5])}"
            if len(df.columns) > 5:
                summary += f" (+{len(df.columns) - 5} more)"
            summary += "\n\n"
        
        return summary
    
    def _create_mime_message(
        self,
        sender: str,
        recipients: List[str],
        subject: str,
        body_text: str,
        body_html: Union[str, None],
        attachments: List[tuple],
    ) -> MIMEMultipart:
        """
        Create a MIME multipart email with text body and CSV attachments.
        
        Args:
            sender: Sender email address
            recipients: List of recipient email addresses
            subject: Email subject line
            body_text: Plain text body
            body_html: HTML body (optional)
            attachments: List of (filename, DataFrame) tuples
            
        Returns:
            MIMEMultipart message ready to send
        """
        # Create message container
        msg = MIMEMultipart('mixed')
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = ', '.join(recipients)
        
        # Create message body container
        msg_body = MIMEMultipart('alternative')
        
        # Add text body
        text_part = MIMEText(body_text, 'plain', 'utf-8')
        msg_body.attach(text_part)
        
        # Add HTML body if provided
        if body_html:
            html_part = MIMEText(body_html, 'html', 'utf-8')
            msg_body.attach(html_part)
        
        msg.attach(msg_body)
        
        # Add CSV attachments
        for filename, df in attachments:
            # Convert DataFrame to CSV bytes
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8')
            csv_bytes = csv_buffer.getvalue().encode('utf-8')
            
            # Create attachment
            attachment = MIMEApplication(csv_bytes)
            attachment.add_header(
                'Content-Disposition',
                'attachment',
                filename=filename
            )
            attachment.add_header('Content-Type', 'text/csv')
            
            msg.attach(attachment)
            
            size_kb = len(csv_bytes) / 1024
            self.logger.info(f"  Attached: {filename} ({size_kb:.1f} KB)")
        
        return msg


# Register the block
@BlockRegistry.register(
    name="send_email_ses",
    inputs=[
        {
            "name": "data",
            "ontology": DataType.INSIGHT_RESULT,
            "required": True,
            "description": "Data to send as CSV attachments (single source or list of sources)"
        },
    ],
    outputs=[],  # No outputs - this is a sink block
    parameters=[
        {
            "name": "sender_email",
            "type": "string",
            "required": True,
            "description": "Sender email address (must be verified in AWS SES)"
        },
        {
            "name": "recipient_email",
            "type": "string",
            "required": True,
            "description": "Recipient email address(es) - string or list"
        },
        {
            "name": "subject",
            "type": "string",
            "default": "Workflow Data Export",
            "description": "Email subject line"
        },
        {
            "name": "body_text",
            "type": "string",
            "default": None,
            "description": "Plain text email body (auto-generated if not provided)"
        },
        {
            "name": "body_html",
            "type": "string",
            "default": None,
            "description": "HTML email body (optional)"
        },
        {
            "name": "attachment_filename_prefix",
            "type": "string",
            "default": "data",
            "description": "Prefix for attachment filenames (e.g., 'report' → report_1.csv, report_2.csv)"
        },
        {
            "name": "include_summary",
            "type": "boolean",
            "default": True,
            "description": "Include data summary in email body"
        },
        {
            "name": "aws_region",
            "type": "string",
            "default": None,
            "description": "AWS region for SES (defaults to AWS_REGION env var or us-east-1)"
        },
    ],
    block_class=SESEmailBlock,
    description="Send workflow data via AWS SES email with CSV attachments (sink block)",
)
def send_email_ses(ctx: BlockContext) -> Dict[str, str]:
    """Send data via AWS SES email."""
    return SESEmailBlock(ctx).run()

