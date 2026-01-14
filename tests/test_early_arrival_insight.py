"""
Unit tests for Early Arrival Insight Block.

Tests all status types: EARLY, EXCESS, NO_PROCEDURES, OK
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, patch

from csv_analyzer.blocks.insights.early_arrival import (
    EarlyArrivalBlock,
    parse_time,
    normalize_date,
    match_arrivals_to_procedures,
)
from csv_analyzer.workflows.base_block import BlockContext


class TestTimeAndDateParsing:
    """Test time and date parsing utilities."""
    
    def test_parse_time_standard_format(self):
        """Test parsing standard time formats."""
        result = parse_time("08:30")
        assert result is not None
        assert result.hour == 8
        assert result.minute == 30
    
    def test_parse_time_with_seconds(self):
        """Test parsing time with seconds."""
        result = parse_time("14:45:30")
        assert result is not None
        assert result.hour == 14
        assert result.minute == 45
    
    def test_parse_time_with_asterisk_prefix(self):
        """Test parsing time with asterisk prefix (common in raw data)."""
        result = parse_time("* 09:15")
        assert result is not None
        assert result.hour == 9
        assert result.minute == 15
    
    def test_parse_time_invalid(self):
        """Test parsing invalid time returns None."""
        assert parse_time("invalid") is None
        assert parse_time("") is None
        assert parse_time(None) is None
    
    def test_normalize_date_standard(self):
        """Test date normalization with standard format."""
        result = normalize_date("2025-12-25")
        assert result == "2025-12-25"
    
    def test_normalize_date_slash_format(self):
        """Test date normalization with slash format."""
        result = normalize_date("25/12/2025")
        assert result == "2025-12-25"
    
    def test_normalize_date_invalid(self):
        """Test invalid date returns the original string."""
        # normalize_date returns the string as-is if it can't parse it
        assert normalize_date("invalid") == "invalid"
        assert normalize_date("") == ""
        assert normalize_date(None) is None


class TestMatchingLogic:
    """Test arrival-to-procedure matching logic."""
    
    def test_match_early_arrival(self):
        """Test detecting EARLY status - arrived too early."""
        arrivals = [
            {
                "employee_name": "Sarah Cohen",
                "employee_id": "E001",
                "shift_date": "2025-12-01",
                "arrival_time": datetime(1900, 1, 1, 7, 0),  # 07:00
                "location": "Clinic A"
            }
        ]
        
        procedures = [
            {
                "treatment_name": "Colonoscopy",
                "procedure_date": "2025-12-01",
                "procedure_time": datetime(1900, 1, 1, 8, 30),  # 08:30
                "category": "Gastro",
                "treating_staff": "Dr. Smith",
                "location": "Clinic A"
            }
        ]
        
        matched, unmatched = match_arrivals_to_procedures(
            arrivals, procedures, max_early_minutes=30
        )
        
        # EARLY arrivals go to unmatched list
        assert len(unmatched) == 1
        assert unmatched[0]["status"] == "EARLY"
        assert unmatched[0]["minutes_early"] == 60  # 90 min gap - 30 threshold
        assert unmatched[0]["matched_procedure_time"] == "08:30"
        assert "Arrived 60 min early" in unmatched[0]["evidence"]
    
    def test_match_ok_arrival(self):
        """Test OK status - arrived within acceptable threshold."""
        arrivals = [
            {
                "employee_name": "Michael Brown",
                "employee_id": "E002",
                "shift_date": "2025-12-01",
                "arrival_time": datetime(1900, 1, 1, 8, 45),  # 08:45
                "location": "Clinic A"
            }
        ]
        
        procedures = [
            {
                "treatment_name": "Ultrasound",
                "procedure_date": "2025-12-01",
                "procedure_time": datetime(1900, 1, 1, 9, 0),  # 09:00
                "category": "Imaging",
                "treating_staff": "Tech Sarah",
                "location": "Clinic A"
            }
        ]
        
        matched, unmatched = match_arrivals_to_procedures(
            arrivals, procedures, max_early_minutes=30
        )
        
        # OK arrivals go to matched list
        assert len(matched) == 1
        assert matched[0]["status"] == "OK"
        assert matched[0]["minutes_before_procedure"] == 15  # 15 min before
        assert "covered procedure" in matched[0]["evidence"]
    
    def test_match_excess_staffing(self):
        """Test EXCESS status - all procedures already have staff."""
        # Two employees arrive, but only one procedure
        arrivals = [
            {
                "employee_name": "Anna Wilson",
                "employee_id": "E003",
                "shift_date": "2025-12-01",
                "arrival_time": datetime(1900, 1, 1, 8, 0),
                "location": "Clinic A"
            },
            {
                "employee_name": "David Miller",
                "employee_id": "E004",
                "shift_date": "2025-12-01",
                "arrival_time": datetime(1900, 1, 1, 8, 0),
                "location": "Clinic A"
            }
        ]
        
        procedures = [
            {
                "treatment_name": "Endoscopy",
                "procedure_date": "2025-12-01",
                "procedure_time": datetime(1900, 1, 1, 8, 15),
                "category": "Gastro",
                "treating_staff": "Dr. Johnson",
                "location": "Clinic A"
            }
        ]
        
        matched, unmatched = match_arrivals_to_procedures(
            arrivals, procedures, max_early_minutes=30
        )
        
        # First should be OK (matched), second should be EXCESS (unmatched)
        assert len(matched) == 1
        assert matched[0]["status"] == "OK"
        
        assert len(unmatched) == 1
        assert unmatched[0]["status"] == "EXCESS"
        assert "all procedures already have assigned staff" in unmatched[0]["evidence"]
    
    def test_match_no_procedures(self):
        """Test NO_PROCEDURES status - no procedures scheduled."""
        arrivals = [
            {
                "employee_name": "Rachel Green",
                "employee_id": "E005",
                "shift_date": "2025-12-25",  # Holiday
                "arrival_time": datetime(1900, 1, 1, 9, 0),
                "location": "Clinic A"
            }
        ]
        
        procedures = []  # No procedures scheduled
        
        matched, unmatched = match_arrivals_to_procedures(
            arrivals, procedures, max_early_minutes=30
        )
        
        # NO_PROCEDURES goes to unmatched list
        assert len(unmatched) == 1
        assert unmatched[0]["status"] == "NO_PROCEDURES"
        assert "no procedures found" in unmatched[0]["evidence"]
    
    def test_same_date_matching(self):
        """Test that arrivals and procedures on the same date match correctly."""
        arrivals = [
            {
                "employee_name": "John Doe",
                "employee_id": "E006",
                "shift_date": "2025-12-01",
                "arrival_time": datetime(1900, 1, 1, 8, 0),
                "location": "Clinic A"
            }
        ]
        
        procedures = [
            {
                "treatment_name": "X-Ray",
                "procedure_date": "2025-12-01",  # Same date
                "procedure_time": datetime(1900, 1, 1, 8, 30),
                "category": "Imaging",
                "treating_staff": "Tech Mike",
                "location": "Clinic A"
            }
        ]
        
        matched, unmatched = match_arrivals_to_procedures(
            arrivals, procedures, max_early_minutes=30
        )
        
        # Should match successfully
        assert len(matched) == 1
        assert matched[0]["status"] == "OK"
    
    def test_custom_threshold(self):
        """Test custom early arrival threshold with high threshold."""
        arrivals = [
            {
                "employee_name": "Test Employee",
                "employee_id": "E007",
                "shift_date": "2025-12-01",
                "arrival_time": datetime(1900, 1, 1, 8, 0),  # 08:00
                "location": "Clinic A"
            }
        ]
        
        procedures = [
            {
                "treatment_name": "Consultation",
                "procedure_date": "2025-12-01",
                "procedure_time": datetime(1900, 1, 1, 9, 0),  # 09:00
                "category": "General",
                "treating_staff": "Dr. Lee",
                "location": "Clinic A"
            }
        ]
        
        # With 70-minute threshold, 60-minute gap should be OK
        matched, unmatched = match_arrivals_to_procedures(
            arrivals, procedures, max_early_minutes=70
        )
        
        assert len(matched) == 1
        assert matched[0]["status"] == "OK"
        assert matched[0]["minutes_before_procedure"] == 60


class TestEarlyArrivalBlock:
    """Test the full EarlyArrivalBlock integration."""
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock BlockContext."""
        ctx = MagicMock(spec=BlockContext)
        ctx.inputs = {"data": "s3://bucket/data.json"}
        ctx.params = {"max_early_minutes": 30}
        ctx.workflow_run_id = "test_run"
        ctx.block_id = "early_arrival_test"
        ctx.bucket = ""
        ctx.local_storage_path = "/tmp/test"
        return ctx
    
    def test_block_with_early_arrivals(self, mock_context, tmp_path):
        """Test block with data containing early arrivals."""
        # Create test data
        shifts_df = pd.DataFrame([
            {
                "shift_start": "07:00",
                "shift_date": "2025-12-01",
                "employee_name": "Sarah Cohen",
                "employee_id": "E001"
            }
        ])
        
        procedures_df = pd.DataFrame([
            {
                "treatment_start_time": "08:30",
                "treatment_date": "2025-12-01",
                "treatment_name": "Colonoscopy",
                "treatment_category": "Gastro",
                "treating_staff": "Dr. Smith"
            }
        ])
        
        classified_data = {
            "employee_shifts": shifts_df,
            "medical_actions": procedures_df
        }
        
        # Mock methods
        block = EarlyArrivalBlock(mock_context)
        block.load_classified_data = MagicMock(return_value=classified_data)
        block.save_to_s3 = MagicMock(return_value="s3://bucket/result.json")
        
        # Run block
        result = block.run()
        
        # Verify results
        assert "result" in result
        assert result["result"] == "s3://bucket/result.json"
        
        # Verify save_to_s3 was called with a DataFrame
        save_call_args = block.save_to_s3.call_args[0]
        result_df = save_call_args[1]
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 1
        assert result_df.iloc[0]["status"] == "EARLY"
        assert result_df.iloc[0]["employee_name"] == "Sarah Cohen"
        assert result_df.iloc[0]["minutes_early"] == 60
    
    def test_block_missing_required_data(self, mock_context):
        """Test block gracefully handles missing required doc types."""
        classified_data = {
            "employee_monthly_salary": pd.DataFrame()  # Wrong doc type
        }
        
        block = EarlyArrivalBlock(mock_context)
        block.load_classified_data = MagicMock(return_value=classified_data)
        block.save_to_s3 = MagicMock(return_value="s3://bucket/result.json")
        
        result = block.run()
        
        # Should return skipped=True
        assert result["skipped"] is True
        assert "employee_shifts" in result["reason"]
        assert "medical_actions" in result["reason"]
    
    def test_block_with_multiple_statuses(self, mock_context):
        """Test block with data containing multiple status types."""
        shifts_df = pd.DataFrame([
            # EARLY
            {
                "shift_start": "07:00",
                "shift_date": "2025-12-01",
                "employee_name": "Early Employee",
                "employee_id": "E001"
            },
            # OK
            {
                "shift_start": "08:45",
                "shift_date": "2025-12-01",
                "employee_name": "On-Time Employee",
                "employee_id": "E002"
            },
            # NO_PROCEDURES
            {
                "shift_start": "09:00",
                "shift_date": "2025-12-25",
                "employee_name": "Holiday Employee",
                "employee_id": "E003"
            }
        ])
        
        procedures_df = pd.DataFrame([
            {
                "treatment_start_time": "08:30",
                "treatment_date": "2025-12-01",
                "treatment_name": "Procedure 1",
                "treatment_category": "Gastro",
                "treating_staff": "Dr. A"
            },
            {
                "treatment_start_time": "09:00",
                "treatment_date": "2025-12-01",
                "treatment_name": "Procedure 2",
                "treatment_category": "Imaging",
                "treating_staff": "Dr. B"
            }
        ])
        
        classified_data = {
            "employee_shifts": shifts_df,
            "medical_actions": procedures_df
        }
        
        block = EarlyArrivalBlock(mock_context)
        block.load_classified_data = MagicMock(return_value=classified_data)
        block.save_to_s3 = MagicMock(return_value="s3://bucket/result.json")
        
        result = block.run()
        
        # Get the DataFrame
        save_call_args = block.save_to_s3.call_args[0]
        result_df = save_call_args[1]
        
        # Should have all 3 shifts
        assert len(result_df) == 3
        
        # Check status distribution
        statuses = result_df["status"].value_counts().to_dict()
        assert statuses.get("EARLY", 0) >= 1
        assert statuses.get("OK", 0) >= 1
        assert statuses.get("NO_PROCEDURES", 0) >= 1
    
    def test_block_empty_shifts(self, mock_context):
        """Test block handles empty shifts gracefully."""
        # Create DataFrames with proper columns
        classified_data = {
            "employee_shifts": pd.DataFrame(columns=[
                "shift_start", "shift_date", "employee_name", "employee_id"
            ]),
            "medical_actions": pd.DataFrame([
                {
                    "treatment_start_time": "08:00",
                    "treatment_date": "2025-12-01",
                    "treatment_name": "Test",
                    "treatment_category": "Test",
                    "treating_staff": "Dr. Test"
                }
            ])
        }
        
        block = EarlyArrivalBlock(mock_context)
        block.load_classified_data = MagicMock(return_value=classified_data)
        block.save_to_s3 = MagicMock(return_value="s3://bucket/result.json")
        
        result = block.run()
        
        # Should return empty result
        assert "result" in result
        save_call_args = block.save_to_s3.call_args[0]
        result_df = save_call_args[1]
        assert len(result_df) == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_exact_threshold_boundary(self):
        """Test arrival exactly at the threshold boundary."""
        arrivals = [
            {
                "employee_name": "Boundary Test",
                "employee_id": "E999",
                "shift_date": "2025-12-01",
                "arrival_time": datetime(1900, 1, 1, 8, 30),  # 08:30
                "location": "Clinic A"
            }
        ]
        
        procedures = [
            {
                "treatment_name": "Test Procedure",
                "procedure_date": "2025-12-01",
                "procedure_time": datetime(1900, 1, 1, 9, 0),  # 09:00
                "category": "Test",
                "treating_staff": "Dr. Test",
                "location": "Clinic A"
            }
        ]
        
        # Gap = 30 minutes, threshold = 30 minutes
        # Should be OK (not early)
        matched, unmatched = match_arrivals_to_procedures(
            arrivals, procedures, max_early_minutes=30
        )
        
        assert len(matched) == 1
        assert matched[0]["status"] == "OK"
        assert matched[0]["minutes_before_procedure"] == 30
    
    def test_multiple_procedures_same_time(self):
        """Test matching with multiple procedures at the same time."""
        arrivals = [
            {
                "employee_name": "Nurse A",
                "employee_id": "E100",
                "shift_date": "2025-12-01",
                "arrival_time": datetime(1900, 1, 1, 8, 0),
                "location": "Clinic A"
            }
        ]
        
        procedures = [
            {
                "procedure_name": "Procedure 1",
                "procedure_date": "2025-12-01",
                "procedure_time": datetime(1900, 1, 1, 8, 30),
                "category": "Gastro",
                "treating_staff": "Dr. A",
                "location": "Clinic A"
            },
            {
                "procedure_name": "Procedure 2",
                "procedure_date": "2025-12-01",
                "procedure_time": datetime(1900, 1, 1, 8, 30),  # Same time
                "category": "Imaging",
                "treating_staff": "Dr. B",
                "location": "Clinic A"
            }
        ]
        
        matched, _ = match_arrivals_to_procedures(
            arrivals, procedures, max_early_minutes=30
        )
        
        # Should match to one of them
        assert len(matched) == 1
        assert matched[0]["status"] == "OK"
        assert matched[0]["matched_procedure_time"] == "08:30"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

