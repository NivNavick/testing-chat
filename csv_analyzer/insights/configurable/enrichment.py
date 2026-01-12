"""
Employee Context Enrichment for Configurable Insights.

This module provides:
- Role-to-procedure mapping for fraud detection
- Employee metadata lookup
- Location normalization
- Context-aware query building
"""

import logging
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

import duckdb

logger = logging.getLogger(__name__)


# ============================================================================
# Role-to-Procedure Mappings
# ============================================================================

# Maps employee roles to the department/procedure keywords they should be associated with
ROLE_TO_PROCEDURES: Dict[str, List[str]] = {
    # Nurses by specialty
    "אח/אחות": ["general", "כללי"],
    "אח/אחות גסטרו": ["גסטרו", "gastro", "גסטרוסקופיה", "קולונוסקופיה"],
    "אחות מנומטריה": ["מנומטריה", "manometry"],
    
    # Medical staff
    "צוות רפואי": ["general", "medical", "רפואי"],
    
    # Technicians
    "טכנאית עיניים": ["עיניים", "eyes", "ophthalmology", "קפסולה"],
    "טכנאי רפואי": ["general", "medical"],
    
    # Administrative
    "מזכירה רפואית": ["admin", "הנהלה"],  # Usually not tied to specific procedures
}

# Hebrew city names for location normalization
HEBREW_CITIES = [
    "בת ים",
    "חדרה", 
    "תל אביב",
    "ירושלים",
    "חיפה",
    "באר שבע",
    "ראשון לציון",
    "פתח תקווה",
    "אשדוד",
    "נתניה",
]

# Department keywords for matching
DEPARTMENT_KEYWORDS = [
    "גסטרו",
    "עיניים",
    "קפסולה",
    "רפואה",
    "מדיקל",
]


# ============================================================================
# Employee Context Data Class
# ============================================================================

@dataclass
class EmployeeContext:
    """Context information about an employee from metadata table."""
    name: str
    role: str
    location_tag: str
    city: Optional[str] = None
    hourly_rate: Optional[float] = None
    expected_procedures: List[str] = field(default_factory=list)
    
    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "EmployeeContext":
        """Create from database row."""
        role = str(row.get("role", row.get("תפקיד", "")))
        name = str(row.get("employee_name", row.get("שם עובד", "")))
        location = str(row.get("location_tag", row.get("תג בשכר", "")))
        city = str(row.get("city", row.get("עיר", ""))) if row.get("city") or row.get("עיר") else None
        rate = row.get("hourly_rate", row.get("תעריף"))
        
        # Parse hourly rate if it's a string like "73/82"
        if rate:
            if isinstance(rate, str) and "/" in rate:
                # Take the first value for base rate
                rate = float(rate.split("/")[0])
            else:
                try:
                    rate = float(rate)
                except (ValueError, TypeError):
                    rate = None
        
        # Get expected procedures based on role
        expected = ROLE_TO_PROCEDURES.get(role, ["general"])
        
        return cls(
            name=name,
            role=role,
            location_tag=location,
            city=city,
            hourly_rate=rate,
            expected_procedures=expected
        )


# ============================================================================
# Employee Context Enricher
# ============================================================================

class EmployeeContextEnricher:
    """
    Enriches insight data with employee metadata.
    
    Uses employee metadata (from salary/HR files) to:
    1. Validate employee exists
    2. Get employee role and expected procedure types
    3. Normalize location information
    4. Build role-based filtering SQL
    """
    
    def __init__(self, duckdb_conn: duckdb.DuckDBPyConnection):
        """
        Initialize enricher with DuckDB connection.
        
        Args:
            duckdb_conn: DuckDB connection with loaded tables
        """
        self.conn = duckdb_conn
        self._employee_cache: Dict[str, EmployeeContext] = {}
    
    def find_metadata_table(self, available_tables: List[str]) -> Optional[str]:
        """
        Find the employee metadata table from available tables.
        
        Looks for tables that match employee_salary or similar patterns.
        
        Args:
            available_tables: List of table names in DuckDB
            
        Returns:
            Table name or None if not found
        """
        metadata_patterns = ["salary", "employee", "personnel", "hr", "staff"]
        
        for table in available_tables:
            table_lower = table.lower()
            for pattern in metadata_patterns:
                if pattern in table_lower:
                    # Verify it has employee-like columns
                    if self._has_employee_columns(table):
                        return table
        
        return None
    
    def _has_employee_columns(self, table_name: str) -> bool:
        """Check if table has columns that look like employee metadata."""
        try:
            schema_df = self.conn.execute(f"DESCRIBE {table_name}").fetchdf()
            columns = [col.lower() for col in schema_df['column_name'].tolist()]
            
            # Look for employee-related columns (Hebrew or English)
            employee_indicators = [
                "employee", "name", "שם", "עובד", "תפקיד", "role"
            ]
            
            matches = sum(1 for ind in employee_indicators 
                         if any(ind in col for col in columns))
            
            return matches >= 2  # At least 2 indicators
            
        except Exception as e:
            logger.warning(f"Could not check columns for {table_name}: {e}")
            return False
    
    def load_employees(self, metadata_table: str) -> Dict[str, EmployeeContext]:
        """
        Load all employees from metadata table.
        
        Args:
            metadata_table: Name of the employee metadata table
            
        Returns:
            Dict mapping employee name -> EmployeeContext
        """
        if self._employee_cache:
            return self._employee_cache
        
        try:
            # Get column names
            schema_df = self.conn.execute(f"DESCRIBE {metadata_table}").fetchdf()
            columns = schema_df['column_name'].tolist()
            
            # Find name column (could be in Hebrew or English)
            name_col = None
            for col in columns:
                if any(x in col.lower() for x in ["שם עובד", "employee_name", "name", "שם"]):
                    name_col = col
                    break
            
            if not name_col:
                logger.warning(f"Could not find employee name column in {metadata_table}")
                return {}
            
            # Fetch all employees
            df = self.conn.execute(f"SELECT * FROM {metadata_table}").fetchdf()
            
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                
                # Map column names to expected format
                mapped_row = self._map_columns(row_dict, columns)
                
                if mapped_row.get("employee_name") or mapped_row.get("שם עובד"):
                    context = EmployeeContext.from_row(mapped_row)
                    if context.name:
                        self._employee_cache[context.name] = context
            
            logger.info(f"Loaded {len(self._employee_cache)} employees from {metadata_table}")
            return self._employee_cache
            
        except Exception as e:
            logger.error(f"Failed to load employees from {metadata_table}: {e}")
            return {}
    
    def _map_columns(self, row: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
        """Map Hebrew column names to standard names."""
        mapping = {
            "שם עובד": "employee_name",
            "תפקיד": "role",
            "תג בשכר": "location_tag",
            "תעריף": "hourly_rate",
        }
        
        mapped = {}
        for col, value in row.items():
            # Keep original
            mapped[col] = value
            # Add mapped version if available
            if col in mapping:
                mapped[mapping[col]] = value
        
        return mapped
    
    def get_employee_context(self, employee_name: str) -> Optional[EmployeeContext]:
        """
        Get context for a specific employee.
        
        Args:
            employee_name: Name to look up
            
        Returns:
            EmployeeContext or None if not found
        """
        return self._employee_cache.get(employee_name)
    
    def extract_city(self, location_string: str) -> Optional[str]:
        """
        Extract Hebrew city name from a location string.
        
        Args:
            location_string: Location like "בת ים - עצמאיים - גיאפה"
            
        Returns:
            City name like "בת ים" or None
        """
        if not location_string:
            return None
        
        for city in HEBREW_CITIES:
            if city in location_string:
                return city
        
        return None
    
    def get_procedure_keywords_for_role(self, role: str) -> List[str]:
        """
        Get procedure keywords that match an employee's role.
        
        Args:
            role: Employee role like "אח/אחות גסטרו"
            
        Returns:
            List of keywords like ["גסטרו", "gastro"]
        """
        # Direct match
        if role in ROLE_TO_PROCEDURES:
            return ROLE_TO_PROCEDURES[role]
        
        # Partial match (role might have extra text)
        for known_role, keywords in ROLE_TO_PROCEDURES.items():
            if known_role in role or role in known_role:
                return keywords
        
        # Default to general
        return ["general"]
    
    def build_role_filter_sql(self, role: str, department_column: str) -> str:
        """
        Generate SQL WHERE clause to filter by employee role.
        
        Args:
            role: Employee role
            department_column: Column name for department in medical table
            
        Returns:
            SQL fragment like "department LIKE '%גסטרו%'"
        """
        keywords = self.get_procedure_keywords_for_role(role)
        
        if not keywords or "general" in keywords:
            # General role matches any department
            return "1=1"
        
        conditions = [f"{department_column} LIKE '%{kw}%'" for kw in keywords if kw != "general"]
        
        if not conditions:
            return "1=1"
        
        return f"({' OR '.join(conditions)})"
    
    def get_employees_by_city(self, city: str) -> List[EmployeeContext]:
        """
        Get all employees working at a specific city.
        
        Args:
            city: City name like "בת ים"
            
        Returns:
            List of EmployeeContext for employees at that city
        """
        return [
            emp for emp in self._employee_cache.values()
            if city in (emp.location_tag or "") or emp.city == city
        ]
    
    def build_enrichment_context(
        self,
        metadata_table: Optional[str] = None,
        shifts_table: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build context dictionary for AI prompt enrichment.
        
        Returns dict with:
        - employees: List of employee names and roles
        - role_mappings: Which roles match which procedures
        - cities: Available cities
        - sql_hints: Helpful SQL patterns
        """
        context = {
            "role_to_procedures": ROLE_TO_PROCEDURES,
            "cities": HEBREW_CITIES,
            "department_keywords": DEPARTMENT_KEYWORDS,
            "employees": [],
            "sql_hints": {
                "city_extraction": "Use substring matching: location LIKE '%בת ים%'",
                "role_filter": "Match role keywords to department: department LIKE '%גסטרו%'",
                "employee_join": "Join on employee name: shifts.employee = metadata.employee_name"
            }
        }
        
        # Add loaded employees
        for name, emp in self._employee_cache.items():
            context["employees"].append({
                "name": name,
                "role": emp.role,
                "city": emp.city or self.extract_city(emp.location_tag),
                "expected_procedures": emp.expected_procedures
            })
        
        return context
    
    def get_all_cities_from_employees(self) -> Set[str]:
        """Get all unique cities from loaded employees."""
        cities = set()
        for emp in self._employee_cache.values():
            if emp.city:
                cities.add(emp.city)
            else:
                city = self.extract_city(emp.location_tag)
                if city:
                    cities.add(city)
        return cities

