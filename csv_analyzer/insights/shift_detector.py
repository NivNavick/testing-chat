"""
Automatic Shift Pattern Detection.

Clusters shifts based on start times to detect patterns like:
- MORNING (05:00-09:00 start)
- MID_DAY (09:00-13:00 start)
- AFTERNOON (13:00-17:00 start)
- EVENING (17:00-21:00 start)
- NIGHT (21:00-05:00 start)

Uses K-Means clustering with automatic optimal cluster detection.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ShiftPattern:
    """Detected shift pattern."""
    cluster_id: int
    label: str  # MORNING, AFTERNOON, etc.
    center_time: str  # "07:30"
    center_minutes: int  # 450
    shift_count: int
    typical_start: str
    typical_end: str
    avg_duration_hours: float


@dataclass
class ShiftAssignment:
    """Assignment of a single shift to a cluster."""
    shift_type: str
    cluster_id: int
    confidence: float  # 0.0 to 1.0
    evidence: str


class ShiftDetector:
    """
    Detects shift patterns from time data using clustering.
    
    Usage:
        detector = ShiftDetector()
        
        # Enrich a DataFrame with detected shift types
        df = detector.enrich_dataframe(df, start_col='shift_start', end_col='shift_end')
        
        # Get detected patterns
        patterns = detector.get_patterns()
    """
    
    # Time ranges for labeling clusters
    SHIFT_LABELS = [
        (5 * 60, 9 * 60, "MORNING"),      # 05:00-09:00
        (9 * 60, 13 * 60, "MID_DAY"),     # 09:00-13:00
        (13 * 60, 17 * 60, "AFTERNOON"),  # 13:00-17:00
        (17 * 60, 21 * 60, "EVENING"),    # 17:00-21:00
        (21 * 60, 29 * 60, "NIGHT"),      # 21:00-05:00 (next day)
        (0, 5 * 60, "NIGHT"),             # 00:00-05:00 (also night)
    ]
    
    def __init__(self, max_clusters: int = 4, min_cluster_size: int = 2):
        """
        Initialize the shift detector.
        
        Args:
            max_clusters: Maximum number of shift patterns to detect
            min_cluster_size: Minimum shifts needed to form a cluster
        """
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self._patterns: List[ShiftPattern] = []
        self._cluster_centers: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
    
    def enrich_dataframe(
        self,
        df: pd.DataFrame,
        start_col: str = "shift_start",
        end_col: str = "shift_end",
    ) -> pd.DataFrame:
        """
        Add shift detection columns to a DataFrame.
        
        Adds:
        - detected_shift_type: MORNING, AFTERNOON, EVENING, NIGHT
        - shift_cluster_id: Raw cluster number
        - shift_start_minutes: Minutes since midnight
        - shift_detection_confidence: Confidence score (0-1)
        - shift_detection_evidence: Explanation
        
        Args:
            df: DataFrame with shift data
            start_col: Column name for shift start time
            end_col: Column name for shift end time
            
        Returns:
            DataFrame with new columns
        """
        if start_col not in df.columns:
            logger.warning(f"Column '{start_col}' not found, skipping shift detection")
            return df
        
        df = df.copy()
        
        # Convert times to minutes
        df["shift_start_minutes"] = df[start_col].apply(self._time_to_minutes)
        
        # Get valid times for clustering
        valid_mask = df["shift_start_minutes"].notna()
        valid_times = df.loc[valid_mask, "shift_start_minutes"].values
        
        if len(valid_times) < self.min_cluster_size:
            logger.warning(f"Not enough valid times for clustering ({len(valid_times)} < {self.min_cluster_size})")
            df["detected_shift_type"] = "UNKNOWN"
            df["shift_cluster_id"] = -1
            df["shift_detection_confidence"] = 0.0
            df["shift_detection_evidence"] = "Insufficient data for shift detection"
            return df
        
        # Perform clustering
        labels, centers, confidences = self._cluster_times(valid_times)
        
        # Map cluster IDs to shift labels
        cluster_to_label = self._label_clusters(centers)
        
        # Build patterns
        self._build_patterns(df, valid_mask, labels, centers, cluster_to_label, start_col, end_col)
        
        # Assign results to DataFrame
        df["shift_cluster_id"] = -1
        df["detected_shift_type"] = "UNKNOWN"
        df["shift_detection_confidence"] = 0.0
        df["shift_detection_evidence"] = ""
        
        valid_indices = df.index[valid_mask]
        for i, idx in enumerate(valid_indices):
            cluster_id = labels[i]
            shift_type = cluster_to_label.get(cluster_id, "UNKNOWN")
            confidence = confidences[i]
            start_time = df.loc[idx, start_col]
            center_time = self._minutes_to_time(int(centers[cluster_id]))
            
            df.loc[idx, "shift_cluster_id"] = cluster_id
            df.loc[idx, "detected_shift_type"] = shift_type
            df.loc[idx, "shift_detection_confidence"] = round(confidence, 2)
            df.loc[idx, "shift_detection_evidence"] = (
                f"Start time {start_time} clustered as {shift_type} "
                f"(center: {center_time}, confidence: {confidence:.0%})"
            )
        
        logger.info(f"Detected {len(set(labels))} shift patterns: {list(cluster_to_label.values())}")
        
        return df
    
    def _time_to_minutes(self, time_val: Any) -> Optional[int]:
        """Convert time value to minutes since midnight."""
        if pd.isna(time_val):
            return None
        
        try:
            if isinstance(time_val, str):
                # Handle "HH:MM" or "HH:MM:SS" format
                parts = time_val.split(":")
                hours = int(parts[0])
                minutes = int(parts[1]) if len(parts) > 1 else 0
                return hours * 60 + minutes
            elif hasattr(time_val, 'hour'):
                # datetime.time or similar
                return time_val.hour * 60 + time_val.minute
            else:
                return None
        except (ValueError, AttributeError):
            return None
    
    def _minutes_to_time(self, minutes: int) -> str:
        """Convert minutes since midnight to HH:MM string."""
        hours = (minutes // 60) % 24
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    
    def _cluster_times(
        self,
        times: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Cluster time values using K-Means.
        
        Returns:
            labels: Cluster assignment for each time
            centers: Cluster center times (in minutes)
            confidences: Confidence score for each assignment
        """
        # Reshape for sklearn
        X = times.reshape(-1, 1)
        
        # Determine optimal number of clusters
        n_samples = len(times)
        n_clusters = min(self.max_clusters, n_samples // self.min_cluster_size)
        n_clusters = max(1, n_clusters)
        
        # Try to find optimal k using simple heuristic
        # (Could use elbow method or silhouette, but keeping it simple)
        unique_times = len(np.unique(times))
        if unique_times < n_clusters:
            n_clusters = unique_times
        
        # Use K-Means clustering
        try:
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
            )
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_.flatten()
            
            # Calculate confidence based on distance to center
            distances = np.abs(times - centers[labels])
            max_distance = 120  # 2 hours = low confidence
            confidences = np.clip(1 - (distances / max_distance), 0, 1)
            
        except ImportError:
            # Fallback to simple time-based rules if sklearn not available
            logger.warning("sklearn not available, using rule-based shift detection")
            labels, centers, confidences = self._rule_based_detection(times)
        
        return labels, centers, confidences
    
    def _rule_based_detection(
        self,
        times: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback rule-based detection without sklearn."""
        labels = np.zeros(len(times), dtype=int)
        confidences = np.ones(len(times))
        
        for i, t in enumerate(times):
            if 5 * 60 <= t < 12 * 60:
                labels[i] = 0  # MORNING
            elif 12 * 60 <= t < 17 * 60:
                labels[i] = 1  # AFTERNOON
            elif 17 * 60 <= t < 21 * 60:
                labels[i] = 2  # EVENING
            else:
                labels[i] = 3  # NIGHT
        
        # Calculate centers from actual data
        unique_labels = np.unique(labels)
        centers = np.array([times[labels == l].mean() for l in range(4)])
        
        return labels, centers, confidences
    
    def _label_clusters(self, centers: np.ndarray) -> Dict[int, str]:
        """Assign human-readable labels to cluster centers."""
        cluster_labels = {}
        
        for cluster_id, center in enumerate(centers):
            center_minutes = int(center)
            
            # Find matching label based on time range
            label = "UNKNOWN"
            for start, end, name in self.SHIFT_LABELS:
                if start <= center_minutes < end:
                    label = name
                    break
            
            cluster_labels[cluster_id] = label
        
        return cluster_labels
    
    def _build_patterns(
        self,
        df: pd.DataFrame,
        valid_mask: pd.Series,
        labels: np.ndarray,
        centers: np.ndarray,
        cluster_to_label: Dict[int, str],
        start_col: str,
        end_col: str,
    ):
        """Build ShiftPattern objects for detected clusters."""
        self._patterns = []
        
        valid_df = df.loc[valid_mask].copy()
        valid_df["_cluster"] = labels
        
        for cluster_id in range(len(centers)):
            cluster_data = valid_df[valid_df["_cluster"] == cluster_id]
            
            if len(cluster_data) == 0:
                continue
            
            # Calculate statistics
            start_times = cluster_data[start_col].apply(self._time_to_minutes)
            
            # Try to calculate duration
            if end_col in cluster_data.columns:
                end_times = cluster_data[end_col].apply(self._time_to_minutes)
                durations = (end_times - start_times) / 60  # hours
                # Handle overnight shifts
                durations = durations.apply(lambda x: x if x > 0 else x + 24)
                avg_duration = durations.mean()
            else:
                avg_duration = 8.0  # Default assumption
            
            pattern = ShiftPattern(
                cluster_id=cluster_id,
                label=cluster_to_label.get(cluster_id, "UNKNOWN"),
                center_time=self._minutes_to_time(int(centers[cluster_id])),
                center_minutes=int(centers[cluster_id]),
                shift_count=len(cluster_data),
                typical_start=self._minutes_to_time(int(start_times.median())),
                typical_end=self._minutes_to_time(int(start_times.median() + avg_duration * 60)),
                avg_duration_hours=round(avg_duration, 1),
            )
            self._patterns.append(pattern)
        
        # Sort by center time
        self._patterns.sort(key=lambda p: p.center_minutes)
    
    def get_patterns(self) -> List[ShiftPattern]:
        """Get detected shift patterns."""
        return self._patterns
    
    def get_pattern_summary(self) -> pd.DataFrame:
        """Get patterns as a DataFrame for reporting."""
        if not self._patterns:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                "shift_type": p.label,
                "cluster_id": p.cluster_id,
                "center_time": p.center_time,
                "typical_start": p.typical_start,
                "typical_end": p.typical_end,
                "avg_duration_hours": p.avg_duration_hours,
                "shift_count": p.shift_count,
            }
            for p in self._patterns
        ])

