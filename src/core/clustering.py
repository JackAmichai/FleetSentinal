"""
FleetSentinel Clustering Module

This module provides additional clustering algorithms and density calculations
for the Sentinel Core. It implements kernel density estimation and can be
extended with DBSCAN or other spatial clustering methods.

The primary clustering is handled by Redis GEORADIUS in sentinel.py,
but this module provides more sophisticated analysis options.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime

from ..models import GeoLocation, RideRequest


@dataclass
class SpatiotemporalPoint:
    """
    A point in space and time for clustering analysis.
    
    Represents a ride request in the spatiotemporal domain.
    """
    request_id: str
    location: GeoLocation
    timestamp: datetime
    weight: float = 1.0  # Base weight, modified by topology
    
    # Metadata for analysis
    user_id: Optional[str] = None
    device_fingerprint: Optional[str] = None
    ip_hash: Optional[str] = None


@dataclass
class SpatiotemporalCluster:
    """
    A cluster of spatiotemporally related requests.
    
    This represents a group of requests that are close together
    in both space and time, potentially indicating a swarm attack.
    """
    cluster_id: str
    centroid: GeoLocation
    points: List[SpatiotemporalPoint] = field(default_factory=list)
    
    # Temporal bounds
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Cluster metrics
    spatial_radius_meters: float = 0.0
    temporal_span_seconds: float = 0.0
    
    @property
    def size(self) -> int:
        """Number of points in the cluster."""
        return len(self.points)
    
    @property
    def total_weight(self) -> float:
        """Sum of all point weights."""
        return sum(p.weight for p in self.points)
    
    @property
    def unique_users(self) -> int:
        """Count of unique user IDs."""
        users = {p.user_id for p in self.points if p.user_id}
        return len(users)
    
    @property
    def unique_devices(self) -> int:
        """Count of unique device fingerprints."""
        devices = {p.device_fingerprint for p in self.points if p.device_fingerprint}
        return len(devices)
    
    def calculate_density(self) -> float:
        """
        Calculate weighted density of the cluster.
        
        Returns requests per square kilometer, weighted.
        """
        if self.spatial_radius_meters <= 0:
            return 0.0
        
        radius_km = self.spatial_radius_meters / 1000
        area_km2 = math.pi * radius_km ** 2
        
        return self.total_weight / max(area_km2, 0.001)
    
    def calculate_velocity(self) -> float:
        """
        Calculate velocity of request arrival.
        
        Returns weighted requests per second.
        """
        if self.temporal_span_seconds <= 0:
            return 0.0
        
        return self.total_weight / self.temporal_span_seconds


class DensityCalculator:
    """
    Kernel Density Estimation for spatiotemporal analysis.
    
    Uses a Gaussian kernel to estimate the density of requests
    at any given point, creating a smooth density surface.
    
    This is useful for:
        - Visualizing density heat maps
        - Finding local maxima (hotspots)
        - More nuanced density estimation than simple counting
    """
    
    def __init__(
        self,
        spatial_bandwidth_meters: float = 500.0,
        temporal_bandwidth_seconds: float = 300.0,
    ):
        """
        Initialize the density calculator.
        
        Args:
            spatial_bandwidth_meters: Spatial kernel bandwidth (radius)
            temporal_bandwidth_seconds: Temporal kernel bandwidth (window)
        """
        self._spatial_h = spatial_bandwidth_meters
        self._temporal_h = temporal_bandwidth_seconds
    
    @staticmethod
    def haversine_distance(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Calculate the great-circle distance between two points.
        
        Uses the Haversine formula for accurate distances on a sphere.
        
        Args:
            lat1, lon1: First point coordinates (degrees)
            lat2, lon2: Second point coordinates (degrees)
            
        Returns:
            Distance in meters
        """
        R = 6371000  # Earth's radius in meters
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = (
            math.sin(delta_phi / 2) ** 2 +
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _gaussian_kernel(self, distance: float, bandwidth: float) -> float:
        """
        Gaussian kernel function.
        
        K(u) = (1/sqrt(2π)) * exp(-u²/2)
        
        Where u = distance / bandwidth
        """
        if bandwidth <= 0:
            return 0.0
        
        u = distance / bandwidth
        return math.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)
    
    def estimate_density(
        self,
        query_location: GeoLocation,
        query_time: datetime,
        points: List[SpatiotemporalPoint],
    ) -> float:
        """
        Estimate density at a query point using kernel density estimation.
        
        The density is a weighted sum of kernel contributions from all points,
        considering both spatial and temporal distance.
        
        Args:
            query_location: Location to estimate density at
            query_time: Time to estimate density at
            points: List of spatiotemporal points
            
        Returns:
            Estimated density (requests per unit area per unit time)
        """
        if not points:
            return 0.0
        
        total_density = 0.0
        
        for point in points:
            # Spatial kernel
            spatial_dist = self.haversine_distance(
                query_location.latitude,
                query_location.longitude,
                point.location.latitude,
                point.location.longitude,
            )
            spatial_kernel = self._gaussian_kernel(spatial_dist, self._spatial_h)
            
            # Temporal kernel
            temporal_dist = abs((query_time - point.timestamp).total_seconds())
            temporal_kernel = self._gaussian_kernel(temporal_dist, self._temporal_h)
            
            # Combined kernel with point weight
            total_density += point.weight * spatial_kernel * temporal_kernel
        
        # Normalize by number of points
        return total_density / len(points)
    
    def find_hotspots(
        self,
        points: List[SpatiotemporalPoint],
        grid_resolution_meters: float = 100.0,
        threshold: float = 0.5,
    ) -> List[Tuple[GeoLocation, float]]:
        """
        Find density hotspots in the point cloud.
        
        Creates a grid over the bounding box of points and finds
        locations where density exceeds the threshold.
        
        Args:
            points: List of spatiotemporal points
            grid_resolution_meters: Grid cell size
            threshold: Minimum density to be considered a hotspot
            
        Returns:
            List of (location, density) for hotspots
        """
        if len(points) < 2:
            return []
        
        # Find bounding box
        lats = [p.location.latitude for p in points]
        lons = [p.location.longitude for p in points]
        times = [p.timestamp for p in points]
        
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        mid_time = times[len(times) // 2]  # Use median time for query
        
        # Convert resolution to degrees (approximate)
        lat_step = grid_resolution_meters / 111000  # ~111km per degree latitude
        lon_step = grid_resolution_meters / (111000 * math.cos(math.radians((min_lat + max_lat) / 2)))
        
        hotspots: List[Tuple[GeoLocation, float]] = []
        
        lat = min_lat
        while lat <= max_lat:
            lon = min_lon
            while lon <= max_lon:
                query_loc = GeoLocation(latitude=lat, longitude=lon)
                density = self.estimate_density(query_loc, mid_time, points)
                
                if density >= threshold:
                    hotspots.append((query_loc, density))
                
                lon += lon_step
            lat += lat_step
        
        return hotspots


class ClusterAnalyzer:
    """
    Analyzes clusters for attack characteristics.
    
    This class looks for patterns that distinguish attacks from
    legitimate high-demand situations:
        - User account patterns (many new accounts)
        - Device fingerprint patterns (same devices)
        - IP address patterns (same IPs or VPN ranges)
        - Timing patterns (too regular or synchronized)
    """
    
    @staticmethod
    def analyze_user_patterns(cluster: SpatiotemporalCluster) -> Dict[str, float]:
        """
        Analyze user account patterns in a cluster.
        
        Returns:
            Dictionary of pattern metrics (0.0 = normal, 1.0 = suspicious)
        """
        if cluster.size == 0:
            return {
                "user_concentration": 0.0,
                "device_concentration": 0.0,
                "user_to_request_ratio": 0.0,
            }
        
        # User concentration: How many requests per unique user
        # Normal: ~1-2 requests per user
        # Suspicious: Many requests from few users
        requests_per_user = cluster.size / max(cluster.unique_users, 1)
        user_concentration = min(requests_per_user / 10, 1.0)  # Cap at 10 req/user
        
        # Device concentration
        requests_per_device = cluster.size / max(cluster.unique_devices, 1)
        device_concentration = min(requests_per_device / 10, 1.0)
        
        # User to request ratio (inverse)
        # Low ratio = few users making many requests = suspicious
        user_ratio = cluster.unique_users / max(cluster.size, 1)
        user_to_request_suspicious = 1.0 - min(user_ratio, 1.0)
        
        return {
            "user_concentration": user_concentration,
            "device_concentration": device_concentration,
            "user_to_request_ratio": user_to_request_suspicious,
        }
    
    @staticmethod
    def analyze_timing_patterns(cluster: SpatiotemporalCluster) -> Dict[str, float]:
        """
        Analyze timing patterns for signs of coordination.
        
        Coordinated attacks often show unnaturally regular timing.
        Legitimate demand is more stochastic.
        
        Returns:
            Dictionary of timing metrics
        """
        if len(cluster.points) < 3:
            return {
                "timing_regularity": 0.0,
                "burst_factor": 0.0,
            }
        
        # Sort points by time
        sorted_points = sorted(cluster.points, key=lambda p: p.timestamp)
        
        # Calculate inter-arrival times
        inter_arrivals = []
        for i in range(1, len(sorted_points)):
            delta = (sorted_points[i].timestamp - sorted_points[i-1].timestamp).total_seconds()
            inter_arrivals.append(delta)
        
        if not inter_arrivals:
            return {"timing_regularity": 0.0, "burst_factor": 0.0}
        
        # Coefficient of variation of inter-arrival times
        # Low CV = regular timing = suspicious
        mean_ia = sum(inter_arrivals) / len(inter_arrivals)
        if mean_ia > 0:
            variance = sum((ia - mean_ia) ** 2 for ia in inter_arrivals) / len(inter_arrivals)
            std_dev = math.sqrt(variance)
            cv = std_dev / mean_ia
        else:
            cv = 0
        
        # Regular timing: CV < 0.5 is suspicious
        timing_regularity = max(0, 1.0 - cv) if cv < 2 else 0
        
        # Burst factor: How much faster than expected
        # Expected: 1 request per 30 seconds for normal demand
        expected_rate = 1 / 30
        actual_rate = len(inter_arrivals) / cluster.temporal_span_seconds if cluster.temporal_span_seconds > 0 else 0
        burst_factor = min(actual_rate / expected_rate, 1.0) if expected_rate > 0 else 0
        
        return {
            "timing_regularity": timing_regularity,
            "burst_factor": burst_factor,
        }
    
    @staticmethod
    def calculate_attack_probability(cluster: SpatiotemporalCluster) -> float:
        """
        Calculate overall attack probability for a cluster.
        
        Combines user patterns and timing patterns into a single score.
        
        Returns:
            Probability from 0.0 to 1.0
        """
        analyzer = ClusterAnalyzer()
        
        user_patterns = analyzer.analyze_user_patterns(cluster)
        timing_patterns = analyzer.analyze_timing_patterns(cluster)
        
        # Weight factors
        weights = {
            "user_concentration": 0.25,
            "device_concentration": 0.20,
            "user_to_request_ratio": 0.20,
            "timing_regularity": 0.20,
            "burst_factor": 0.15,
        }
        
        total = 0.0
        for metric, weight in weights.items():
            value = user_patterns.get(metric, timing_patterns.get(metric, 0.0))
            total += value * weight
        
        return min(total, 1.0)
