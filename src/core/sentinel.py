"""
FleetSentinel Sentinel Core - Layer 1: Swarm Detection Engine

This module implements the core swarm detection logic using Redis geospatial
commands for efficient clustering and real-time threat assessment.

Architecture:
    - Uses Redis GEOADD/GEORADIUS for O(log(N)) spatial queries
    - Implements sliding time window with automatic TTL expiration
    - Calculates confidence scores using density, velocity, and topology

Performance:
    - Target: <50ms for cluster analysis
    - Designed for 100,000 concurrent requests per region
    - Graceful degradation on Redis failures

Author: FleetSentinel Security Team
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Set, Tuple, Any
import logging

import redis.asyncio as redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

from ..models import (
    RideRequest,
    GeoLocation,
    ClusterMetrics,
    ThreatAssessment,
    ThreatLevel,
    TopologyData,
    TopologyType,
    ContextData,
    DetectionConfig,
)


logger = logging.getLogger(__name__)


class RedisConnectionManager:
    """
    Manages Redis connections with automatic reconnection and pooling.
    
    Why: In a mission-critical system, we cannot afford to fail completely
    on temporary Redis issues. This manager provides connection resilience.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        max_connections: int = 100,
        socket_timeout: float = 5.0,
        retry_attempts: int = 3,
        retry_delay: float = 0.1,
    ):
        self._redis_url = redis_url
        self._max_connections = max_connections
        self._socket_timeout = socket_timeout
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay
        self._pool: Optional[redis.ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._lock = asyncio.Lock()
    
    async def get_client(self) -> redis.Redis:
        """
        Get a Redis client, creating connection pool if needed.
        
        Returns:
            Connected Redis client
            
        Raises:
            RedisConnectionError: If unable to connect after retries
        """
        if self._client is not None:
            return self._client
            
        async with self._lock:
            # Double-check after acquiring lock
            if self._client is not None:
                return self._client
                
            for attempt in range(self._retry_attempts):
                try:
                    self._pool = redis.ConnectionPool.from_url(
                        self._redis_url,
                        max_connections=self._max_connections,
                        socket_timeout=self._socket_timeout,
                        decode_responses=True,
                    )
                    self._client = redis.Redis(connection_pool=self._pool)
                    
                    # Verify connection
                    await self._client.ping()
                    logger.info("Redis connection established successfully")
                    return self._client
                    
                except RedisError as e:
                    logger.warning(
                        f"Redis connection attempt {attempt + 1} failed: {e}"
                    )
                    if attempt < self._retry_attempts - 1:
                        await asyncio.sleep(self._retry_delay * (attempt + 1))
                    else:
                        raise RedisConnectionError(
                            f"Failed to connect to Redis after {self._retry_attempts} attempts"
                        ) from e
        
        # Should never reach here
        raise RedisConnectionError("Unexpected state in connection manager")
    
    async def close(self) -> None:
        """Close Redis connections gracefully."""
        if self._client:
            await self._client.close()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None


class SwarmDetector:
    """
    The Sentinel Core - Detects abnormal clustering of ride requests.
    
    This class is the heart of FleetSentinel's detection capability.
    It maintains a Redis-backed geospatial index of recent requests
    and calculates threat confidence scores in real-time.
    
    Key Features:
        - Sliding time window (configurable, default 5 minutes)
        - Topology-weighted density calculation
        - Velocity of arrival analysis
        - User risk signal aggregation
    
    Confidence Score Formula:
        C_threat = sigmoid(α * ρ(x) * W_topo + δ_context + user_risk)
        
    Where:
        - ρ(x) = density at location x
        - W_topo = topology weight (dead-end = 2.5, arterial = 0.2)
        - δ_context = context modifier from Oracle services
        - α = scaling factor
    
    Example:
        detector = SwarmDetector(redis_url="redis://localhost:6379")
        await detector.initialize()
        
        assessment = await detector.analyze_request(ride_request)
        if assessment.threat_level == ThreatLevel.CRITICAL:
            # Block the request
            ...
    """
    
    # Redis key prefixes for namespace isolation
    GEO_KEY_PREFIX = "sentinel:geo:"
    REQUEST_KEY_PREFIX = "sentinel:req:"
    METRICS_KEY_PREFIX = "sentinel:metrics:"
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        config: Optional[DetectionConfig] = None,
    ):
        """
        Initialize the SwarmDetector.
        
        Args:
            redis_url: Redis connection URL
            config: Detection configuration (uses defaults if not provided)
        """
        self._redis_manager = RedisConnectionManager(redis_url)
        self._config = config or DetectionConfig()
        self._initialized = False
        
    async def initialize(self) -> None:
        """
        Initialize the detector and verify Redis connectivity.
        
        Call this before using any detection methods.
        """
        try:
            client = await self._redis_manager.get_client()
            await client.ping()
            self._initialized = True
            logger.info("SwarmDetector initialized successfully")
        except RedisError as e:
            logger.error(f"Failed to initialize SwarmDetector: {e}")
            raise
    
    async def close(self) -> None:
        """Clean up resources."""
        await self._redis_manager.close()
        self._initialized = False
    
    def _get_time_bucket(self, timestamp: datetime) -> str:
        """
        Get the time bucket key for a given timestamp.
        
        We partition data into 1-minute buckets for efficient cleanup.
        The sliding window spans multiple buckets.
        """
        bucket = timestamp.strftime("%Y%m%d%H%M")
        return f"{self.GEO_KEY_PREFIX}{bucket}"
    
    def _generate_request_key(self, request: RideRequest) -> str:
        """
        Generate a unique key for a request in Redis.
        
        Format: sentinel:req:{request_id}:{timestamp_ms}
        """
        ts_ms = int(request.timestamp.timestamp() * 1000)
        return f"{self.REQUEST_KEY_PREFIX}{request.request_id}:{ts_ms}"
    
    async def register_request(self, request: RideRequest) -> bool:
        """
        Register a new ride request in the geospatial index.
        
        This adds the request to the sliding time window and updates
        all relevant metrics. Called for every incoming request.
        
        Args:
            request: The ride request to register
            
        Returns:
            True if registered successfully, False on error
            
        Why register every request?
            We need full visibility to detect patterns. Even legitimate
            requests contribute to density calculations. We use TTL
            to automatically expire old data.
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            client = await self._redis_manager.get_client()
            geo_key = self._get_time_bucket(request.timestamp)
            
            # Use pipeline for atomic operations
            async with client.pipeline(transaction=True) as pipe:
                # Add to geospatial index
                # GEOADD key longitude latitude member
                await pipe.geoadd(
                    geo_key,
                    (
                        request.pickup_location.longitude,
                        request.pickup_location.latitude,
                        request.request_id,
                    ),
                )
                
                # Set TTL on geo key (time window + buffer)
                ttl_seconds = self._config.time_window_seconds + 60
                await pipe.expire(geo_key, ttl_seconds)
                
                # Store request metadata for user analysis
                request_key = self._generate_request_key(request)
                request_data = {
                    "request_id": request.request_id,
                    "user_id": request.user_id,
                    "timestamp": request.timestamp.isoformat(),
                    "lat": request.pickup_location.latitude,
                    "lon": request.pickup_location.longitude,
                    "ip_hash": request.user_ip_hash or "",
                    "device_fp": request.device_fingerprint or "",
                    "account_age": request.account_age_days,
                    "historical_rides": request.historical_rides,
                    "payment_verified": "1" if request.payment_method_verified else "0",
                }
                await pipe.hset(request_key, mapping=request_data)
                await pipe.expire(request_key, ttl_seconds)
                
                await pipe.execute()
                
            logger.debug(f"Registered request {request.request_id}")
            return True
            
        except RedisError as e:
            logger.error(f"Failed to register request: {e}")
            return False
    
    async def get_nearby_requests(
        self,
        location: GeoLocation,
        radius_meters: Optional[float] = None,
        time_window_seconds: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Get all requests within radius of a location in the time window.
        
        Args:
            location: Center point for search
            radius_meters: Search radius (default from config)
            time_window_seconds: Time window (default from config)
            
        Returns:
            List of (request_id, distance_meters) tuples
            
        Implementation:
            We query multiple time bucket keys (1 per minute) and
            aggregate the results. This allows efficient TTL cleanup
            while supporting arbitrary time windows.
        """
        if not self._initialized:
            await self.initialize()
            
        radius = radius_meters or self._config.detection_radius_meters
        window = time_window_seconds or self._config.time_window_seconds
        
        try:
            client = await self._redis_manager.get_client()
            
            # Calculate time buckets to query
            now = datetime.utcnow()
            buckets_to_query = []
            for minutes_ago in range(0, (window // 60) + 2):
                bucket_time = now - timedelta(minutes=minutes_ago)
                bucket_key = self._get_time_bucket(bucket_time)
                buckets_to_query.append(bucket_key)
            
            # Query each bucket and aggregate results
            all_results: List[Tuple[str, float]] = []
            
            for bucket_key in buckets_to_query:
                try:
                    # GEORADIUS key longitude latitude radius unit [WITHDIST]
                    results = await client.georadius(
                        bucket_key,
                        location.longitude,
                        location.latitude,
                        radius,
                        unit="m",
                        withdist=True,
                    )
                    
                    if results:
                        for item in results:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                request_id = item[0]
                                distance = float(item[1])
                                all_results.append((request_id, distance))
                                
                except RedisError as e:
                    # Log but continue - bucket might not exist
                    logger.debug(f"Bucket {bucket_key} query failed: {e}")
                    continue
            
            # Deduplicate by request_id (keep closest)
            seen: Dict[str, float] = {}
            for req_id, dist in all_results:
                if req_id not in seen or dist < seen[req_id]:
                    seen[req_id] = dist
            
            return [(k, v) for k, v in seen.items()]
            
        except RedisError as e:
            logger.error(f"Failed to get nearby requests: {e}")
            return []
    
    async def get_request_metadata(
        self, request_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve metadata for a list of request IDs.
        
        Used for user risk analysis (account age, device fingerprints, etc.)
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            client = await self._redis_manager.get_client()
            
            # Find request keys using pattern matching
            # Note: SCAN is used in production instead of KEYS
            results: Dict[str, Dict[str, Any]] = {}
            
            for req_id in request_ids:
                # Search for the request key
                pattern = f"{self.REQUEST_KEY_PREFIX}{req_id}:*"
                cursor = 0
                found = False
                
                while not found:
                    cursor, keys = await client.scan(
                        cursor=cursor, match=pattern, count=100
                    )
                    
                    for key in keys:
                        data = await client.hgetall(key)
                        if data:
                            results[req_id] = data
                            found = True
                            break
                    
                    if cursor == 0:
                        break
            
            return results
            
        except RedisError as e:
            logger.error(f"Failed to get request metadata: {e}")
            return {}
    
    async def calculate_cluster_metrics(
        self,
        location: GeoLocation,
        nearby_requests: List[Tuple[str, float]],
        time_window_seconds: int,
    ) -> ClusterMetrics:
        """
        Calculate detailed metrics for a cluster of requests.
        
        This analyzes the requests near a location to determine:
            - Total count and density
            - Unique users, IPs, and devices
            - New account ratio (attack indicator)
            - Unverified payment ratio
            
        Args:
            location: Center of the cluster
            nearby_requests: List of (request_id, distance) from get_nearby_requests
            time_window_seconds: Time window for analysis
            
        Returns:
            ClusterMetrics with all calculated values
        """
        request_count = len(nearby_requests)
        
        if request_count == 0:
            return ClusterMetrics(
                location=location,
                request_count=0,
                time_window_seconds=time_window_seconds,
            )
        
        # Get metadata for all requests
        request_ids = [req_id for req_id, _ in nearby_requests]
        metadata = await self.get_request_metadata(request_ids)
        
        # Analyze user signals
        unique_users: Set[str] = set()
        unique_ips: Set[str] = set()
        unique_devices: Set[str] = set()
        new_accounts = 0
        unverified_payments = 0
        
        for req_id, data in metadata.items():
            user_id = data.get("user_id", "")
            ip_hash = data.get("ip_hash", "")
            device_fp = data.get("device_fp", "")
            account_age = int(data.get("account_age", 0))
            payment_verified = data.get("payment_verified", "0") == "1"
            
            if user_id:
                unique_users.add(user_id)
            if ip_hash:
                unique_ips.add(ip_hash)
            if device_fp:
                unique_devices.add(device_fp)
            
            # New account = less than 7 days old
            if account_age < 7:
                new_accounts += 1
            
            if not payment_verified:
                unverified_payments += 1
        
        # Calculate rates
        arrival_rate = request_count / max(time_window_seconds, 1)
        new_accounts_ratio = new_accounts / max(request_count, 1)
        unverified_ratio = unverified_payments / max(request_count, 1)
        
        # Density score: requests per square kilometer
        # Using circular area with detection radius
        radius_km = self._config.detection_radius_meters / 1000
        area_km2 = math.pi * radius_km ** 2
        density_score = request_count / max(area_km2, 0.001)
        
        # Velocity score: how fast requests are arriving
        # Normalized against baseline (10 requests per 5 minutes is normal)
        baseline_rate = 10 / 300  # 0.033 per second
        velocity_score = arrival_rate / baseline_rate if baseline_rate > 0 else 0
        
        return ClusterMetrics(
            location=location,
            request_count=request_count,
            unique_users=len(unique_users),
            unique_ips=len(unique_ips),
            unique_devices=len(unique_devices),
            time_window_seconds=time_window_seconds,
            arrival_rate_per_second=arrival_rate,
            density_score=density_score,
            velocity_score=velocity_score,
            new_accounts_ratio=new_accounts_ratio,
            unverified_payment_ratio=unverified_ratio,
        )
    
    def _sigmoid(self, x: float) -> float:
        """
        Sigmoid activation function for confidence normalization.
        
        Maps any real number to (0, 1) range.
        """
        return 1 / (1 + math.exp(-x))
    
    def calculate_confidence(
        self,
        cluster_metrics: ClusterMetrics,
        topology_data: Optional[TopologyData] = None,
        context_data: Optional[ContextData] = None,
    ) -> float:
        """
        Calculate the Attack Confidence Score.
        
        This is THE critical calculation. It produces a score from 0.0 to 1.0
        indicating the probability that the detected cluster is an attack.
        
        Formula:
            C_threat = σ(α * (density_component + velocity_component) * W_topo + δ_context + user_risk)
            
        Components:
            - density_component: Based on density_score
            - velocity_component: Based on arrival rate
            - W_topo: Topology weight (dead-end = 2.5)
            - δ_context: Context modifier (concert = -5, protest = +5)
            - user_risk: Based on new accounts and unverified payments
        
        Args:
            cluster_metrics: Cluster analysis results
            topology_data: Street topology (optional)
            context_data: External context (optional)
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        config = self._config
        
        # =================================================================
        # Component 1: Density Score
        # Higher density = more suspicious
        # Threshold: >100 requests per km² is notable, >500 is suspicious
        # =================================================================
        density_raw = cluster_metrics.density_score
        density_normalized = min(density_raw / 500, 2.0)  # Cap at 2.0
        density_component = density_normalized * config.density_weight
        
        # =================================================================
        # Component 2: Velocity Score  
        # Fast arrival of requests is suspicious
        # Velocity > 5x baseline is notable, >10x is suspicious
        # =================================================================
        velocity_raw = cluster_metrics.velocity_score
        velocity_normalized = min(velocity_raw / 10, 2.0)  # Cap at 2.0
        velocity_component = velocity_normalized * config.velocity_weight
        
        # =================================================================
        # Component 3: Topology Weight
        # Dead-ends and alleys are high risk
        # Default to residential (1.0) if no topology data
        # =================================================================
        topo_weight = 1.0
        if topology_data:
            topo_weight = topology_data.weight
        topology_component = (density_component + velocity_component) * (topo_weight - 1.0)
        
        # =================================================================
        # Component 4: User Risk Signals
        # High ratio of new accounts or unverified payments is suspicious
        # =================================================================
        new_account_risk = cluster_metrics.new_accounts_ratio * 2.0
        unverified_risk = cluster_metrics.unverified_payment_ratio * 1.5
        user_risk_component = (new_account_risk + unverified_risk) * config.user_risk_weight
        
        # =================================================================
        # Component 5: Context Modifier
        # This can significantly reduce (concert) or increase (protest) score
        # =================================================================
        context_modifier = 0.0
        if context_data:
            context_modifier = context_data.modifier
        
        # =================================================================
        # Final Calculation
        # =================================================================
        raw_score = (
            (density_component + velocity_component + topology_component) 
            * config.alpha_scaling 
            + context_modifier * 0.1  # Scale context modifier
            + user_risk_component
        )
        
        # Apply sigmoid for normalization to 0-1
        confidence = self._sigmoid(raw_score - 0.5)  # Center around 0.5
        
        # Ensure bounds
        return max(0.0, min(1.0, confidence))
    
    def _get_threat_level(self, confidence: float) -> ThreatLevel:
        """
        Map confidence score to categorical threat level.
        """
        if confidence < 0.2:
            return ThreatLevel.NONE
        elif confidence < 0.4:
            return ThreatLevel.LOW
        elif confidence < 0.7:
            return ThreatLevel.MEDIUM
        elif confidence < 0.9:
            return ThreatLevel.HIGH
        else:
            return ThreatLevel.CRITICAL
    
    def _generate_explanation(
        self,
        cluster_metrics: ClusterMetrics,
        topology_data: Optional[TopologyData],
        context_data: Optional[ContextData],
        confidence: float,
        threat_level: ThreatLevel,
    ) -> str:
        """
        Generate human-readable explanation of the threat assessment.
        
        Why: Operators need to understand WHY a decision was made.
        This is critical for trust and incident response.
        """
        parts = []
        
        # Request count
        parts.append(
            f"Detected {cluster_metrics.request_count} requests "
            f"within {self._config.detection_radius_meters}m "
            f"over {cluster_metrics.time_window_seconds} seconds."
        )
        
        # Density analysis
        if cluster_metrics.density_score > 100:
            parts.append(
                f"Density: {cluster_metrics.density_score:.1f} reqs/km² (elevated)."
            )
        
        # Velocity analysis
        if cluster_metrics.velocity_score > 3:
            parts.append(
                f"Arrival rate: {cluster_metrics.velocity_score:.1f}x baseline (rapid)."
            )
        
        # Unique users analysis
        if cluster_metrics.unique_users < cluster_metrics.request_count * 0.5:
            parts.append(
                f"Only {cluster_metrics.unique_users} unique users "
                f"for {cluster_metrics.request_count} requests (suspicious)."
            )
        
        # Topology
        if topology_data:
            parts.append(f"Topology: {topology_data.topology_type.value}.")
            if topology_data.topology_type in (TopologyType.DEAD_END, TopologyType.ALLEY):
                parts.append("HIGH TRAP RISK: Limited escape routes.")
        
        # Context
        if context_data and context_data.context_type != ContextType.NONE:
            if context_data.event_name:
                parts.append(f"Context: {context_data.event_name} ({context_data.context_type.value}).")
            else:
                parts.append(f"Context: {context_data.context_type.value}.")
        
        # User risk
        if cluster_metrics.new_accounts_ratio > 0.3:
            parts.append(
                f"WARNING: {cluster_metrics.new_accounts_ratio*100:.0f}% new accounts."
            )
        
        # Final verdict
        parts.append(f"Threat Level: {threat_level.value} (Confidence: {confidence:.2%}).")
        
        return " ".join(parts)
    
    def _generate_recommendations(
        self,
        threat_level: ThreatLevel,
        topology_data: Optional[TopologyData],
    ) -> List[str]:
        """
        Generate actionable recommendations based on threat assessment.
        """
        recommendations = []
        
        if threat_level == ThreatLevel.NONE:
            recommendations.append("No action required. Continue normal operations.")
            
        elif threat_level == ThreatLevel.LOW:
            recommendations.append("Monitor situation. No immediate action required.")
            recommendations.append("Consider logging for pattern analysis.")
            
        elif threat_level == ThreatLevel.MEDIUM:
            recommendations.append("ACTIVATE LIABILITY GATE: Require security deposit.")
            recommendations.append("Alert nearest vehicles to be prepared for rerouting.")
            
        elif threat_level == ThreatLevel.HIGH:
            recommendations.append("BLOCK NEW REQUESTS to this area.")
            recommendations.append("Reroute vehicles already en route.")
            recommendations.append("Alert operations center for manual review.")
            
        elif threat_level == ThreatLevel.CRITICAL:
            recommendations.append("CRITICAL: Block all requests immediately.")
            recommendations.append("GEOFENCE: Create exclusion zone around location.")
            recommendations.append("Engage vehicle defense protocols for trapped assets.")
            recommendations.append("Notify security team and consider law enforcement alert.")
        
        # Topology-specific recommendations
        if topology_data and topology_data.topology_type in (TopologyType.DEAD_END, TopologyType.ALLEY):
            recommendations.append("TOPOLOGY ALERT: Dead-end location. Extra caution advised.")
        
        return recommendations
    
    async def analyze_request(
        self,
        request: RideRequest,
        topology_data: Optional[TopologyData] = None,
        context_data: Optional[ContextData] = None,
    ) -> ThreatAssessment:
        """
        Analyze a ride request and produce a threat assessment.
        
        This is the main entry point for the Sentinel Core.
        It registers the request, queries nearby requests,
        and calculates the confidence score.
        
        Args:
            request: The ride request to analyze
            topology_data: Street topology (optional, enhances accuracy)
            context_data: External context (optional, from Oracle layer)
            
        Returns:
            ThreatAssessment with confidence score and threat level
        """
        start_time = time.perf_counter()
        
        # Register this request in the sliding window
        await self.register_request(request)
        
        # Get all nearby requests in the time window
        nearby = await self.get_nearby_requests(
            location=request.pickup_location,
            radius_meters=self._config.detection_radius_meters,
            time_window_seconds=self._config.time_window_seconds,
        )
        
        # Calculate cluster metrics
        cluster_metrics = await self.calculate_cluster_metrics(
            location=request.pickup_location,
            nearby_requests=nearby,
            time_window_seconds=self._config.time_window_seconds,
        )
        
        # Calculate confidence score
        confidence = self.calculate_confidence(
            cluster_metrics=cluster_metrics,
            topology_data=topology_data,
            context_data=context_data,
        )
        
        # Determine threat level
        threat_level = self._get_threat_level(confidence)
        
        # Generate explanation and recommendations
        explanation = self._generate_explanation(
            cluster_metrics, topology_data, context_data, confidence, threat_level
        )
        recommendations = self._generate_recommendations(threat_level, topology_data)
        
        # Calculate component scores for transparency
        config = self._config
        density_component = min(cluster_metrics.density_score / 500, 2.0) * config.density_weight
        velocity_component = min(cluster_metrics.velocity_score / 10, 2.0) * config.velocity_weight
        topo_component = (topology_data.weight - 1.0) if topology_data else 0.0
        context_component = (context_data.modifier * 0.1) if context_data else 0.0
        user_risk = (cluster_metrics.new_accounts_ratio * 2.0 + 
                     cluster_metrics.unverified_payment_ratio * 1.5) * config.user_risk_weight
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            f"Request {request.request_id}: confidence={confidence:.3f}, "
            f"threat_level={threat_level.value}, elapsed_ms={elapsed_ms:.1f}"
        )
        
        return ThreatAssessment(
            request_id=request.request_id,
            timestamp=datetime.utcnow(),
            location=request.pickup_location,
            confidence_score=confidence,
            threat_level=threat_level,
            density_component=density_component,
            velocity_component=velocity_component,
            topology_component=topo_component,
            context_component=context_component,
            user_risk_component=user_risk,
            cluster_metrics=cluster_metrics,
            topology_data=topology_data,
            context_data=context_data,
            explanation=explanation,
            recommendations=recommendations,
        )
    
    async def analyze_location(
        self,
        location: GeoLocation,
        topology_data: Optional[TopologyData] = None,
        context_data: Optional[ContextData] = None,
    ) -> ThreatAssessment:
        """
        Analyze a location without registering a new request.
        
        Useful for:
            - Proactive monitoring of hot zones
            - Re-evaluating existing requests after context update
            - Admin dashboard queries
            
        Args:
            location: The location to analyze
            topology_data: Street topology (optional)
            context_data: External context (optional)
            
        Returns:
            ThreatAssessment for the location
        """
        # Get nearby requests without registering new one
        nearby = await self.get_nearby_requests(location)
        
        cluster_metrics = await self.calculate_cluster_metrics(
            location=location,
            nearby_requests=nearby,
            time_window_seconds=self._config.time_window_seconds,
        )
        
        confidence = self.calculate_confidence(
            cluster_metrics=cluster_metrics,
            topology_data=topology_data,
            context_data=context_data,
        )
        
        threat_level = self._get_threat_level(confidence)
        explanation = self._generate_explanation(
            cluster_metrics, topology_data, context_data, confidence, threat_level
        )
        recommendations = self._generate_recommendations(threat_level, topology_data)
        
        return ThreatAssessment(
            request_id="location-scan",
            location=location,
            confidence_score=confidence,
            threat_level=threat_level,
            cluster_metrics=cluster_metrics,
            topology_data=topology_data,
            context_data=context_data,
            explanation=explanation,
            recommendations=recommendations,
        )
