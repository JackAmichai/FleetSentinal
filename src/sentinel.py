"""
FleetSentinel - Main Decision Engine

This module ties all 4 layers together into a unified decision pipeline.
It provides the main entry point `evaluate_ride_request()` that orchestrates:
    - Layer 1: Swarm Detection (Sentinel Core)
    - Layer 2: Context Verification (Oracle)
    - Layer 3: Liability Enforcement (Gate)
    - Layer 4: Vehicle Defense Commands (Edge)

The pipeline produces a final Decision:
    - DISPATCH: Safe to proceed
    - REQUIRE_LIABILITY: Grey zone - require security deposit
    - BLOCK_DANGEROUS: Confirmed threat - reject
    - DEFER_TO_HUMAN: Edge case needs review

Performance Target: <200ms end-to-end latency
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

from .models import (
    RideRequest,
    GeoLocation,
    Decision,
    ThreatLevel,
    ThreatAssessment,
    SentinelDecision,
    VehicleState,
    TopologyData,
    TopologyType,
    ContextData,
    ContextType,
    DetectionConfig,
    LiabilityConfig,
)
from .core.sentinel import SwarmDetector
from .oracles.context_service import ContextVerificationService, ContextVerdict
from .gates.liability import LiabilityEnforcementMiddleware, LiabilityResult


logger = logging.getLogger(__name__)


class TopologyService:
    """
    Mock topology service for street type lookups.
    
    In production, this would integrate with:
        - OpenStreetMap Overpass API
        - HERE Maps API
        - Google Maps Roads API
        
    For now, we simulate topology based on mock data.
    """
    
    # Mock topology data for test locations
    MOCK_TOPOLOGIES = {
        # San Francisco dead-end example
        "37.7850:-122.4050": TopologyData(
            topology_type=TopologyType.DEAD_END,
            road_name="Test Dead End Street",
            has_multiple_exits=False,
            exit_count=1,
        ),
        # San Francisco arterial
        "37.7749:-122.4194": TopologyData(
            topology_type=TopologyType.ARTERIAL,
            road_name="Market Street",
            has_multiple_exits=True,
            exit_count=6,
        ),
        # Chase Center area
        "37.7680:-122.3879": TopologyData(
            topology_type=TopologyType.ARTERIAL,
            road_name="Third Street",
            has_multiple_exits=True,
            exit_count=4,
        ),
    }
    
    @classmethod
    def get_topology(cls, location: GeoLocation) -> TopologyData:
        """
        Get topology data for a location.
        
        Args:
            location: Geographic location
            
        Returns:
            TopologyData (defaults to RESIDENTIAL if unknown)
        """
        # Round to 4 decimal places for lookup
        key = f"{round(location.latitude, 4)}:{round(location.longitude, 4)}"
        
        # Check exact match
        if key in cls.MOCK_TOPOLOGIES:
            return cls.MOCK_TOPOLOGIES[key]
        
        # Check nearby (within ~100m)
        for mock_key, topology in cls.MOCK_TOPOLOGIES.items():
            mock_lat, mock_lon = map(float, mock_key.split(":"))
            if (
                abs(location.latitude - mock_lat) < 0.001 and
                abs(location.longitude - mock_lon) < 0.001
            ):
                return topology
        
        # Default to residential
        return TopologyData(
            topology_type=TopologyType.RESIDENTIAL,
            has_multiple_exits=True,
            exit_count=2,
        )
    
    @classmethod
    def register_topology(
        cls,
        latitude: float,
        longitude: float,
        topology_type: TopologyType,
        exit_count: int = 2,
    ) -> None:
        """
        Register topology for a location (for testing).
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            topology_type: Street type
            exit_count: Number of exits
        """
        key = f"{round(latitude, 4)}:{round(longitude, 4)}"
        cls.MOCK_TOPOLOGIES[key] = TopologyData(
            topology_type=topology_type,
            has_multiple_exits=exit_count > 1,
            exit_count=exit_count,
        )


class FleetSentinel:
    """
    The FleetSentinel Decision Engine.
    
    This is the main class that orchestrates the entire defense system.
    It provides a single entry point for evaluating ride requests.
    
    Architecture:
        Request -> Sentinel Core -> Oracle -> Gate -> Decision
                      |              |         |
                   Clustering    Context   Liability
                   Analysis    Verification   Hold
    
    Example:
        sentinel = FleetSentinel(redis_url="redis://localhost:6379")
        await sentinel.initialize()
        
        decision = await sentinel.evaluate_ride_request(request)
        
        if decision.decision == Decision.DISPATCH:
            # Safe to dispatch vehicle
        elif decision.decision == Decision.REQUIRE_LIABILITY:
            # Require $500 deposit first
        elif decision.decision == Decision.BLOCK_DANGEROUS:
            # Block the request
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        detection_config: Optional[DetectionConfig] = None,
        liability_config: Optional[LiabilityConfig] = None,
    ):
        """
        Initialize FleetSentinel.
        
        Args:
            redis_url: Redis connection URL
            detection_config: Configuration for swarm detection
            liability_config: Configuration for liability gate
        """
        self._detection_config = detection_config or DetectionConfig()
        self._liability_config = liability_config or LiabilityConfig()
        
        # Initialize layers
        self._swarm_detector = SwarmDetector(
            redis_url=redis_url,
            config=self._detection_config,
        )
        self._context_service = ContextVerificationService()
        self._liability_gate = LiabilityEnforcementMiddleware(
            hold_amount_cents=self._liability_config.hold_amount_cents,
        )
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all subsystems."""
        await self._swarm_detector.initialize()
        self._initialized = True
        logger.info("FleetSentinel initialized")
    
    async def close(self) -> None:
        """Clean up resources."""
        await self._swarm_detector.close()
        self._initialized = False
    
    async def evaluate_ride_request(
        self,
        request: RideRequest,
        user_payment_method_id: Optional[str] = None,
    ) -> SentinelDecision:
        """
        Evaluate a ride request through the full defense pipeline.
        
        This is the MAIN ENTRY POINT for FleetSentinel.
        
        Pipeline:
            1. Layer 1 (Sentinel): Analyze request density and calculate confidence
            2. Layer 2 (Oracle): Verify context if density is high
            3. Make initial decision based on confidence + context
            4. Layer 3 (Gate): Apply liability hold if grey zone
            5. Return final decision with vehicle instructions
        
        Args:
            request: The ride request to evaluate
            user_payment_method_id: User's payment method (for liability holds)
            
        Returns:
            SentinelDecision with final verdict and supporting data
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        pipeline_stages: List[str] = []
        
        try:
            # =================================================================
            # STAGE 1: Get Topology Data
            # =================================================================
            topology_data = TopologyService.get_topology(request.pickup_location)
            pipeline_stages.append("topology_lookup")
            
            # =================================================================
            # STAGE 2: Layer 1 - Swarm Detection (Sentinel Core)
            # =================================================================
            # First pass without context - just density analysis
            initial_assessment = await self._swarm_detector.analyze_request(
                request=request,
                topology_data=topology_data,
                context_data=None,  # Context added in next stage
            )
            pipeline_stages.append("swarm_detection")
            
            # =================================================================
            # STAGE 3: Layer 2 - Context Verification (if needed)
            # Only query oracle if density is elevated
            # =================================================================
            context_data: Optional[ContextData] = None
            
            if initial_assessment.cluster_metrics and \
               initial_assessment.cluster_metrics.density_score > 50:
                # Query context oracle
                context_result = await self._context_service.verify_context(
                    latitude=request.pickup_location.latitude,
                    longitude=request.pickup_location.longitude,
                    density_score=initial_assessment.cluster_metrics.density_score,
                )
                context_data = context_result.to_context_data()
                pipeline_stages.append("context_verification")
                
                # Re-analyze with context
                threat_assessment = await self._swarm_detector.analyze_request(
                    request=request,
                    topology_data=topology_data,
                    context_data=context_data,
                )
            else:
                # Use initial assessment
                threat_assessment = initial_assessment
            
            # =================================================================
            # STAGE 4: Make Decision Based on Confidence Score
            # =================================================================
            confidence = threat_assessment.confidence_score
            threat_level = threat_assessment.threat_level
            
            # Decision logic:
            # - CRITICAL (>0.8): BLOCK
            # - HIGH (0.7-0.8): BLOCK with human review
            # - MEDIUM (0.4-0.7): REQUIRE_LIABILITY
            # - LOW (<0.4): DISPATCH
            
            if confidence >= self._detection_config.confidence_threshold_block:
                decision = Decision.BLOCK_DANGEROUS
                vehicle_alert = VehicleState.ALERT
            elif confidence >= self._detection_config.confidence_threshold_liability:
                decision = Decision.REQUIRE_LIABILITY
                vehicle_alert = VehicleState.ALERT
            else:
                decision = Decision.DISPATCH
                vehicle_alert = VehicleState.NORMAL
            
            pipeline_stages.append("decision_logic")
            
            # =================================================================
            # STAGE 5: Layer 3 - Liability Gate (if applicable)
            # =================================================================
            liability_required = False
            liability_amount = 0
            payment_intent_id = None
            
            if decision == Decision.REQUIRE_LIABILITY:
                if user_payment_method_id:
                    # Attempt to place hold
                    liability_result = await self._liability_gate.enforce(
                        request_id=request.request_id,
                        user_id=request.user_id,
                        payment_method_id=user_payment_method_id,
                        confidence_score=confidence,
                        threat_level=threat_level.value,
                    )
                    pipeline_stages.append("liability_enforcement")
                    
                    if liability_result.success and liability_result.hold:
                        # Hold placed successfully - allow with deposit
                        liability_required = True
                        liability_amount = liability_result.hold.amount_cents
                        payment_intent_id = liability_result.hold.payment_intent_id
                    else:
                        # Hold failed - user declined or card issue
                        # Treat as block (user self-selected out)
                        decision = Decision.BLOCK_DANGEROUS
                        threat_assessment.explanation += (
                            " Liability hold failed - request blocked."
                        )
                else:
                    # No payment method provided
                    liability_required = True
                    liability_amount = self._liability_config.hold_amount_cents
            
            # =================================================================
            # STAGE 6: Generate Special Instructions for Vehicle
            # =================================================================
            special_instructions: List[str] = []
            
            if threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
                special_instructions.append("ALERT: High-risk destination")
                special_instructions.append("Monitor surroundings continuously")
                special_instructions.append("Prepare defensive protocols")
            
            if topology_data.topology_type == TopologyType.DEAD_END:
                special_instructions.append("WARNING: Dead-end street")
                special_instructions.append("Verify exit route before entering")
            
            if liability_required:
                special_instructions.append(
                    f"NOTE: ${liability_amount/100:.2f} security deposit active"
                )
            
            # Calculate processing time
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            pipeline_stages.append("complete")
            
            # Build final decision
            sentinel_decision = SentinelDecision(
                request_id=request.request_id,
                timestamp=datetime.utcnow(),
                decision=decision,
                threat_assessment=threat_assessment,
                liability_required=liability_required,
                liability_amount_cents=liability_amount,
                payment_intent_id=payment_intent_id,
                vehicle_alert_level=vehicle_alert,
                special_instructions=special_instructions,
                processing_time_ms=processing_time_ms,
                pipeline_stages=pipeline_stages,
            )
            
            # Log decision
            logger.info(
                f"Request {request.request_id}: decision={decision.value}, "
                f"confidence={confidence:.3f}, threat_level={threat_level.value}, "
                f"processing_time_ms={processing_time_ms:.1f}"
            )
            
            return sentinel_decision
            
        except Exception as e:
            logger.error(f"Error evaluating request {request.request_id}: {e}")
            
            # Fail-safe: allow with alert
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            
            return SentinelDecision(
                request_id=request.request_id,
                decision=Decision.DEFER_TO_HUMAN,
                threat_assessment=ThreatAssessment(
                    request_id=request.request_id,
                    location=request.pickup_location,
                    confidence_score=0.5,
                    threat_level=ThreatLevel.MEDIUM,
                    explanation=f"Error in evaluation pipeline: {e}",
                ),
                vehicle_alert_level=VehicleState.ALERT,
                special_instructions=["CAUTION: Automated evaluation failed"],
                processing_time_ms=processing_time_ms,
                pipeline_stages=pipeline_stages + ["error"],
            )
    
    async def release_liability_hold(self, request_id: str) -> bool:
        """
        Release a liability hold after successful ride completion.
        
        Args:
            request_id: ID of the ride request
            
        Returns:
            True if hold was released
        """
        hold = self._liability_gate.get_hold_by_request(request_id)
        if hold:
            return await self._liability_gate.release(hold.hold_id)
        return False
    
    async def capture_liability_hold(
        self, request_id: str, amount_cents: Optional[int] = None
    ) -> bool:
        """
        Capture a liability hold due to vehicle damage.
        
        Args:
            request_id: ID of the ride request
            amount_cents: Amount to capture (None = full amount)
            
        Returns:
            True if hold was captured
        """
        hold = self._liability_gate.get_hold_by_request(request_id)
        if hold:
            return await self._liability_gate.capture(hold.hold_id, amount_cents)
        return False


async def evaluate_ride_request(
    request: RideRequest,
    sentinel: Optional[FleetSentinel] = None,
    redis_url: str = "redis://localhost:6379/0",
    user_payment_method_id: Optional[str] = None,
) -> SentinelDecision:
    """
    Convenience function to evaluate a ride request.
    
    This is the main entry point for external callers who don't want
    to manage the FleetSentinel lifecycle.
    
    Args:
        request: The ride request to evaluate
        sentinel: Optional pre-initialized FleetSentinel instance
        redis_url: Redis URL (used if sentinel not provided)
        user_payment_method_id: User's payment method for liability holds
        
    Returns:
        SentinelDecision with the final verdict
        
    Example:
        request = RideRequest(
            user_id="user_123",
            pickup_location=GeoLocation(latitude=37.7749, longitude=-122.4194),
        )
        
        decision = await evaluate_ride_request(request)
        
        if decision.decision == Decision.DISPATCH:
            print("Safe to dispatch!")
    """
    if sentinel is None:
        # Create and use temporary sentinel
        sentinel = FleetSentinel(redis_url=redis_url)
        try:
            await sentinel.initialize()
            return await sentinel.evaluate_ride_request(
                request, user_payment_method_id
            )
        finally:
            await sentinel.close()
    else:
        return await sentinel.evaluate_ride_request(
            request, user_payment_method_id
        )


# =============================================================================
# KAFKA STREAM PROCESSOR (SIMULATED)
# =============================================================================

class KafkaStreamProcessor:
    """
    Kafka-based stream processor for high-throughput request evaluation.
    
    This simulates a production Kafka consumer that:
        1. Consumes ride requests from a Kafka topic
        2. Evaluates each through FleetSentinel
        3. Publishes decisions to an output topic
    
    In production, this would use aiokafka or confluent-kafka.
    """
    
    def __init__(
        self,
        sentinel: FleetSentinel,
        input_topic: str = "ride-requests",
        output_topic: str = "sentinel-decisions",
        consumer_group: str = "fleetsentinel-consumer",
    ):
        """Initialize the stream processor."""
        self._sentinel = sentinel
        self._input_topic = input_topic
        self._output_topic = output_topic
        self._consumer_group = consumer_group
        self._running = False
        self._processed_count = 0
    
    async def start(self) -> None:
        """Start processing the stream."""
        self._running = True
        logger.info(
            f"Kafka processor started: {self._input_topic} -> {self._output_topic}"
        )
        
        # In production, this would be a Kafka consumer loop
        # For simulation, we just set the flag
    
    async def stop(self) -> None:
        """Stop processing."""
        self._running = False
        logger.info(f"Kafka processor stopped. Processed {self._processed_count} requests.")
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single request from the stream.
        
        Args:
            request_data: Raw request data from Kafka
            
        Returns:
            Decision data to publish to output topic
        """
        if not self._running:
            raise RuntimeError("Processor not running")
        
        # Parse request
        request = RideRequest(
            request_id=request_data.get("request_id", ""),
            user_id=request_data.get("user_id", ""),
            pickup_location=GeoLocation(
                latitude=request_data.get("pickup_lat", 0.0),
                longitude=request_data.get("pickup_lon", 0.0),
            ),
            account_age_days=request_data.get("account_age_days", 0),
            historical_rides=request_data.get("historical_rides", 0),
            payment_method_verified=request_data.get("payment_verified", False),
        )
        
        # Evaluate
        decision = await self._sentinel.evaluate_ride_request(request)
        self._processed_count += 1
        
        # Format for Kafka output
        return {
            "request_id": decision.request_id,
            "decision": decision.decision.value,
            "confidence_score": decision.threat_assessment.confidence_score,
            "threat_level": decision.threat_assessment.threat_level.value,
            "liability_required": decision.liability_required,
            "liability_amount_cents": decision.liability_amount_cents,
            "processing_time_ms": decision.processing_time_ms,
            "timestamp": decision.timestamp.isoformat(),
        }
