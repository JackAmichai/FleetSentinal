"""
FleetSentinel Core Data Models

This module defines all the core data structures used throughout the FleetSentinel system.
Using Pydantic for robust validation and type safety in a mission-critical environment.

Design Philosophy:
    - Immutability where possible (frozen dataclasses)
    - Strict typing for production safety
    - Rich enums for clear decision states
    - Comprehensive validation at the boundary
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum, auto
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator, ConfigDict


# =============================================================================
# ENUMS - Clear State Definitions
# =============================================================================

class Decision(str, Enum):
    """
    The final verdict for a ride request.
    
    DISPATCH: Safe to proceed, route vehicle normally.
    REQUIRE_LIABILITY: Grey zone - require $500 security deposit.
    BLOCK_DANGEROUS: Confirmed threat - reject request entirely.
    DEFER_TO_HUMAN: Edge case requiring human review (ops center).
    """
    DISPATCH = "DISPATCH"
    REQUIRE_LIABILITY = "REQUIRE_LIABILITY"
    BLOCK_DANGEROUS = "BLOCK_DANGEROUS"
    DEFER_TO_HUMAN = "DEFER_TO_HUMAN"


class ThreatLevel(str, Enum):
    """
    Categorical threat assessment based on confidence score.
    
    Mapping:
        NONE: 0.0 - 0.2 (Normal operation)
        LOW: 0.2 - 0.4 (Monitor, no action)
        MEDIUM: 0.4 - 0.7 (Liability gate triggered)
        HIGH: 0.7 - 0.9 (Enhanced scrutiny)
        CRITICAL: 0.9 - 1.0 (Block immediately)
    """
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TopologyType(str, Enum):
    """
    Street topology classification from OpenStreetMap data.
    
    Different topologies have different trap risks:
        - ARTERIAL: Major road, easy escape (low risk)
        - RESIDENTIAL: Standard street (medium risk)
        - COLLECTOR: Medium traffic road (medium risk)
        - LOCAL: Small local street (higher risk)
        - DEAD_END: Cul-de-sac or dead end (highest risk)
        - ALLEY: Narrow passage (highest risk)
    """
    ARTERIAL = "ARTERIAL"
    RESIDENTIAL = "RESIDENTIAL"
    COLLECTOR = "COLLECTOR"
    LOCAL = "LOCAL"
    DEAD_END = "DEAD_END"
    ALLEY = "ALLEY"


class ContextType(str, Enum):
    """
    Context classification from Oracle services.
    
    SCHEDULED_EVENT: Known event (concert, game, conference)
    CIVIL_UNREST: Protest, riot, or civil disturbance
    EMERGENCY: Fire, accident, police activity
    WEATHER: Severe weather causing clustering
    CONSTRUCTION: Road work causing detours
    NONE: No external context found
    """
    SCHEDULED_EVENT = "SCHEDULED_EVENT"
    CIVIL_UNREST = "CIVIL_UNREST"
    EMERGENCY = "EMERGENCY"
    WEATHER = "WEATHER"
    CONSTRUCTION = "CONSTRUCTION"
    NONE = "NONE"


class VehicleState(str, Enum):
    """
    Vehicle operational state for edge defense logic.
    """
    NORMAL = "NORMAL"
    ALERT = "ALERT"
    LOCKDOWN = "LOCKDOWN"
    TURTLE_MODE = "TURTLE_MODE"
    EMERGENCY_ESCAPE = "EMERGENCY_ESCAPE"


class PaymentStatus(str, Enum):
    """
    Payment/liability hold status.
    """
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    AUTHORIZED = "AUTHORIZED"
    CAPTURED = "CAPTURED"
    FAILED = "FAILED"
    RELEASED = "RELEASED"


# =============================================================================
# CORE DATA MODELS
# =============================================================================

class GeoLocation(BaseModel):
    """
    Geographic coordinates with optional metadata.
    
    Attributes:
        latitude: WGS84 latitude (-90 to 90)
        longitude: WGS84 longitude (-180 to 180)
        accuracy_meters: GPS accuracy if available
        altitude_meters: Altitude if available
    """
    model_config = ConfigDict(frozen=True)
    
    latitude: float = Field(..., ge=-90, le=90, description="WGS84 latitude")
    longitude: float = Field(..., ge=-180, le=180, description="WGS84 longitude")
    accuracy_meters: Optional[float] = Field(None, ge=0)
    altitude_meters: Optional[float] = None
    
    @property
    def as_tuple(self) -> tuple[float, float]:
        """Return (longitude, latitude) for Redis GEOADD."""
        return (self.longitude, self.latitude)
    
    def __str__(self) -> str:
        return f"({self.latitude:.6f}, {self.longitude:.6f})"


class RideRequest(BaseModel):
    """
    Incoming ride request to be evaluated by FleetSentinel.
    
    This is the primary input to the detection pipeline. Every field
    is captured for forensic analysis if an attack is confirmed.
    
    Attributes:
        request_id: Unique identifier for tracing
        user_id: Requesting user's account ID
        timestamp: UTC timestamp of request
        pickup_location: Where the user wants to be picked up
        dropoff_location: Destination (optional for initial request)
        user_ip_hash: Hashed IP for correlation (privacy-preserving)
        device_fingerprint: Device identifier hash
        account_age_days: How old the user's account is
        historical_rides: User's ride history count
        payment_method_verified: Whether payment is verified
    """
    model_config = ConfigDict(frozen=True)
    
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    pickup_location: GeoLocation
    dropoff_location: Optional[GeoLocation] = None
    
    # User Risk Signals
    user_ip_hash: Optional[str] = None
    device_fingerprint: Optional[str] = None
    account_age_days: int = Field(default=0, ge=0)
    historical_rides: int = Field(default=0, ge=0)
    payment_method_verified: bool = False
    
    # Request metadata
    source_app: str = "mobile"
    request_metadata: Dict[str, Any] = Field(default_factory=dict)


class TopologyData(BaseModel):
    """
    Street topology information for a location.
    
    Sourced from OpenStreetMap or proprietary map data.
    Used to weight the trap risk of a location.
    """
    model_config = ConfigDict(frozen=True)
    
    topology_type: TopologyType
    road_name: Optional[str] = None
    road_width_meters: Optional[float] = None
    has_multiple_exits: bool = True
    exit_count: int = Field(default=2, ge=0)
    speed_limit_mph: Optional[float] = None
    
    @property
    def weight(self) -> float:
        """
        Calculate topology weight for confidence scoring.
        
        Dead-ends and alleys get 2.5x weight (high trap risk).
        Arterial roads get 0.2x weight (easy escape).
        """
        weights = {
            TopologyType.ARTERIAL: 0.2,
            TopologyType.COLLECTOR: 0.5,
            TopologyType.RESIDENTIAL: 1.0,
            TopologyType.LOCAL: 1.5,
            TopologyType.DEAD_END: 2.5,
            TopologyType.ALLEY: 2.5,
        }
        base_weight = weights.get(self.topology_type, 1.0)
        
        # Additional penalty for single-exit locations
        if not self.has_multiple_exits or self.exit_count <= 1:
            base_weight *= 1.5
            
        return base_weight


class ContextData(BaseModel):
    """
    External context information from Oracle services.
    
    This data helps distinguish legitimate crowds from attacks.
    A Taylor Swift concert should not trigger a block.
    """
    model_config = ConfigDict(frozen=True)
    
    context_type: ContextType
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    event_name: Optional[str] = None
    event_venue: Optional[str] = None
    expected_attendance: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    source: str = "unknown"
    raw_data: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def modifier(self) -> float:
        """
        Calculate context modifier for confidence scoring.
        
        Positive modifiers increase threat (civil unrest).
        Negative modifiers decrease threat (known events).
        """
        modifiers = {
            ContextType.SCHEDULED_EVENT: -5.0,
            ContextType.CIVIL_UNREST: 5.0,
            ContextType.EMERGENCY: 2.0,
            ContextType.WEATHER: -1.0,
            ContextType.CONSTRUCTION: -0.5,
            ContextType.NONE: 0.0,
        }
        return modifiers.get(self.context_type, 0.0) * self.confidence


class ClusterMetrics(BaseModel):
    """
    Metrics for a detected request cluster.
    
    These metrics feed into the confidence calculation.
    """
    location: GeoLocation
    request_count: int = Field(default=0, ge=0)
    unique_users: int = Field(default=0, ge=0)
    unique_ips: int = Field(default=0, ge=0)
    unique_devices: int = Field(default=0, ge=0)
    time_window_seconds: int = Field(default=300, ge=1)
    arrival_rate_per_second: float = Field(default=0.0, ge=0.0)
    density_score: float = Field(default=0.0, ge=0.0)
    velocity_score: float = Field(default=0.0, ge=0.0)
    
    # Risk indicators
    new_accounts_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    unverified_payment_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class ThreatAssessment(BaseModel):
    """
    Complete threat assessment for a ride request or location.
    
    This is the output of the Sentinel Core analysis.
    """
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    location: GeoLocation
    
    # Confidence Score (0.0 - 1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    threat_level: ThreatLevel
    
    # Component scores (for explainability)
    density_component: float = Field(default=0.0)
    velocity_component: float = Field(default=0.0)
    topology_component: float = Field(default=0.0)
    context_component: float = Field(default=0.0)
    user_risk_component: float = Field(default=0.0)
    
    # Supporting data
    cluster_metrics: Optional[ClusterMetrics] = None
    topology_data: Optional[TopologyData] = None
    context_data: Optional[ContextData] = None
    
    # Explanation for operators
    explanation: str = ""
    recommendations: List[str] = Field(default_factory=list)


class SentinelDecision(BaseModel):
    """
    Final decision from the FleetSentinel system.
    
    This is what gets returned to the dispatch engine.
    """
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # The verdict
    decision: Decision
    threat_assessment: ThreatAssessment
    
    # Liability gate info (if applicable)
    liability_required: bool = False
    liability_amount_cents: int = 0
    payment_intent_id: Optional[str] = None
    
    # Vehicle instructions (if dispatched)
    vehicle_alert_level: VehicleState = VehicleState.NORMAL
    special_instructions: List[str] = Field(default_factory=list)
    
    # Audit trail
    processing_time_ms: float = 0.0
    pipeline_stages: List[str] = Field(default_factory=list)


# =============================================================================
# VEHICLE EDGE MODELS
# =============================================================================

class VehicleStatus(BaseModel):
    """
    Current status of a vehicle in the fleet.
    """
    vehicle_id: str
    location: GeoLocation
    state: VehicleState
    heading_degrees: float = Field(default=0.0, ge=0.0, le=360.0)
    speed_mph: float = Field(default=0.0, ge=0.0)
    
    # Sensor data
    proximity_objects: int = Field(default=0, ge=0)
    blocked_exits: int = Field(default=0, ge=0)
    
    # Defense state
    doors_locked: bool = False
    windows_closed: bool = True
    audio_warning_active: bool = False
    turtle_mode_active: bool = False


class DefenseCommand(BaseModel):
    """
    Command to a vehicle's self-defense unit.
    """
    vehicle_id: str
    command_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    target_state: VehicleState
    engage_turtle_mode: bool = False
    turtle_speed_mph: float = Field(default=0.5, ge=0.0, le=2.0)
    engage_lockdown: bool = False
    play_audio_warning: bool = False
    audio_message: str = ""
    
    # Path planning overrides
    override_pedestrian_gap: bool = False
    minimum_gap_meters: float = Field(default=0.5, ge=0.0)
    
    # Escape routing
    calculate_escape_route: bool = False
    preferred_direction_degrees: Optional[float] = None


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class DetectionConfig(BaseModel):
    """
    Configuration for the swarm detection engine.
    """
    # Geospatial
    detection_radius_meters: float = 500.0
    time_window_seconds: int = 300
    
    # Thresholds
    min_requests_for_alert: int = 10
    confidence_threshold_block: float = 0.8
    confidence_threshold_liability: float = 0.4
    
    # Weights
    density_weight: float = 0.4
    velocity_weight: float = 0.3
    topology_weight: float = 0.2
    user_risk_weight: float = 0.1
    
    # Scaling factor for sigmoid
    alpha_scaling: float = 0.5


class LiabilityConfig(BaseModel):
    """
    Configuration for the liability gate.
    """
    hold_amount_cents: int = 50000  # $500.00
    hold_description: str = "FleetSentinel Security Deposit"
    hold_duration_hours: int = 24
    auto_release_on_completion: bool = True


class VehicleDefenseConfig(BaseModel):
    """
    Configuration for vehicle self-defense.
    """
    turtle_speed_mph: float = 0.5
    minimum_gap_meters: float = 0.5
    audio_warning_duration_seconds: int = 30
    lockdown_threshold: int = 5  # Number of blocked exits
    auto_engage_turtle: bool = True
