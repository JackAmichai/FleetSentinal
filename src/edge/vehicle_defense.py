"""
FleetSentinel Vehicle Self-Defense Unit

Layer 4: The Escape Protocol - Physical Safety for Trapped Assets

This module provides on-vehicle logic for self-defense when a vehicle
is physically surrounded or trapped. It interfaces with the vehicle's
control systems to override normal "polite" driving behavior.

Features:
    1. Turtle Mode: Override standard gap-keeping to creep through crowds
    2. Vandalism Guard: Lock doors, close windows, trigger audio warnings
    3. Escape Path Planning: Calculate optimal exit routes

Theory of Operation:
    Normal AV behavior is optimized for politeness and passenger comfort.
    When trapped, we must prioritize:
        1. Asset safety (prevent vehicle damage)
        2. Passenger safety (if occupied)
        3. Escape (get out of danger zone)
    
    This requires overriding normal constraints:
        - Pedestrian gap distance: 2m -> 0.5m
        - Max speed in crowd: 5mph -> 0.5mph (slower but continuous)
        - Audio warnings: Silent -> Active deterrence

Safety Notes:
    - Turtle mode is NEVER aggressive, just persistent
    - Audio warnings are legally compliant (no siren, just announcements)
    - All actions are logged for liability protection
"""

from __future__ import annotations

import asyncio
import math
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum, auto
from dataclasses import dataclass, field
import logging

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from ..models import (
    GeoLocation,
    VehicleState,
    VehicleStatus,
    DefenseCommand,
)


logger = logging.getLogger(__name__)


class ThreatProximity(str, Enum):
    """
    Classification of physical threats around vehicle.
    """
    NONE = "NONE"           # No immediate threat
    DISTANT = "DISTANT"     # Threat detected but far (>10m)
    NEARBY = "NEARBY"       # Threat closing in (2-10m)
    IMMINENT = "IMMINENT"   # Threat adjacent (<2m)
    CONTACT = "CONTACT"     # Physical contact detected


class AudioWarningType(str, Enum):
    """
    Types of audio warnings the vehicle can issue.
    """
    POLITE_DISPERSAL = "POLITE_DISPERSAL"
    FIRM_WARNING = "FIRM_WARNING"
    LEGAL_NOTICE = "LEGAL_NOTICE"
    EMERGENCY_HORN = "EMERGENCY_HORN"


# Pre-recorded audio messages (in production, these would be actual audio files)
AUDIO_MESSAGES = {
    AudioWarningType.POLITE_DISPERSAL: (
        "Attention: This is an autonomous vehicle attempting to pass. "
        "Please clear a path. Thank you for your cooperation."
    ),
    AudioWarningType.FIRM_WARNING: (
        "Warning: This vehicle is attempting to move. "
        "Please step away from the vehicle immediately. "
        "This area is being recorded."
    ),
    AudioWarningType.LEGAL_NOTICE: (
        "Legal Notice: Intentional obstruction of autonomous vehicles "
        "may constitute a criminal offense. This incident is being recorded "
        "and may be reported to law enforcement."
    ),
    AudioWarningType.EMERGENCY_HORN: (
        "EMERGENCY: Vehicle in distress. Please clear the area immediately."
    ),
}


@dataclass
class SurroundingState:
    """
    Current state of the area surrounding the vehicle.
    
    This is computed from sensor data (simulated in this implementation).
    """
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Object counts by direction (0-360 degrees in 45-degree sectors)
    sector_counts: Dict[str, int] = field(default_factory=dict)
    
    # Closest object distances by direction
    closest_distances: Dict[str, float] = field(default_factory=dict)
    
    # Exit analysis
    blocked_exits: int = 0
    available_exits: int = 4
    best_exit_direction: Optional[float] = None
    
    # Threat assessment
    threat_proximity: ThreatProximity = ThreatProximity.NONE
    is_surrounded: bool = False
    
    # Movement feasibility
    can_move_forward: bool = True
    can_move_backward: bool = True
    minimum_clearance_meters: float = 10.0
    
    def get_safest_direction(self) -> Tuple[float, float]:
        """
        Get the safest direction to move based on sensor data.
        
        Returns:
            Tuple of (direction_degrees, clearance_meters)
        """
        if not self.closest_distances:
            return (0.0, 10.0)
        
        # Find direction with maximum clearance
        max_distance = 0.0
        safest_direction = 0.0
        
        for direction_str, distance in self.closest_distances.items():
            if distance > max_distance:
                max_distance = distance
                safest_direction = float(direction_str)
        
        return (safest_direction, max_distance)


class TurtleMode:
    """
    Turtle Mode - Slow but persistent movement through crowds.
    
    When a vehicle is surrounded, normal driving algorithms will
    freeze (too many pedestrians too close). Turtle mode overrides
    this by:
        1. Reducing minimum gap distance (2m -> 0.5m)
        2. Moving at constant slow speed (0.5 mph)
        3. Continuously broadcasting audio warnings
        4. Prioritizing forward progress over comfort
    
    Why "Turtle"?
        - Like a turtle, we move slowly but steadily
        - The hard shell (locked doors) protects the inside
        - We don't stop, just slow down
    """
    
    DEFAULT_SPEED_MPH = 0.5
    DEFAULT_MIN_GAP_METERS = 0.5
    
    def __init__(
        self,
        speed_mph: float = DEFAULT_SPEED_MPH,
        min_gap_meters: float = DEFAULT_MIN_GAP_METERS,
    ):
        """
        Initialize Turtle Mode.
        
        Args:
            speed_mph: Target speed in turtle mode
            min_gap_meters: Minimum gap to maintain
        """
        self.speed_mph = speed_mph
        self.min_gap_meters = min_gap_meters
        self.is_active = False
        self.activated_at: Optional[datetime] = None
        self.total_distance_meters = 0.0
        self._target_heading: Optional[float] = None
    
    def activate(self, target_heading: Optional[float] = None) -> Dict[str, Any]:
        """
        Activate turtle mode.
        
        Args:
            target_heading: Optional target heading (degrees)
            
        Returns:
            Dict with path planning override parameters
        """
        self.is_active = True
        self.activated_at = datetime.utcnow()
        self._target_heading = target_heading
        
        logger.warning(
            f"TURTLE MODE ACTIVATED: speed={self.speed_mph}mph, "
            f"min_gap={self.min_gap_meters}m, heading={target_heading}"
        )
        
        # Return path planning overrides
        return {
            "override_active": True,
            "max_speed_mph": self.speed_mph,
            "min_pedestrian_gap_meters": self.min_gap_meters,
            "min_vehicle_gap_meters": self.min_gap_meters,
            "ignore_crosswalk_wait": True,  # Don't wait for pedestrian signals
            "continuous_motion": True,       # Keep creeping even if obstacles present
            "target_heading": target_heading,
            "priority": "escape",
        }
    
    def deactivate(self) -> Dict[str, Any]:
        """
        Deactivate turtle mode and return to normal operation.
        
        Returns:
            Summary of turtle mode session
        """
        if not self.is_active:
            return {"was_active": False}
        
        duration = (datetime.utcnow() - self.activated_at).total_seconds() if self.activated_at else 0
        
        self.is_active = False
        
        logger.info(
            f"TURTLE MODE DEACTIVATED: duration={duration:.1f}s, "
            f"distance={self.total_distance_meters:.1f}m"
        )
        
        return {
            "was_active": True,
            "duration_seconds": duration,
            "distance_covered_meters": self.total_distance_meters,
            "deactivated_at": datetime.utcnow().isoformat(),
        }
    
    def get_control_commands(
        self, current_state: SurroundingState
    ) -> Dict[str, Any]:
        """
        Get control commands for turtle mode movement.
        
        This would interface with the vehicle's control system.
        
        Args:
            current_state: Current surrounding state
            
        Returns:
            Control commands for the vehicle
        """
        if not self.is_active:
            return {"active": False}
        
        # Determine movement direction
        safest_dir, clearance = current_state.get_safest_direction()
        
        # Use target heading if set, otherwise use safest direction
        target_heading = self._target_heading if self._target_heading is not None else safest_dir
        
        # Calculate speed based on clearance
        # Slower when closer to obstacles
        if clearance < 1.0:
            speed_factor = clearance / 1.0  # Reduce speed proportionally
        else:
            speed_factor = 1.0
        
        actual_speed = self.speed_mph * speed_factor
        
        return {
            "active": True,
            "target_speed_mph": actual_speed,
            "target_heading_degrees": target_heading,
            "steering_override": True,
            "min_gap_override": self.min_gap_meters,
            "clearance_ahead": clearance,
        }


class VandalismGuard:
    """
    Vandalism Guard - Protect vehicle from physical damage.
    
    When threat is detected, this system:
        1. Locks all doors
        2. Closes all windows
        3. Activates audio warnings
        4. Triggers external recording
        5. Dims interior lights (reduce visibility)
    
    The goal is to make the vehicle a less attractive target
    while documenting any attempted damage.
    """
    
    def __init__(self):
        """Initialize Vandalism Guard."""
        self.is_active = False
        self.activated_at: Optional[datetime] = None
        
        # State tracking
        self.doors_locked = False
        self.windows_closed = True
        self.audio_active = False
        self.recording_active = False
        
        # Escalation level (0-3)
        self.escalation_level = 0
    
    def activate(self, escalation_level: int = 1) -> Dict[str, Any]:
        """
        Activate vandalism guard.
        
        Args:
            escalation_level: 0=passive, 1=defensive, 2=active, 3=emergency
            
        Returns:
            Dict with protection commands
        """
        self.is_active = True
        self.activated_at = datetime.utcnow()
        self.escalation_level = min(max(escalation_level, 0), 3)
        
        # Always do these
        self.doors_locked = True
        self.windows_closed = True
        self.recording_active = True
        
        # Escalation-dependent actions
        if escalation_level >= 1:
            self.audio_active = True
            audio_type = AudioWarningType.POLITE_DISPERSAL
        if escalation_level >= 2:
            audio_type = AudioWarningType.FIRM_WARNING
        if escalation_level >= 3:
            audio_type = AudioWarningType.LEGAL_NOTICE
        
        logger.warning(
            f"VANDALISM GUARD ACTIVATED: level={escalation_level}, "
            f"doors_locked={self.doors_locked}, audio={self.audio_active}"
        )
        
        return {
            "active": True,
            "escalation_level": self.escalation_level,
            "commands": {
                "lock_doors": True,
                "close_windows": True,
                "start_recording": True,
                "dim_interior": escalation_level >= 2,
                "audio_warning": {
                    "enabled": self.audio_active,
                    "type": audio_type.value if self.audio_active else None,
                    "message": AUDIO_MESSAGES.get(audio_type) if self.audio_active else None,
                    "repeat_interval_seconds": 15,
                },
            },
        }
    
    def escalate(self) -> Dict[str, Any]:
        """
        Escalate to next protection level.
        
        Returns:
            New protection commands
        """
        if self.escalation_level < 3:
            return self.activate(self.escalation_level + 1)
        return {"already_at_max": True}
    
    def deactivate(self) -> Dict[str, Any]:
        """
        Deactivate vandalism guard.
        
        Returns:
            Summary of session
        """
        if not self.is_active:
            return {"was_active": False}
        
        duration = (datetime.utcnow() - self.activated_at).total_seconds() if self.activated_at else 0
        
        self.is_active = False
        self.audio_active = False
        self.escalation_level = 0
        # Note: recording continues for evidence preservation
        
        logger.info(f"VANDALISM GUARD DEACTIVATED: duration={duration:.1f}s")
        
        return {
            "was_active": True,
            "duration_seconds": duration,
            "max_escalation_reached": self.escalation_level,
            "recording_preserved": True,
        }
    
    def get_audio_message(self) -> Optional[str]:
        """Get current audio warning message."""
        if not self.audio_active:
            return None
        
        audio_types = [
            AudioWarningType.POLITE_DISPERSAL,
            AudioWarningType.POLITE_DISPERSAL,
            AudioWarningType.FIRM_WARNING,
            AudioWarningType.LEGAL_NOTICE,
        ]
        
        audio_type = audio_types[min(self.escalation_level, 3)]
        return AUDIO_MESSAGES[audio_type]


class EscapePathPlanner:
    """
    Escape Path Planner - Calculate optimal exit routes.
    
    When surrounded, this planner:
        1. Analyzes surrounding sensor data
        2. Identifies potential exit routes
        3. Scores routes by safety and feasibility
        4. Returns optimal escape path
    
    Uses simple heuristics for the mock implementation.
    In production, this would use full path planning with
    real sensor data and HD maps.
    """
    
    def __init__(self):
        """Initialize escape planner."""
        self._current_plan: Optional[Dict[str, Any]] = None
    
    def analyze_surroundings(
        self, surrounding_state: SurroundingState
    ) -> Dict[str, Any]:
        """
        Analyze surroundings and identify escape options.
        
        Args:
            surrounding_state: Current sensor state
            
        Returns:
            Analysis results with escape options
        """
        # Count blocked vs open sectors
        blocked_sectors = []
        open_sectors = []
        
        for direction, count in surrounding_state.sector_counts.items():
            if count > 3:  # More than 3 objects = blocked
                blocked_sectors.append(float(direction))
            else:
                open_sectors.append(float(direction))
        
        # Find largest contiguous open arc
        best_escape_direction = None
        best_clearance = 0.0
        
        for direction in open_sectors:
            clearance = surrounding_state.closest_distances.get(str(direction), 0)
            if clearance > best_clearance:
                best_clearance = clearance
                best_escape_direction = direction
        
        return {
            "blocked_sectors": blocked_sectors,
            "open_sectors": open_sectors,
            "best_escape_direction": best_escape_direction,
            "best_clearance_meters": best_clearance,
            "escape_feasibility": "high" if best_clearance > 2.0 else (
                "medium" if best_clearance > 0.5 else "low"
            ),
            "is_surrounded": len(blocked_sectors) >= 6,
        }
    
    def plan_escape(
        self,
        current_location: GeoLocation,
        surrounding_state: SurroundingState,
        target_location: Optional[GeoLocation] = None,
    ) -> Dict[str, Any]:
        """
        Plan an escape route from current position.
        
        Args:
            current_location: Vehicle's current position
            surrounding_state: Current sensor state
            target_location: Optional preferred destination
            
        Returns:
            Escape plan with waypoints and instructions
        """
        analysis = self.analyze_surroundings(surrounding_state)
        
        escape_direction = analysis["best_escape_direction"]
        
        if escape_direction is None:
            # Completely surrounded - no clear path
            return {
                "success": False,
                "reason": "No escape route available",
                "recommendation": "Engage turtle mode and creep forward",
                "turtle_mode_recommended": True,
                "alert_dispatch": True,
            }
        
        # Calculate escape waypoint (50m in escape direction)
        escape_distance_m = 50
        
        # Simple lat/lon offset calculation
        # 1 degree lat ≈ 111km, 1 degree lon varies by latitude
        lat_offset = (escape_distance_m / 111000) * math.cos(math.radians(escape_direction))
        lon_offset = (escape_distance_m / (111000 * math.cos(math.radians(current_location.latitude)))) * math.sin(math.radians(escape_direction))
        
        waypoint = GeoLocation(
            latitude=current_location.latitude + lat_offset,
            longitude=current_location.longitude + lon_offset,
        )
        
        self._current_plan = {
            "success": True,
            "escape_direction_degrees": escape_direction,
            "clearance_meters": analysis["best_clearance_meters"],
            "feasibility": analysis["escape_feasibility"],
            "waypoint": {
                "latitude": waypoint.latitude,
                "longitude": waypoint.longitude,
            },
            "instructions": [
                f"Turn to heading {escape_direction:.0f}°",
                "Engage turtle mode if needed",
                "Proceed to escape waypoint",
                "Resume normal routing once clear",
            ],
            "estimated_time_seconds": escape_distance_m / (0.5 * 0.447),  # 0.5 mph in m/s
        }
        
        return self._current_plan


class ThreatDetectionModel:
    """
    PyTorch-based threat detection model for on-vehicle inference.
    
    This model processes sensor data to detect:
        - Crowd density
        - Aggressive behavior patterns
        - Potential blockades
    
    Note: This is a simplified mock for demonstration.
    Production would use a trained model with real sensor inputs.
    """
    
    def __init__(self):
        """Initialize threat detection model."""
        self.model = None
        
        if TORCH_AVAILABLE:
            self._build_model()
        else:
            logger.warning("PyTorch not available - threat detection using fallback")
    
    def _build_model(self):
        """Build the neural network model."""
        if not TORCH_AVAILABLE:
            return
        
        # Simple feedforward network for threat scoring
        self.model = nn.Sequential(
            nn.Linear(16, 32),   # Input: 16 sensor features
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),    # Output: [no_threat, low, medium, high]
            nn.Softmax(dim=1),
        )
        
        # Initialize with random weights (in production, load trained weights)
        for param in self.model.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
    
    def predict(self, sensor_features: List[float]) -> Dict[str, float]:
        """
        Predict threat level from sensor features.
        
        Args:
            sensor_features: List of 16 normalized sensor values
            
        Returns:
            Dict with threat probabilities
        """
        if not TORCH_AVAILABLE or self.model is None:
            # Fallback: simple heuristic
            avg_density = sum(sensor_features[:8]) / 8 if len(sensor_features) >= 8 else 0
            return {
                "no_threat": max(0, 1 - avg_density),
                "low": avg_density * 0.3,
                "medium": avg_density * 0.4,
                "high": avg_density * 0.3,
            }
        
        # Pad or truncate to 16 features
        features = sensor_features[:16] + [0.0] * max(0, 16 - len(sensor_features))
        
        # Convert to tensor
        input_tensor = torch.tensor([features], dtype=torch.float32)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        probs = output[0].tolist()
        return {
            "no_threat": probs[0],
            "low": probs[1],
            "medium": probs[2],
            "high": probs[3],
        }


class VehicleSelfDefenseUnit:
    """
    The Vehicle Self-Defense Unit - Central coordinator for escape protocols.
    
    This class coordinates all vehicle defense systems:
        - Turtle Mode (movement)
        - Vandalism Guard (protection)
        - Escape Path Planner (navigation)
        - Threat Detection (analysis)
    
    Usage:
        defense = VehicleSelfDefenseUnit(vehicle_id="v123")
        
        # When threat detected
        status = await defense.assess_situation(surrounding_state)
        
        if status["threat_level"] == "high":
            # Engage full defense
            await defense.engage_full_defense(surrounding_state)
        
        # When clear
        await defense.disengage()
    """
    
    def __init__(
        self,
        vehicle_id: str,
        turtle_speed_mph: float = 0.5,
        min_gap_meters: float = 0.5,
    ):
        """
        Initialize the Vehicle Self-Defense Unit.
        
        Args:
            vehicle_id: Unique vehicle identifier
            turtle_speed_mph: Speed in turtle mode
            min_gap_meters: Minimum gap in turtle mode
        """
        self.vehicle_id = vehicle_id
        
        # Initialize subsystems
        self.turtle_mode = TurtleMode(turtle_speed_mph, min_gap_meters)
        self.vandalism_guard = VandalismGuard()
        self.escape_planner = EscapePathPlanner()
        self.threat_model = ThreatDetectionModel()
        
        # State
        self.current_state = VehicleState.NORMAL
        self.defense_engaged_at: Optional[datetime] = None
        self.incident_id: Optional[str] = None
        
        # Telemetry
        self.telemetry_log: List[Dict[str, Any]] = []
    
    def _log_telemetry(self, event: str, data: Dict[str, Any]) -> None:
        """Log telemetry event for incident analysis."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "vehicle_id": self.vehicle_id,
            "incident_id": self.incident_id,
            "event": event,
            "state": self.current_state.value,
            "data": data,
        }
        self.telemetry_log.append(entry)
        logger.info(f"Vehicle {self.vehicle_id} telemetry: {event}")
    
    async def assess_situation(
        self, surrounding_state: SurroundingState
    ) -> Dict[str, Any]:
        """
        Assess the current threat situation.
        
        Args:
            surrounding_state: Current sensor state
            
        Returns:
            Threat assessment with recommended actions
        """
        # Extract features for ML model
        sensor_features = []
        
        # Sector densities (8 features)
        for i in range(0, 360, 45):
            count = surrounding_state.sector_counts.get(str(i), 0)
            sensor_features.append(min(count / 10, 1.0))  # Normalize to 0-1
        
        # Sector distances (8 features)
        for i in range(0, 360, 45):
            distance = surrounding_state.closest_distances.get(str(i), 10.0)
            sensor_features.append(max(0, 1 - distance / 10))  # Inverse distance
        
        # Get threat prediction
        threat_probs = self.threat_model.predict(sensor_features)
        
        # Determine threat level
        max_threat = max(threat_probs.items(), key=lambda x: x[1])
        threat_level = max_threat[0]
        confidence = max_threat[1]
        
        # Map to VehicleState
        state_map = {
            "no_threat": VehicleState.NORMAL,
            "low": VehicleState.ALERT,
            "medium": VehicleState.LOCKDOWN,
            "high": VehicleState.TURTLE_MODE,
        }
        recommended_state = state_map.get(threat_level, VehicleState.ALERT)
        
        # Get escape analysis
        escape_analysis = self.escape_planner.analyze_surroundings(surrounding_state)
        
        assessment = {
            "threat_level": threat_level,
            "threat_confidence": confidence,
            "threat_probabilities": threat_probs,
            "is_surrounded": surrounding_state.is_surrounded,
            "blocked_exits": surrounding_state.blocked_exits,
            "recommended_state": recommended_state.value,
            "escape_feasibility": escape_analysis["escape_feasibility"],
            "recommended_actions": [],
        }
        
        # Add recommendations
        if threat_level == "high":
            assessment["recommended_actions"] = [
                "Engage turtle mode",
                "Activate vandalism guard",
                "Alert dispatch center",
                "Begin escape protocol",
            ]
        elif threat_level == "medium":
            assessment["recommended_actions"] = [
                "Lock doors and windows",
                "Prepare turtle mode",
                "Monitor situation",
            ]
        elif threat_level == "low":
            assessment["recommended_actions"] = [
                "Heighten awareness",
                "Prepare defensive systems",
            ]
        
        return assessment
    
    async def engage_turtle_mode(
        self,
        surrounding_state: SurroundingState,
        target_heading: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Engage turtle mode for escape through crowds.
        
        Args:
            surrounding_state: Current sensor state
            target_heading: Preferred escape direction (auto-calculated if None)
            
        Returns:
            Turtle mode activation result
        """
        if self.turtle_mode.is_active:
            return {"already_active": True}
        
        # Create incident ID if new engagement
        if self.incident_id is None:
            self.incident_id = f"incident_{uuid.uuid4().hex[:12]}"
        
        # Calculate escape heading if not provided
        if target_heading is None:
            escape_plan = self.escape_planner.plan_escape(
                GeoLocation(latitude=0, longitude=0),  # Would use actual location
                surrounding_state,
            )
            if escape_plan["success"]:
                target_heading = escape_plan["escape_direction_degrees"]
        
        # Activate turtle mode
        result = self.turtle_mode.activate(target_heading)
        
        # Update state
        self.current_state = VehicleState.TURTLE_MODE
        self.defense_engaged_at = datetime.utcnow()
        
        self._log_telemetry("turtle_mode_engaged", {
            "target_heading": target_heading,
            "speed_mph": self.turtle_mode.speed_mph,
            "min_gap_meters": self.turtle_mode.min_gap_meters,
        })
        
        return {
            "success": True,
            "incident_id": self.incident_id,
            "turtle_mode": result,
            "state": self.current_state.value,
        }
    
    async def engage_vandalism_guard(
        self, escalation_level: int = 1
    ) -> Dict[str, Any]:
        """
        Engage vandalism guard protection.
        
        Args:
            escalation_level: Protection level (0-3)
            
        Returns:
            Guard activation result
        """
        # Create incident ID if new engagement
        if self.incident_id is None:
            self.incident_id = f"incident_{uuid.uuid4().hex[:12]}"
        
        # Activate guard
        result = self.vandalism_guard.activate(escalation_level)
        
        # Update state if not already in higher state
        if self.current_state in (VehicleState.NORMAL, VehicleState.ALERT):
            self.current_state = VehicleState.LOCKDOWN
            self.defense_engaged_at = datetime.utcnow()
        
        self._log_telemetry("vandalism_guard_engaged", {
            "escalation_level": escalation_level,
            "doors_locked": self.vandalism_guard.doors_locked,
            "audio_active": self.vandalism_guard.audio_active,
        })
        
        return {
            "success": True,
            "incident_id": self.incident_id,
            "vandalism_guard": result,
            "state": self.current_state.value,
        }
    
    async def engage_full_defense(
        self,
        surrounding_state: SurroundingState,
        escalation_level: int = 2,
    ) -> Dict[str, Any]:
        """
        Engage full defense protocols.
        
        This activates all defensive systems:
            - Turtle mode for escape
            - Vandalism guard for protection
            - Escape path planning
        
        Args:
            surrounding_state: Current sensor state
            escalation_level: Protection level for vandalism guard
            
        Returns:
            Full defense engagement result
        """
        self.incident_id = f"incident_{uuid.uuid4().hex[:12]}"
        self.defense_engaged_at = datetime.utcnow()
        
        # Engage all systems in parallel
        turtle_result = await self.engage_turtle_mode(surrounding_state)
        guard_result = await self.engage_vandalism_guard(escalation_level)
        
        # Plan escape route
        escape_plan = self.escape_planner.plan_escape(
            GeoLocation(latitude=0, longitude=0),  # Would use actual location
            surrounding_state,
        )
        
        self.current_state = VehicleState.EMERGENCY_ESCAPE
        
        self._log_telemetry("full_defense_engaged", {
            "turtle_mode": turtle_result["success"],
            "vandalism_guard": guard_result["success"],
            "escape_plan": escape_plan["success"],
        })
        
        return {
            "success": True,
            "incident_id": self.incident_id,
            "state": self.current_state.value,
            "turtle_mode": turtle_result,
            "vandalism_guard": guard_result,
            "escape_plan": escape_plan,
            "alert_dispatch": True,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    async def disengage(self) -> Dict[str, Any]:
        """
        Disengage all defensive systems and return to normal operation.
        
        Returns:
            Disengagement summary with incident report
        """
        if self.current_state == VehicleState.NORMAL:
            return {"was_engaged": False}
        
        # Calculate incident duration
        duration_seconds = 0
        if self.defense_engaged_at:
            duration_seconds = (datetime.utcnow() - self.defense_engaged_at).total_seconds()
        
        # Deactivate systems
        turtle_result = self.turtle_mode.deactivate()
        guard_result = self.vandalism_guard.deactivate()
        
        # Generate incident report
        incident_report = {
            "incident_id": self.incident_id,
            "vehicle_id": self.vehicle_id,
            "duration_seconds": duration_seconds,
            "max_state_reached": self.current_state.value,
            "turtle_mode_used": turtle_result.get("was_active", False),
            "distance_escaped_meters": turtle_result.get("distance_covered_meters", 0),
            "vandalism_guard_used": guard_result.get("was_active", False),
            "max_escalation": guard_result.get("max_escalation_reached", 0),
            "telemetry_events": len(self.telemetry_log),
            "resolved_at": datetime.utcnow().isoformat(),
        }
        
        # Reset state
        self.current_state = VehicleState.NORMAL
        self.defense_engaged_at = None
        
        self._log_telemetry("defense_disengaged", incident_report)
        
        # Keep incident_id for report reference, but clear for next incident
        old_incident_id = self.incident_id
        self.incident_id = None
        
        return {
            "success": True,
            "incident_id": old_incident_id,
            "incident_report": incident_report,
            "state": VehicleState.NORMAL.value,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current defense unit status."""
        return {
            "vehicle_id": self.vehicle_id,
            "current_state": self.current_state.value,
            "incident_id": self.incident_id,
            "turtle_mode_active": self.turtle_mode.is_active,
            "vandalism_guard_active": self.vandalism_guard.is_active,
            "vandalism_guard_level": self.vandalism_guard.escalation_level,
            "defense_engaged_at": self.defense_engaged_at.isoformat() if self.defense_engaged_at else None,
            "telemetry_entries": len(self.telemetry_log),
        }
