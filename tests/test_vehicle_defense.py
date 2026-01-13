"""
FleetSentinel Test - Vehicle Defense Module

Tests for the Layer 4 Vehicle Self-Defense Unit.
Validates:
- Turtle mode activation and parameters
- Vandalism escalation levels
- Escape path planning
- PyTorch threat detection model
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime
from typing import Optional

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import (
    GeoLocation,
    ThreatLevel,
)
from edge.vehicle_defense import (
    VehicleSelfDefenseUnit,
    TurtleMode,
    VandalismGuard,
    EscapePathPlanner,
    ThreatDetectionModel,
    VehicleState,
    DefenseConfig,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def defense_unit():
    """Create a vehicle self-defense unit."""
    config = DefenseConfig(
        turtle_speed_mph=0.5,
        turtle_gap_meters=0.5,
        max_escalation_level=3,
        escape_route_radius_meters=500,
    )
    return VehicleSelfDefenseUnit(
        vehicle_id="vehicle_001",
        config=config,
    )


@pytest.fixture
def vehicle_state():
    """Create a sample vehicle state."""
    return VehicleState(
        vehicle_id="vehicle_001",
        location=GeoLocation(latitude=37.7749, longitude=-122.4194),
        heading_degrees=90.0,
        speed_mph=15.0,
        battery_percent=85.0,
        occupied=True,
        current_threat_level=ThreatLevel.NONE,
    )


@pytest.fixture
def threat_model():
    """Create a threat detection model."""
    return ThreatDetectionModel()


# =============================================================================
# UNIT TESTS - TURTLE MODE
# =============================================================================

class TestTurtleMode:
    """Tests for Turtle Mode activation."""
    
    def test_turtle_mode_defaults(self):
        """Test default turtle mode parameters."""
        turtle = TurtleMode()
        
        assert turtle.active is False
        assert turtle.max_speed_mph == 0.5
        assert turtle.min_gap_meters == 0.5
    
    def test_turtle_mode_activation(self):
        """Test activating turtle mode."""
        turtle = TurtleMode()
        
        turtle.activate()
        
        assert turtle.active is True
        assert turtle.activation_time is not None
    
    def test_turtle_mode_deactivation(self):
        """Test deactivating turtle mode."""
        turtle = TurtleMode()
        
        turtle.activate()
        assert turtle.active is True
        
        turtle.deactivate()
        assert turtle.active is False
    
    def test_turtle_mode_speed_limit(self):
        """Test speed limiting in turtle mode."""
        turtle = TurtleMode(max_speed_mph=0.5)
        turtle.activate()
        
        # Request 30 mph, should get turtle speed
        limited_speed = turtle.limit_speed(30.0)
        
        assert limited_speed == 0.5
    
    def test_turtle_mode_gap_enforcement(self):
        """Test minimum gap enforcement."""
        turtle = TurtleMode(min_gap_meters=0.5)
        turtle.activate()
        
        # Request 0.1m gap, should get minimum
        enforced_gap = turtle.enforce_gap(0.1)
        
        assert enforced_gap == 0.5
    
    def test_inactive_turtle_mode_no_limits(self):
        """Test that inactive turtle mode doesn't limit."""
        turtle = TurtleMode(max_speed_mph=0.5)
        
        # Not activated, should return original speed
        limited_speed = turtle.limit_speed(30.0)
        
        assert limited_speed == 30.0


class TestVandalismGuard:
    """Tests for Vandalism Guard escalation."""
    
    def test_escalation_levels(self):
        """Test vandalism escalation levels."""
        guard = VandalismGuard()
        
        assert guard.current_level == 0
        
        # Escalate through levels
        guard.escalate()
        assert guard.current_level == 1
        
        guard.escalate()
        assert guard.current_level == 2
        
        guard.escalate()
        assert guard.current_level == 3
        
        # Should not exceed max
        guard.escalate()
        assert guard.current_level == 3
    
    def test_escalation_actions(self):
        """Test actions at each escalation level."""
        guard = VandalismGuard()
        
        # Level 0: Normal
        actions = guard.get_current_actions()
        assert "monitor" in actions
        
        # Level 1: Warning
        guard.escalate()
        actions = guard.get_current_actions()
        assert "record_video" in actions
        assert "audio_warning" in actions
        
        # Level 2: Alert
        guard.escalate()
        actions = guard.get_current_actions()
        assert "alert_authorities" in actions
        assert "enable_siren" in actions
        
        # Level 3: Maximum
        guard.escalate()
        actions = guard.get_current_actions()
        assert "lock_doors" in actions
        assert "disable_vehicle" in actions
    
    def test_de_escalation(self):
        """Test de-escalation after calm period."""
        guard = VandalismGuard()
        
        guard.escalate()
        guard.escalate()
        assert guard.current_level == 2
        
        guard.de_escalate()
        assert guard.current_level == 1
        
        guard.de_escalate()
        assert guard.current_level == 0
        
        # Should not go below 0
        guard.de_escalate()
        assert guard.current_level == 0
    
    def test_auto_de_escalation(self):
        """Test automatic de-escalation after time."""
        guard = VandalismGuard(auto_deescalate_seconds=1)
        
        guard.escalate()
        guard.escalate()
        assert guard.current_level == 2
        
        # Simulate time passing (in real implementation)
        # For now, just test the method exists
        guard.check_auto_deescalate()


class TestEscapePathPlanner:
    """Tests for Escape Path Planning."""
    
    @pytest.fixture
    def planner(self):
        """Create an escape path planner."""
        return EscapePathPlanner(search_radius_meters=500)
    
    def test_plan_escape_route(self, planner):
        """Test escape route planning."""
        current_location = GeoLocation(latitude=37.7749, longitude=-122.4194)
        threat_location = GeoLocation(latitude=37.7750, longitude=-122.4190)
        
        route = planner.plan_escape_route(
            current_location=current_location,
            threat_location=threat_location,
        )
        
        assert route is not None
        assert route.waypoints is not None
        assert len(route.waypoints) > 0
        
        # Route should move away from threat
        first_waypoint = route.waypoints[0]
        # Simple check: not going toward threat
        assert first_waypoint != threat_location
    
    def test_escape_route_with_multiple_threats(self, planner):
        """Test escape planning with multiple threats."""
        current_location = GeoLocation(latitude=37.7749, longitude=-122.4194)
        threats = [
            GeoLocation(latitude=37.7750, longitude=-122.4190),
            GeoLocation(latitude=37.7748, longitude=-122.4198),
        ]
        
        route = planner.plan_escape_route(
            current_location=current_location,
            threat_locations=threats,
        )
        
        assert route is not None
        assert route.safe_zone_distance_meters > 0
    
    def test_no_escape_route_when_trapped(self, planner):
        """Test handling when no escape route exists."""
        current_location = GeoLocation(latitude=37.7749, longitude=-122.4194)
        
        # Surrounded by threats
        threats = [
            GeoLocation(latitude=37.7750, longitude=-122.4194),  # North
            GeoLocation(latitude=37.7748, longitude=-122.4194),  # South
            GeoLocation(latitude=37.7749, longitude=-122.4190),  # East
            GeoLocation(latitude=37.7749, longitude=-122.4198),  # West
        ]
        
        route = planner.plan_escape_route(
            current_location=current_location,
            threat_locations=threats,
        )
        
        # Should return a route even if suboptimal
        assert route is not None
        # But may flag as high risk
        assert route.risk_level >= ThreatLevel.MEDIUM


class TestThreatDetectionModel:
    """Tests for the PyTorch threat detection model."""
    
    def test_model_initialization(self, threat_model):
        """Test model initializes correctly."""
        assert threat_model is not None
        assert threat_model.model is not None
    
    def test_model_inference(self, threat_model):
        """Test model can run inference."""
        # Create dummy sensor data
        sensor_data = {
            "accelerometer": [0.1, 0.2, 9.8],  # Normal gravity
            "gyroscope": [0.0, 0.0, 0.0],  # No rotation
            "proximity": [100, 100, 100, 100],  # Nothing close
            "audio_level_db": 60,  # Normal ambient
        }
        
        prediction = threat_model.predict(sensor_data)
        
        assert prediction is not None
        assert "threat_probability" in prediction
        assert 0.0 <= prediction["threat_probability"] <= 1.0
    
    def test_model_detects_impact(self, threat_model):
        """Test model detects physical impact."""
        # Simulate impact sensor data
        sensor_data = {
            "accelerometer": [5.0, 10.0, 15.0],  # High acceleration
            "gyroscope": [2.0, 3.0, 1.0],  # Rapid rotation
            "proximity": [10, 5, 10, 100],  # Something very close
            "audio_level_db": 95,  # Loud noise
        }
        
        prediction = threat_model.predict(sensor_data)
        
        # Should detect threat
        assert prediction["threat_probability"] > 0.5
    
    def test_model_normal_driving(self, threat_model):
        """Test model recognizes normal driving conditions."""
        # Normal driving sensor data
        sensor_data = {
            "accelerometer": [0.5, 0.3, 9.8],  # Minor acceleration
            "gyroscope": [0.1, 0.1, 0.5],  # Slight turn
            "proximity": [200, 150, 200, 300],  # Normal distances
            "audio_level_db": 70,  # Road noise
        }
        
        prediction = threat_model.predict(sensor_data)
        
        # Should not detect threat
        assert prediction["threat_probability"] < 0.3


# =============================================================================
# INTEGRATION TESTS - DEFENSE UNIT
# =============================================================================

class TestVehicleSelfDefenseUnit:
    """Integration tests for the full defense unit."""
    
    @pytest.mark.asyncio
    async def test_defense_unit_initialization(self, defense_unit):
        """Test defense unit initializes correctly."""
        await defense_unit.initialize()
        
        assert defense_unit.initialized is True
        assert defense_unit.turtle_mode is not None
        assert defense_unit.vandalism_guard is not None
    
    @pytest.mark.asyncio
    async def test_threat_response_low(self, defense_unit, vehicle_state):
        """Test response to low threat level."""
        await defense_unit.initialize()
        
        response = await defense_unit.respond_to_threat(
            vehicle_state=vehicle_state,
            threat_level=ThreatLevel.LOW,
        )
        
        assert response.turtle_mode_active is False
        assert response.escalation_level == 0
        assert "monitor" in response.actions
    
    @pytest.mark.asyncio
    async def test_threat_response_medium(self, defense_unit, vehicle_state):
        """Test response to medium threat level."""
        await defense_unit.initialize()
        
        response = await defense_unit.respond_to_threat(
            vehicle_state=vehicle_state,
            threat_level=ThreatLevel.MEDIUM,
        )
        
        assert response.turtle_mode_active is True
        assert response.escalation_level >= 1
        assert "record_video" in response.actions
    
    @pytest.mark.asyncio
    async def test_threat_response_high(self, defense_unit, vehicle_state):
        """Test response to high threat level."""
        await defense_unit.initialize()
        
        response = await defense_unit.respond_to_threat(
            vehicle_state=vehicle_state,
            threat_level=ThreatLevel.HIGH,
        )
        
        assert response.turtle_mode_active is True
        assert response.escalation_level >= 2
        assert "alert_authorities" in response.actions
        assert response.escape_route is not None
    
    @pytest.mark.asyncio
    async def test_threat_response_critical(self, defense_unit, vehicle_state):
        """Test response to critical threat level."""
        await defense_unit.initialize()
        
        response = await defense_unit.respond_to_threat(
            vehicle_state=vehicle_state,
            threat_level=ThreatLevel.CRITICAL,
        )
        
        assert response.turtle_mode_active is True
        assert response.escalation_level == 3  # Maximum
        assert "lock_doors" in response.actions
        assert "disable_vehicle" in response.actions
        assert response.escape_route is not None
    
    @pytest.mark.asyncio
    async def test_continuous_monitoring(self, defense_unit, vehicle_state):
        """Test continuous threat monitoring."""
        await defense_unit.initialize()
        
        # Simulate sensor readings over time
        sensor_readings = [
            {"accelerometer": [0.1, 0.2, 9.8], "proximity": [100, 100, 100, 100]},
            {"accelerometer": [1.0, 1.0, 9.8], "proximity": [50, 100, 100, 100]},
            {"accelerometer": [5.0, 5.0, 9.8], "proximity": [10, 20, 100, 100]},
        ]
        
        for reading in sensor_readings:
            threat = await defense_unit.analyze_sensors(reading)
            assert threat is not None
            assert "threat_level" in threat


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
