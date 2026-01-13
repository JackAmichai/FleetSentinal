"""
FleetSentinel Edge Package

Layer 4: Vehicle Self-Defense and Escape Protocols
"""

from .vehicle_defense import (
    VehicleSelfDefenseUnit,
    TurtleMode,
    VandalismGuard,
    EscapePathPlanner,
)

__all__ = [
    "VehicleSelfDefenseUnit",
    "TurtleMode",
    "VandalismGuard",
    "EscapePathPlanner",
]
