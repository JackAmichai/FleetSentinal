"""
FleetSentinel Civic Data Client

This module provides integration with civic data sources to identify:
    - Protests and civil unrest
    - Emergency situations (fires, accidents)
    - Road closures and construction
    - Weather events

Sources integrated:
    - GDELT Project (global news events)
    - City permit databases
    - Police/Fire scanner APIs
    - Weather services

Why Civic Data?
    A sudden spike in ride requests to an area with ongoing civil unrest
    is HIGH RISK. We need to geofence dangerous zones to protect our assets.
    Conversely, a spike near a peaceful permitted event is LOW RISK.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum
import logging

import httpx


logger = logging.getLogger(__name__)


class CivicEventType(str, Enum):
    """Types of civic events we track."""
    PROTEST = "PROTEST"
    CIVIL_UNREST = "CIVIL_UNREST"
    EMERGENCY = "EMERGENCY"
    ROAD_CLOSURE = "ROAD_CLOSURE"
    CONSTRUCTION = "CONSTRUCTION"
    PARADE = "PARADE"
    MARATHON = "MARATHON"
    STREET_FAIR = "STREET_FAIR"
    WEATHER = "WEATHER"
    UNKNOWN = "UNKNOWN"


class CivicAlert:
    """
    Represents a civic event or alert from data sources.
    
    This could be:
        - A scheduled event (parade, marathon)
        - An unscheduled event (protest, emergency)
        - A disruption (road closure, construction)
    """
    
    def __init__(
        self,
        alert_id: str,
        event_type: CivicEventType,
        title: str,
        description: str,
        latitude: float,
        longitude: float,
        radius_meters: float = 500.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        severity: int = 1,  # 1-5 scale
        is_active: bool = True,
        source: str = "unknown",
    ):
        self.alert_id = alert_id
        self.event_type = event_type
        self.title = title
        self.description = description
        self.latitude = latitude
        self.longitude = longitude
        self.radius_meters = radius_meters
        self.start_time = start_time
        self.end_time = end_time
        self.severity = severity
        self.is_active = is_active
        self.source = source
    
    @property
    def is_dangerous(self) -> bool:
        """
        Determine if this event poses danger to vehicles/passengers.
        
        Dangerous events include:
            - Civil unrest
            - Protests (severity > 3)
            - Active emergencies
        """
        if self.event_type == CivicEventType.CIVIL_UNREST:
            return True
        if self.event_type == CivicEventType.PROTEST and self.severity >= 3:
            return True
        if self.event_type == CivicEventType.EMERGENCY and self.severity >= 4:
            return True
        return False
    
    @property
    def is_safe_crowd(self) -> bool:
        """
        Determine if this event represents a safe crowd gathering.
        
        Safe events include:
            - Parades (permitted)
            - Marathons
            - Street fairs
        """
        return self.event_type in (
            CivicEventType.PARADE,
            CivicEventType.MARATHON,
            CivicEventType.STREET_FAIR,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "event_type": self.event_type.value,
            "title": self.title,
            "description": self.description,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "radius_meters": self.radius_meters,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "severity": self.severity,
            "is_active": self.is_active,
            "is_dangerous": self.is_dangerous,
            "source": self.source,
        }


class CivicDataClient(ABC):
    """
    Abstract base class for civic data clients.
    
    All civic data sources should implement this interface.
    """
    
    @abstractmethod
    async def get_alerts_near_location(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 2.0,
    ) -> List[CivicAlert]:
        """
        Query civic alerts near a location.
        
        Args:
            latitude: Center latitude
            longitude: Center longitude
            radius_km: Search radius in kilometers
            
        Returns:
            List of CivicAlert objects
        """
        pass
    
    @abstractmethod
    async def get_active_protests(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 5.0,
    ) -> List[CivicAlert]:
        """
        Specifically query for protest/civil unrest data.
        
        This is prioritized because protests pose the highest risk.
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the data source is available."""
        pass


class MockCivicDataClient(CivicDataClient):
    """
    Mock civic data client for development and testing.
    
    Provides realistic mock data for testing different scenarios:
        - No alerts (normal)
        - Permitted event (safe crowd)
        - Active protest (dangerous)
        - Emergency situation
    
    Scenarios can be configured via add_mock_alert() for testing.
    """
    
    # Static list of mock alerts (can be modified per test)
    MOCK_ALERTS: List[Dict[str, Any]] = []
    
    def __init__(self):
        """Initialize the mock client with default alerts."""
        self._enabled = True
        self._generate_default_alerts()
    
    def _generate_default_alerts(self) -> None:
        """Generate default mock alerts for testing."""
        now = datetime.utcnow()
        
        self.MOCK_ALERTS = [
            # Peaceful parade in San Francisco
            {
                "alert_id": "civic_001",
                "event_type": CivicEventType.PARADE,
                "title": "Pride Parade",
                "description": "Annual Pride Parade on Market Street",
                "latitude": 37.7749,
                "longitude": -122.4194,
                "radius_meters": 1000,
                "start_time": now + timedelta(hours=1),
                "end_time": now + timedelta(hours=5),
                "severity": 1,
                "is_active": True,
                "source": "mock_civic",
            },
            # Road construction
            {
                "alert_id": "civic_002",
                "event_type": CivicEventType.CONSTRUCTION,
                "title": "Road Construction",
                "description": "Utility work on Mission Street",
                "latitude": 37.7600,
                "longitude": -122.4100,
                "radius_meters": 200,
                "start_time": now - timedelta(days=1),
                "end_time": now + timedelta(days=7),
                "severity": 2,
                "is_active": True,
                "source": "mock_civic",
            },
        ]
    
    def add_mock_alert(
        self,
        event_type: CivicEventType,
        title: str,
        latitude: float,
        longitude: float,
        severity: int = 1,
        description: str = "",
        is_active: bool = True,
    ) -> str:
        """
        Add a custom mock alert for testing.
        
        Example - add a protest:
            client.add_mock_alert(
                event_type=CivicEventType.PROTEST,
                title="Downtown Protest",
                latitude=37.7749,
                longitude=-122.4194,
                severity=4,
            )
        """
        alert_id = f"civic_{len(self.MOCK_ALERTS) + 1:03d}"
        now = datetime.utcnow()
        
        self.MOCK_ALERTS.append({
            "alert_id": alert_id,
            "event_type": event_type,
            "title": title,
            "description": description or f"{event_type.value} at location",
            "latitude": latitude,
            "longitude": longitude,
            "radius_meters": 500,
            "start_time": now - timedelta(hours=1),
            "end_time": now + timedelta(hours=3),
            "severity": severity,
            "is_active": is_active,
            "source": "mock_civic",
        })
        
        return alert_id
    
    def add_protest(
        self,
        latitude: float,
        longitude: float,
        severity: int = 4,
        title: str = "Active Protest",
    ) -> str:
        """Convenience method to add a protest alert."""
        return self.add_mock_alert(
            event_type=CivicEventType.PROTEST,
            title=title,
            latitude=latitude,
            longitude=longitude,
            severity=severity,
            description="Active protest gathering",
        )
    
    def add_civil_unrest(
        self,
        latitude: float,
        longitude: float,
        title: str = "Civil Unrest",
    ) -> str:
        """Convenience method to add civil unrest alert."""
        return self.add_mock_alert(
            event_type=CivicEventType.CIVIL_UNREST,
            title=title,
            latitude=latitude,
            longitude=longitude,
            severity=5,
            description="Civil unrest reported in area - exercise extreme caution",
        )
    
    def clear_alerts(self) -> None:
        """Clear all mock alerts."""
        self.MOCK_ALERTS = []
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in km between two points."""
        import math
        R = 6371
        
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
    
    async def get_alerts_near_location(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 2.0,
    ) -> List[CivicAlert]:
        """Query mock alerts near a location."""
        if not self._enabled:
            return []
        
        # Simulate API latency
        await asyncio.sleep(0.05)
        
        results: List[CivicAlert] = []
        
        for alert_data in self.MOCK_ALERTS:
            distance = self._haversine_distance(
                latitude, longitude,
                alert_data["latitude"], alert_data["longitude"]
            )
            
            if distance <= radius_km:
                alert = CivicAlert(
                    alert_id=alert_data["alert_id"],
                    event_type=alert_data["event_type"],
                    title=alert_data["title"],
                    description=alert_data["description"],
                    latitude=alert_data["latitude"],
                    longitude=alert_data["longitude"],
                    radius_meters=alert_data["radius_meters"],
                    start_time=alert_data["start_time"],
                    end_time=alert_data["end_time"],
                    severity=alert_data["severity"],
                    is_active=alert_data["is_active"],
                    source=alert_data["source"],
                )
                results.append(alert)
        
        logger.debug(f"Mock Civic Data: Found {len(results)} alerts near ({latitude}, {longitude})")
        return results
    
    async def get_active_protests(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 5.0,
    ) -> List[CivicAlert]:
        """Get protests and civil unrest alerts."""
        all_alerts = await self.get_alerts_near_location(latitude, longitude, radius_km)
        
        return [
            alert for alert in all_alerts
            if alert.event_type in (CivicEventType.PROTEST, CivicEventType.CIVIL_UNREST)
            and alert.is_active
        ]
    
    async def health_check(self) -> bool:
        """Mock health check."""
        await asyncio.sleep(0.01)
        return self._enabled
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable the mock client."""
        self._enabled = enabled


class GDELTClient(CivicDataClient):
    """
    GDELT Project API client for global news and events.
    
    The GDELT Project monitors news media worldwide and provides:
        - Protest tracking
        - Civil unrest detection
        - Emergency event monitoring
        
    API: https://api.gdeltproject.org/
    
    Note: GDELT data may have 15-60 minute delay from real events.
    """
    
    BASE_URL = "https://api.gdeltproject.org/api/v2"
    
    def __init__(self, timeout_seconds: float = 5.0):
        """Initialize GDELT client."""
        self._timeout = timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=self._timeout,
            )
        return self._client
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def get_alerts_near_location(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 2.0,
    ) -> List[CivicAlert]:
        """
        Query GDELT for news events near a location.
        
        Uses the GDELT GEO API to find geographically tagged events.
        """
        try:
            client = await self._get_client()
            
            # GDELT GEO query
            # Format: latitude:longitude:radiuskm
            geo_query = f"near:{latitude}:{longitude}:{int(radius_km)}"
            
            params = {
                "query": f"{geo_query} protest OR riot OR demonstration OR unrest",
                "mode": "artlist",
                "maxrecords": 50,
                "format": "json",
                "timespan": "7d",
            }
            
            response = await client.get("/doc/doc", params=params)
            response.raise_for_status()
            
            data = response.json()
            alerts = []
            
            for article in data.get("articles", []):
                alert = self._parse_article(article)
                if alert:
                    alerts.append(alert)
            
            return alerts
            
        except httpx.HTTPError as e:
            logger.error(f"GDELT API error: {e}")
            return []
    
    def _parse_article(self, article: Dict[str, Any]) -> Optional[CivicAlert]:
        """Parse GDELT article into CivicAlert."""
        # GDELT article parsing would go here
        # This is a simplified implementation
        title = article.get("title", "")
        
        # Determine event type from keywords
        title_lower = title.lower()
        if "protest" in title_lower:
            event_type = CivicEventType.PROTEST
            severity = 3
        elif "riot" in title_lower or "unrest" in title_lower:
            event_type = CivicEventType.CIVIL_UNREST
            severity = 5
        else:
            return None
        
        # Get location (GDELT provides these)
        lat = article.get("latitude")
        lon = article.get("longitude")
        
        if lat is None or lon is None:
            return None
        
        return CivicAlert(
            alert_id=f"gdelt_{hash(title) % 10000:04d}",
            event_type=event_type,
            title=title[:100],
            description=article.get("seendate", ""),
            latitude=lat,
            longitude=lon,
            severity=severity,
            source="gdelt",
        )
    
    async def get_active_protests(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 5.0,
    ) -> List[CivicAlert]:
        """Get protest-specific alerts from GDELT."""
        alerts = await self.get_alerts_near_location(latitude, longitude, radius_km)
        return [
            a for a in alerts 
            if a.event_type in (CivicEventType.PROTEST, CivicEventType.CIVIL_UNREST)
        ]
    
    async def health_check(self) -> bool:
        """Check GDELT API availability."""
        try:
            client = await self._get_client()
            response = await client.get(
                "/doc/doc",
                params={"query": "test", "mode": "artlist", "maxrecords": 1, "format": "json"}
            )
            return response.status_code == 200
        except Exception:
            return False
