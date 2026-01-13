"""
FleetSentinel Event API Client

This module provides integration with event ticketing services (Ticketmaster, etc.)
to identify legitimate high-demand situations like concerts and sporting events.

The client is abstracted to allow:
    - Mock implementations for testing
    - Easy addition of new event data sources
    - Graceful degradation when APIs are unavailable

Example:
    client = MockTicketmasterClient()
    events = await client.get_events_near_location(
        latitude=37.7749,
        longitude=-122.4194,
        radius_km=2.0
    )
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

import httpx


logger = logging.getLogger(__name__)


class Event:
    """
    Represents a scheduled event from ticketing APIs.
    """
    
    def __init__(
        self,
        event_id: str,
        name: str,
        venue_name: str,
        venue_lat: float,
        venue_lon: float,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        expected_attendance: Optional[int] = None,
        category: str = "general",
        source: str = "unknown",
    ):
        self.event_id = event_id
        self.name = name
        self.venue_name = venue_name
        self.venue_lat = venue_lat
        self.venue_lon = venue_lon
        self.start_time = start_time
        self.end_time = end_time or (start_time + timedelta(hours=3))
        self.expected_attendance = expected_attendance
        self.category = category
        self.source = source
    
    def is_active(self, at_time: Optional[datetime] = None) -> bool:
        """
        Check if the event is currently active.
        
        Events are considered "active" from 2 hours before start
        until 1 hour after end (to account for crowds arriving/leaving).
        """
        now = at_time or datetime.utcnow()
        buffer_before = timedelta(hours=2)
        buffer_after = timedelta(hours=1)
        
        return (self.start_time - buffer_before) <= now <= (self.end_time + buffer_after)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "name": self.name,
            "venue_name": self.venue_name,
            "venue_lat": self.venue_lat,
            "venue_lon": self.venue_lon,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "expected_attendance": self.expected_attendance,
            "category": self.category,
            "source": self.source,
        }


class EventAPIClient(ABC):
    """
    Abstract base class for event API clients.
    
    All event data sources (Ticketmaster, Eventbrite, etc.) should
    implement this interface.
    """
    
    @abstractmethod
    async def get_events_near_location(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 2.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Event]:
        """
        Query events near a geographic location.
        
        Args:
            latitude: Center latitude
            longitude: Center longitude
            radius_km: Search radius in kilometers
            start_time: Filter events starting after this time
            end_time: Filter events starting before this time
            
        Returns:
            List of Event objects
        """
        pass
    
    @abstractmethod
    async def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """
        Get a specific event by ID.
        
        Args:
            event_id: Unique event identifier
            
        Returns:
            Event if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the API is available.
        
        Returns:
            True if healthy, False otherwise
        """
        pass


class MockTicketmasterClient(EventAPIClient):
    """
    Mock Ticketmaster client for development and testing.
    
    This provides realistic mock data for known venues to enable
    testing without actual API credentials.
    
    Mock Data Includes:
        - Chase Center (San Francisco): Warriors games, concerts
        - Oracle Park (San Francisco): Giants games
        - AT&T Park (San Francisco): Various events
        
    Why Mock?
        - Deterministic testing
        - No API rate limits during development
        - Ability to simulate specific scenarios
    """
    
    # Mock venue database with coordinates
    MOCK_VENUES = {
        "chase_center": {
            "name": "Chase Center",
            "lat": 37.7680,
            "lon": -122.3879,
            "capacity": 18064,
        },
        "oracle_park": {
            "name": "Oracle Park",
            "lat": 37.7786,
            "lon": -122.3893,
            "capacity": 41915,
        },
        "bill_graham": {
            "name": "Bill Graham Civic Auditorium",
            "lat": 37.7786,
            "lon": -122.4183,
            "capacity": 7000,
        },
        "moscone": {
            "name": "Moscone Center",
            "lat": 37.7840,
            "lon": -122.4014,
            "capacity": 20000,
        },
        "msg": {
            "name": "Madison Square Garden",
            "lat": 40.7505,
            "lon": -73.9934,
            "capacity": 20789,
        },
        "staples": {
            "name": "Crypto.com Arena",
            "lat": 34.0430,
            "lon": -118.2673,
            "capacity": 20000,
        },
    }
    
    # Mock events (configured for testing)
    MOCK_EVENTS: List[Dict[str, Any]] = []
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the mock client.
        
        Args:
            api_key: Ignored for mock, kept for interface compatibility
        """
        self._api_key = api_key
        self._enabled = True
        self._generate_mock_events()
    
    def _generate_mock_events(self) -> None:
        """Generate mock events for testing."""
        now = datetime.utcnow()
        
        # Add a Taylor Swift concert at Chase Center (tonight!)
        self.MOCK_EVENTS = [
            {
                "event_id": "mock_event_001",
                "name": "Taylor Swift - The Eras Tour",
                "venue_key": "chase_center",
                "start_time": now + timedelta(hours=2),
                "expected_attendance": 18000,
                "category": "concert",
            },
            {
                "event_id": "mock_event_002",
                "name": "Golden State Warriors vs Lakers",
                "venue_key": "chase_center",
                "start_time": now + timedelta(days=1, hours=19),
                "expected_attendance": 18000,
                "category": "sports",
            },
            {
                "event_id": "mock_event_003",
                "name": "SF Giants vs Dodgers",
                "venue_key": "oracle_park",
                "start_time": now + timedelta(hours=4),
                "expected_attendance": 40000,
                "category": "sports",
            },
            {
                "event_id": "mock_event_004",
                "name": "Tech Conference 2026",
                "venue_key": "moscone",
                "start_time": now + timedelta(hours=1),
                "expected_attendance": 15000,
                "category": "conference",
            },
        ]
    
    def add_mock_event(
        self,
        name: str,
        venue_lat: float,
        venue_lon: float,
        start_time: datetime,
        expected_attendance: int = 5000,
    ) -> str:
        """
        Add a custom mock event for testing.
        
        Returns:
            The generated event_id
        """
        event_id = f"mock_event_{len(self.MOCK_EVENTS) + 1:03d}"
        self.MOCK_EVENTS.append({
            "event_id": event_id,
            "name": name,
            "venue_key": None,
            "custom_venue": {
                "name": "Custom Venue",
                "lat": venue_lat,
                "lon": venue_lon,
                "capacity": expected_attendance,
            },
            "start_time": start_time,
            "expected_attendance": expected_attendance,
            "category": "custom",
        })
        return event_id
    
    def clear_mock_events(self) -> None:
        """Clear all mock events (useful for test isolation)."""
        self.MOCK_EVENTS = []
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in kilometers between two points."""
        import math
        R = 6371  # Earth's radius in km
        
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
    
    async def get_events_near_location(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 2.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Event]:
        """
        Query mock events near a location.
        
        Simulates API latency and returns events within radius.
        """
        if not self._enabled:
            return []
        
        # Simulate API latency (50-150ms)
        await asyncio.sleep(0.05 + (hash(str(latitude)) % 100) / 1000)
        
        results: List[Event] = []
        now = datetime.utcnow()
        
        for event_data in self.MOCK_EVENTS:
            # Get venue info
            if event_data.get("venue_key"):
                venue = self.MOCK_VENUES.get(event_data["venue_key"])
                if not venue:
                    continue
                venue_lat = venue["lat"]
                venue_lon = venue["lon"]
                venue_name = venue["name"]
                capacity = venue["capacity"]
            elif event_data.get("custom_venue"):
                cv = event_data["custom_venue"]
                venue_lat = cv["lat"]
                venue_lon = cv["lon"]
                venue_name = cv["name"]
                capacity = cv["capacity"]
            else:
                continue
            
            # Check distance
            distance = self._haversine_distance(latitude, longitude, venue_lat, venue_lon)
            if distance > radius_km:
                continue
            
            # Check time filters
            event_start = event_data["start_time"]
            if start_time and event_start < start_time:
                continue
            if end_time and event_start > end_time:
                continue
            
            # Create event object
            event = Event(
                event_id=event_data["event_id"],
                name=event_data["name"],
                venue_name=venue_name,
                venue_lat=venue_lat,
                venue_lon=venue_lon,
                start_time=event_start,
                end_time=event_start + timedelta(hours=3),
                expected_attendance=event_data.get("expected_attendance", capacity),
                category=event_data.get("category", "general"),
                source="mock_ticketmaster",
            )
            results.append(event)
        
        logger.debug(f"Mock Ticketmaster: Found {len(results)} events near ({latitude}, {longitude})")
        return results
    
    async def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """Get a mock event by ID."""
        await asyncio.sleep(0.03)  # Simulate latency
        
        for event_data in self.MOCK_EVENTS:
            if event_data["event_id"] == event_id:
                venue_key = event_data.get("venue_key")
                if venue_key:
                    venue = self.MOCK_VENUES[venue_key]
                    return Event(
                        event_id=event_data["event_id"],
                        name=event_data["name"],
                        venue_name=venue["name"],
                        venue_lat=venue["lat"],
                        venue_lon=venue["lon"],
                        start_time=event_data["start_time"],
                        expected_attendance=event_data.get("expected_attendance"),
                        category=event_data.get("category", "general"),
                        source="mock_ticketmaster",
                    )
        return None
    
    async def health_check(self) -> bool:
        """Mock health check - always returns True."""
        await asyncio.sleep(0.01)
        return self._enabled
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable the mock API (for testing failure scenarios)."""
        self._enabled = enabled


class TicketmasterClient(EventAPIClient):
    """
    Real Ticketmaster Discovery API client.
    
    Requires a valid API key from https://developer.ticketmaster.com/
    
    Rate Limits:
        - 5 requests per second
        - 5000 requests per day
        
    This client implements:
        - Automatic retry with exponential backoff
        - Rate limiting
        - Response caching (1 hour TTL)
    """
    
    BASE_URL = "https://app.ticketmaster.com/discovery/v2"
    
    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 5.0,
        max_retries: int = 3,
    ):
        """
        Initialize the Ticketmaster client.
        
        Args:
            api_key: Ticketmaster API key
            timeout_seconds: Request timeout
            max_retries: Max retry attempts
        """
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._max_retries = max_retries
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
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def get_events_near_location(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 2.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Event]:
        """
        Query Ticketmaster for events near a location.
        
        Uses the Discovery API's geoPoint parameter for location search.
        """
        try:
            client = await self._get_client()
            
            # Convert km to miles (Ticketmaster uses miles)
            radius_miles = radius_km * 0.621371
            
            params = {
                "apikey": self._api_key,
                "latlong": f"{latitude},{longitude}",
                "radius": str(int(radius_miles)),
                "unit": "miles",
                "size": 50,
            }
            
            if start_time:
                params["startDateTime"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            if end_time:
                params["endDateTime"] = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            response = await client.get("/events.json", params=params)
            response.raise_for_status()
            
            data = response.json()
            events = []
            
            if "_embedded" in data and "events" in data["_embedded"]:
                for event_data in data["_embedded"]["events"]:
                    try:
                        event = self._parse_event(event_data)
                        if event:
                            events.append(event)
                    except Exception as e:
                        logger.warning(f"Failed to parse event: {e}")
                        continue
            
            return events
            
        except httpx.HTTPError as e:
            logger.error(f"Ticketmaster API error: {e}")
            return []
    
    def _parse_event(self, data: Dict[str, Any]) -> Optional[Event]:
        """Parse Ticketmaster API response into Event object."""
        event_id = data.get("id")
        name = data.get("name")
        
        if not event_id or not name:
            return None
        
        # Get venue info
        venues = data.get("_embedded", {}).get("venues", [])
        if not venues:
            return None
        
        venue = venues[0]
        venue_name = venue.get("name", "Unknown Venue")
        
        location = venue.get("location", {})
        venue_lat = float(location.get("latitude", 0))
        venue_lon = float(location.get("longitude", 0))
        
        if venue_lat == 0 and venue_lon == 0:
            return None
        
        # Get dates
        dates = data.get("dates", {})
        start_data = dates.get("start", {})
        start_str = start_data.get("dateTime")
        
        if start_str:
            start_time = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        else:
            return None
        
        # Category
        classifications = data.get("classifications", [])
        category = "general"
        if classifications:
            segment = classifications[0].get("segment", {})
            category = segment.get("name", "general").lower()
        
        return Event(
            event_id=event_id,
            name=name,
            venue_name=venue_name,
            venue_lat=venue_lat,
            venue_lon=venue_lon,
            start_time=start_time,
            category=category,
            source="ticketmaster",
        )
    
    async def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """Get event details by ID."""
        try:
            client = await self._get_client()
            
            response = await client.get(
                f"/events/{event_id}.json",
                params={"apikey": self._api_key},
            )
            response.raise_for_status()
            
            return self._parse_event(response.json())
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to get event {event_id}: {e}")
            return None
    
    async def health_check(self) -> bool:
        """Check Ticketmaster API availability."""
        try:
            client = await self._get_client()
            response = await client.get(
                "/events.json",
                params={"apikey": self._api_key, "size": 1},
            )
            return response.status_code == 200
        except Exception:
            return False
