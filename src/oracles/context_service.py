"""
FleetSentinel Context Verification Service

Layer 2: The Context Oracle - The "False Positive Killer"

This service is CRITICAL for preventing false positives. Without it,
every Taylor Swift concert would trigger a lockdown.

Logic Flow:
    1. IF Density is High AND EventAPI confirms a concert -> SAFE (High Demand Mode)
    2. IF Density is High AND NewsAPI confirms Civil Unrest -> DANGER (Geofence Lock)
    3. IF Density is High AND No Context -> SUSPICIOUS (Trigger Defense)

The service queries multiple sources in parallel and synthesizes
the results into a unified context assessment.

Example:
    service = ContextVerificationService()
    context = await service.verify_context(
        latitude=37.7680,
        longitude=-122.3879,
        density_score=500.0,
    )
    
    if context.is_legitimate_event:
        # Allow high demand
    elif context.is_dangerous:
        # Geofence and block
    else:
        # Proceed with liability gate
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import logging

from ..models import GeoLocation, ContextData, ContextType
from .event_api import EventAPIClient, MockTicketmasterClient, Event
from .civic_data import CivicDataClient, MockCivicDataClient, CivicAlert, CivicEventType


logger = logging.getLogger(__name__)


class ContextVerdict(str, Enum):
    """
    The final verdict from context verification.
    
    SAFE_HIGH_DEMAND: Legitimate event causing crowd (allow with monitoring)
    SAFE_INFRASTRUCTURE: Road work, etc. (allow with alternate routing)
    DANGEROUS: Civil unrest or emergency (geofence and block)
    SUSPICIOUS: No context found for high density (proceed with caution)
    NORMAL: Low density, no special context (allow)
    """
    SAFE_HIGH_DEMAND = "SAFE_HIGH_DEMAND"
    SAFE_INFRASTRUCTURE = "SAFE_INFRASTRUCTURE"
    DANGEROUS = "DANGEROUS"
    SUSPICIOUS = "SUSPICIOUS"
    NORMAL = "NORMAL"


class ContextResult:
    """
    Complete context verification result.
    
    Contains all information gathered from oracle sources,
    synthesized into actionable intelligence.
    """
    
    def __init__(
        self,
        verdict: ContextVerdict,
        context_type: ContextType,
        confidence: float,
        events: Optional[List[Event]] = None,
        alerts: Optional[List[CivicAlert]] = None,
        primary_event: Optional[Event] = None,
        primary_alert: Optional[CivicAlert] = None,
        explanation: str = "",
        recommendations: Optional[List[str]] = None,
    ):
        self.verdict = verdict
        self.context_type = context_type
        self.confidence = confidence
        self.events = events or []
        self.alerts = alerts or []
        self.primary_event = primary_event
        self.primary_alert = primary_alert
        self.explanation = explanation
        self.recommendations = recommendations or []
        self.timestamp = datetime.utcnow()
    
    @property
    def is_legitimate_event(self) -> bool:
        """True if density is explained by a legitimate event."""
        return self.verdict in (
            ContextVerdict.SAFE_HIGH_DEMAND,
            ContextVerdict.SAFE_INFRASTRUCTURE,
        )
    
    @property
    def is_dangerous(self) -> bool:
        """True if the situation poses danger to vehicles/passengers."""
        return self.verdict == ContextVerdict.DANGEROUS
    
    @property
    def modifier(self) -> float:
        """
        Get the confidence modifier for threat calculation.
        
        Positive = increases threat
        Negative = decreases threat
        """
        modifiers = {
            ContextVerdict.SAFE_HIGH_DEMAND: -5.0,
            ContextVerdict.SAFE_INFRASTRUCTURE: -2.0,
            ContextVerdict.DANGEROUS: 5.0,
            ContextVerdict.SUSPICIOUS: 0.0,
            ContextVerdict.NORMAL: 0.0,
        }
        return modifiers.get(self.verdict, 0.0) * self.confidence
    
    def to_context_data(self) -> ContextData:
        """Convert to ContextData model for threat assessment."""
        event_name = None
        event_venue = None
        expected_attendance = None
        start_time = None
        end_time = None
        
        if self.primary_event:
            event_name = self.primary_event.name
            event_venue = self.primary_event.venue_name
            expected_attendance = self.primary_event.expected_attendance
            start_time = self.primary_event.start_time
            end_time = self.primary_event.end_time
        elif self.primary_alert:
            event_name = self.primary_alert.title
            start_time = self.primary_alert.start_time
            end_time = self.primary_alert.end_time
        
        return ContextData(
            context_type=self.context_type,
            confidence=self.confidence,
            event_name=event_name,
            event_venue=event_venue,
            expected_attendance=expected_attendance,
            start_time=start_time,
            end_time=end_time,
            source="context_oracle",
            raw_data={
                "verdict": self.verdict.value,
                "events_count": len(self.events),
                "alerts_count": len(self.alerts),
            },
        )


class ContextOracle:
    """
    A single oracle source with health tracking.
    
    This wraps individual API clients and tracks their availability
    to enable graceful degradation.
    """
    
    def __init__(
        self,
        name: str,
        client: Any,
        timeout_seconds: float = 2.0,
    ):
        self.name = name
        self.client = client
        self.timeout = timeout_seconds
        self.is_healthy = True
        self.last_check = datetime.utcnow()
        self.failure_count = 0
        self.success_count = 0
    
    async def check_health(self) -> bool:
        """Check if this oracle is healthy."""
        try:
            result = await asyncio.wait_for(
                self.client.health_check(),
                timeout=self.timeout,
            )
            self.is_healthy = result
            self.last_check = datetime.utcnow()
            if result:
                self.success_count += 1
                self.failure_count = 0
            else:
                self.failure_count += 1
            return result
        except Exception as e:
            logger.warning(f"Oracle {self.name} health check failed: {e}")
            self.failure_count += 1
            self.is_healthy = False
            return False


class ContextVerificationService:
    """
    The Context Oracle Service - Prevents false positives.
    
    This service is the "brain" that synthesizes multiple data sources
    to understand WHY there's high demand at a location.
    
    Data Sources:
        - Ticketmaster (concerts, sports, theater)
        - Civic Data (protests, emergencies, construction)
        - Social Sentiment (future: Twitter/Reddit monitoring)
    
    Decision Logic:
        1. Query all sources in parallel
        2. Prioritize danger signals (civil unrest trumps concert)
        3. If event found, reduce threat score
        4. If no context, maintain suspicion
    
    Performance:
        - All API calls have 2-second timeout
        - Failed sources are gracefully degraded
        - Results are cached for 5 minutes
    """
    
    def __init__(
        self,
        event_client: Optional[EventAPIClient] = None,
        civic_client: Optional[CivicDataClient] = None,
        timeout_seconds: float = 2.0,
        cache_ttl_seconds: int = 300,
    ):
        """
        Initialize the Context Verification Service.
        
        Args:
            event_client: Event API client (defaults to mock)
            civic_client: Civic data client (defaults to mock)
            timeout_seconds: Timeout for API calls
            cache_ttl_seconds: How long to cache results
        """
        self._event_client = event_client or MockTicketmasterClient()
        self._civic_client = civic_client or MockCivicDataClient()
        self._timeout = timeout_seconds
        self._cache_ttl = cache_ttl_seconds
        
        # Result cache: {location_hash: (result, timestamp)}
        self._cache: Dict[str, Tuple[ContextResult, datetime]] = {}
        
        # Oracle health tracking
        self._oracles = [
            ContextOracle("events", self._event_client, timeout_seconds),
            ContextOracle("civic", self._civic_client, timeout_seconds),
        ]
    
    def _get_cache_key(self, latitude: float, longitude: float) -> str:
        """Generate cache key for a location (rounded to ~100m precision)."""
        lat_rounded = round(latitude, 3)
        lon_rounded = round(longitude, 3)
        return f"{lat_rounded}:{lon_rounded}"
    
    def _get_cached_result(
        self, latitude: float, longitude: float
    ) -> Optional[ContextResult]:
        """Get cached result if still valid."""
        key = self._get_cache_key(latitude, longitude)
        
        if key in self._cache:
            result, timestamp = self._cache[key]
            if (datetime.utcnow() - timestamp).total_seconds() < self._cache_ttl:
                logger.debug(f"Cache hit for {key}")
                return result
            else:
                del self._cache[key]
        
        return None
    
    def _cache_result(
        self, latitude: float, longitude: float, result: ContextResult
    ) -> None:
        """Cache a result."""
        key = self._get_cache_key(latitude, longitude)
        self._cache[key] = (result, datetime.utcnow())
    
    async def verify_context(
        self,
        latitude: float,
        longitude: float,
        density_score: float = 0.0,
        radius_km: float = 2.0,
    ) -> ContextResult:
        """
        Verify context for a location to explain high demand.
        
        This is the main entry point for the oracle service.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude  
            density_score: Current density at location (for context)
            radius_km: Search radius for events/alerts
            
        Returns:
            ContextResult with verdict and supporting data
        """
        # Check cache first
        cached = self._get_cached_result(latitude, longitude)
        if cached:
            return cached
        
        # Query all sources in parallel
        events_task = self._query_events(latitude, longitude, radius_km)
        alerts_task = self._query_alerts(latitude, longitude, radius_km)
        
        events, alerts = await asyncio.gather(
            events_task, alerts_task, return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(events, Exception):
            logger.warning(f"Event query failed: {events}")
            events = []
        if isinstance(alerts, Exception):
            logger.warning(f"Alert query failed: {alerts}")
            alerts = []
        
        # Synthesize results
        result = self._synthesize_context(events, alerts, density_score)
        
        # Cache and return
        self._cache_result(latitude, longitude, result)
        
        logger.info(
            f"Context verification at ({latitude}, {longitude}): "
            f"verdict={result.verdict.value}, confidence={result.confidence:.2f}"
        )
        
        return result
    
    async def _query_events(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
    ) -> List[Event]:
        """Query event API with timeout."""
        try:
            return await asyncio.wait_for(
                self._event_client.get_events_near_location(
                    latitude, longitude, radius_km
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Event API timeout")
            return []
        except Exception as e:
            logger.error(f"Event API error: {e}")
            return []
    
    async def _query_alerts(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
    ) -> List[CivicAlert]:
        """Query civic data with timeout."""
        try:
            return await asyncio.wait_for(
                self._civic_client.get_alerts_near_location(
                    latitude, longitude, radius_km
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Civic data API timeout")
            return []
        except Exception as e:
            logger.error(f"Civic data API error: {e}")
            return []
    
    def _synthesize_context(
        self,
        events: List[Event],
        alerts: List[CivicAlert],
        density_score: float,
    ) -> ContextResult:
        """
        Synthesize context from all data sources.
        
        Priority Order:
            1. DANGEROUS: Any civil unrest or high-severity protest
            2. SAFE_HIGH_DEMAND: Active ticketed event
            3. SAFE_INFRASTRUCTURE: Road work, permitted event
            4. SUSPICIOUS: High density with no explanation
            5. NORMAL: Low density, no concerns
        """
        # First, check for danger signals (highest priority)
        dangerous_alerts = [
            a for a in alerts 
            if a.is_dangerous and a.is_active
        ]
        
        if dangerous_alerts:
            # Sort by severity (highest first)
            dangerous_alerts.sort(key=lambda a: a.severity, reverse=True)
            primary_alert = dangerous_alerts[0]
            
            return ContextResult(
                verdict=ContextVerdict.DANGEROUS,
                context_type=ContextType.CIVIL_UNREST,
                confidence=min(primary_alert.severity / 5.0, 1.0),
                events=events,
                alerts=alerts,
                primary_alert=primary_alert,
                explanation=(
                    f"DANGER: {primary_alert.title} detected at location. "
                    f"Severity: {primary_alert.severity}/5. "
                    f"Recommend immediate geofencing."
                ),
                recommendations=[
                    "GEOFENCE: Create exclusion zone",
                    "REROUTE: Divert all vehicles",
                    "ALERT: Notify operations center",
                ],
            )
        
        # Second, check for legitimate events
        active_events = [e for e in events if e.is_active()]
        
        if active_events:
            # Prefer larger events (by expected attendance)
            active_events.sort(
                key=lambda e: e.expected_attendance or 0, 
                reverse=True
            )
            primary_event = active_events[0]
            
            # Calculate confidence based on event size
            attendance = primary_event.expected_attendance or 1000
            confidence = min(attendance / 10000, 1.0)  # 10k+ = full confidence
            
            return ContextResult(
                verdict=ContextVerdict.SAFE_HIGH_DEMAND,
                context_type=ContextType.SCHEDULED_EVENT,
                confidence=confidence,
                events=events,
                alerts=alerts,
                primary_event=primary_event,
                explanation=(
                    f"Legitimate event detected: {primary_event.name} "
                    f"at {primary_event.venue_name}. "
                    f"Expected attendance: {attendance:,}. "
                    f"High demand is expected and normal."
                ),
                recommendations=[
                    "ALLOW: Proceed with normal dispatch",
                    "MONITOR: Track demand vs event timeline",
                    "SURGE: Consider surge pricing (optional)",
                ],
            )
        
        # Third, check for safe infrastructure/permitted events
        safe_crowd_alerts = [
            a for a in alerts 
            if a.is_safe_crowd and a.is_active
        ]
        
        if safe_crowd_alerts:
            primary_alert = safe_crowd_alerts[0]
            
            return ContextResult(
                verdict=ContextVerdict.SAFE_INFRASTRUCTURE,
                context_type=ContextType.SCHEDULED_EVENT,
                confidence=0.8,
                events=events,
                alerts=alerts,
                primary_alert=primary_alert,
                explanation=(
                    f"Permitted event: {primary_alert.title}. "
                    f"High demand is expected and authorized."
                ),
                recommendations=[
                    "ALLOW: Proceed with adjusted routing",
                    "NOTE: Some road closures may apply",
                ],
            )
        
        # Fourth, if we have high density but no context -> suspicious
        if density_score > 100:  # Arbitrary threshold
            return ContextResult(
                verdict=ContextVerdict.SUSPICIOUS,
                context_type=ContextType.NONE,
                confidence=0.0,  # No context found
                events=events,
                alerts=alerts,
                explanation=(
                    f"High density detected ({density_score:.0f} req/km²) "
                    f"with no contextual explanation. "
                    f"No events, protests, or incidents found. "
                    f"Recommend elevated caution."
                ),
                recommendations=[
                    "CAUTION: Apply liability gate",
                    "INVESTIGATE: Manual review recommended",
                    "MONITOR: Watch for pattern development",
                ],
            )
        
        # Default: Normal situation
        return ContextResult(
            verdict=ContextVerdict.NORMAL,
            context_type=ContextType.NONE,
            confidence=0.0,
            events=events,
            alerts=alerts,
            explanation="Normal operating conditions. No special context.",
            recommendations=["ALLOW: Standard dispatch procedures"],
        )
    
    async def check_oracle_health(self) -> Dict[str, bool]:
        """
        Check health of all oracle sources.
        
        Returns dict of {oracle_name: is_healthy}
        """
        results = {}
        
        for oracle in self._oracles:
            is_healthy = await oracle.check_health()
            results[oracle.name] = is_healthy
        
        return results
    
    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()
        logger.info("Context cache cleared")
