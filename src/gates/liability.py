"""
FleetSentinel Liability Enforcement Middleware

Layer 3: The "Skin in the Game" Economic Deterrence Gate

This module implements economic deterrence for "grey zone" situations
where we suspect but cannot confirm an attack. Instead of blocking
the request outright, we require a substantial security deposit.

Theory of Operation:
    - Malicious actors are rarely willing to risk real money
    - Legitimate desperate users (medical emergency, etc.) will accept
    - The $500 hold creates economic friction for coordinated attacks
    - Funds are fully refunded if ride completes normally

Logic:
    IF confidence_score is between 0.4 and 0.7:
        -> Do NOT ban the user
        -> Force a $500 Pre-Authorization Hold
        -> If user accepts -> proceed with caution
        -> If user declines -> request blocked (self-selection)

Integration:
    Uses Stripe Payment Intents API for holds:
        1. Create PaymentIntent with capture_method='manual'
        2. Confirm PaymentIntent (creates hold on card)
        3. Either capture (if vehicle damaged) or cancel (normal completion)
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, field
import logging

import httpx


logger = logging.getLogger(__name__)


class HoldStatus(str, Enum):
    """Status of a liability hold."""
    PENDING = "PENDING"          # Hold requested, awaiting user action
    AUTHORIZED = "AUTHORIZED"    # Hold confirmed on card
    CAPTURED = "CAPTURED"        # Funds captured (damage occurred)
    RELEASED = "RELEASED"        # Hold released (normal completion)
    FAILED = "FAILED"            # User declined or card failed
    EXPIRED = "EXPIRED"          # Hold expired without action


@dataclass
class LiabilityHold:
    """
    Represents a liability security hold.
    
    This tracks the lifecycle of a hold from creation to resolution.
    """
    hold_id: str
    request_id: str
    user_id: str
    amount_cents: int
    currency: str = "usd"
    status: HoldStatus = HoldStatus.PENDING
    
    # Payment provider references
    payment_intent_id: Optional[str] = None
    payment_method_id: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    authorized_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Metadata
    reason: str = "FleetSentinel Security Deposit"
    confidence_score: float = 0.0
    threat_level: str = ""
    
    @property
    def is_active(self) -> bool:
        """Check if hold is currently active."""
        return self.status == HoldStatus.AUTHORIZED
    
    @property
    def is_resolved(self) -> bool:
        """Check if hold has been resolved."""
        return self.status in (
            HoldStatus.CAPTURED,
            HoldStatus.RELEASED,
            HoldStatus.FAILED,
            HoldStatus.EXPIRED,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hold_id": self.hold_id,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "amount_cents": self.amount_cents,
            "currency": self.currency,
            "status": self.status.value,
            "payment_intent_id": self.payment_intent_id,
            "created_at": self.created_at.isoformat(),
            "authorized_at": self.authorized_at.isoformat() if self.authorized_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "reason": self.reason,
            "confidence_score": self.confidence_score,
        }


@dataclass
class LiabilityResult:
    """
    Result of a liability gate check.
    
    This is returned after attempting to create and authorize a hold.
    """
    success: bool
    hold: Optional[LiabilityHold] = None
    requires_user_action: bool = False
    client_secret: Optional[str] = None  # For Stripe.js confirmation
    error_message: Optional[str] = None
    
    @property
    def can_proceed(self) -> bool:
        """Check if request can proceed after liability check."""
        return self.success and self.hold is not None and self.hold.is_active


class PaymentGateway(ABC):
    """
    Abstract interface for payment processing.
    
    Implementations can be:
        - Stripe (production)
        - Mock (testing)
        - Square, Adyen, etc. (future)
    """
    
    @abstractmethod
    async def create_hold(
        self,
        amount_cents: int,
        currency: str,
        user_id: str,
        payment_method_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a pre-authorization hold on a payment method.
        
        Args:
            amount_cents: Amount to hold in cents
            currency: 3-letter currency code (e.g., 'usd')
            user_id: User's account ID
            payment_method_id: Stored payment method ID
            metadata: Additional metadata to store
            
        Returns:
            Dict with payment_intent_id, client_secret, status
        """
        pass
    
    @abstractmethod
    async def capture_hold(
        self,
        payment_intent_id: str,
        amount_cents: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Capture (charge) a previously authorized hold.
        
        Only used when vehicle damage is confirmed.
        
        Args:
            payment_intent_id: ID of the hold to capture
            amount_cents: Amount to capture (None = full amount)
            
        Returns:
            Dict with status, amount_captured
        """
        pass
    
    @abstractmethod
    async def release_hold(
        self,
        payment_intent_id: str,
    ) -> Dict[str, Any]:
        """
        Release (cancel) a previously authorized hold.
        
        Used when ride completes normally.
        
        Args:
            payment_intent_id: ID of the hold to release
            
        Returns:
            Dict with status
        """
        pass
    
    @abstractmethod
    async def get_hold_status(
        self,
        payment_intent_id: str,
    ) -> Dict[str, Any]:
        """
        Get current status of a hold.
        
        Args:
            payment_intent_id: ID of the hold to check
            
        Returns:
            Dict with status, amount, created timestamp
        """
        pass


class MockStripeGateway(PaymentGateway):
    """
    Mock Stripe implementation for testing.
    
    This simulates Stripe's PaymentIntent API behavior:
        - create_hold -> Creates PaymentIntent with manual capture
        - capture_hold -> Captures the PaymentIntent
        - release_hold -> Cancels the PaymentIntent
    
    For testing, you can configure failure scenarios:
        - set_decline_all(True) -> All holds fail
        - add_declined_user(user_id) -> Specific user fails
    """
    
    def __init__(self):
        """Initialize mock gateway."""
        self._holds: Dict[str, Dict[str, Any]] = {}
        self._decline_all = False
        self._declined_users: set = set()
        self._declined_methods: set = set()
    
    def set_decline_all(self, decline: bool) -> None:
        """Configure mock to decline all holds."""
        self._decline_all = decline
    
    def add_declined_user(self, user_id: str) -> None:
        """Add a user to the decline list."""
        self._declined_users.add(user_id)
    
    def add_declined_method(self, payment_method_id: str) -> None:
        """Add a payment method to the decline list."""
        self._declined_methods.add(payment_method_id)
    
    def reset(self) -> None:
        """Reset all mock state."""
        self._holds.clear()
        self._decline_all = False
        self._declined_users.clear()
        self._declined_methods.clear()
    
    async def create_hold(
        self,
        amount_cents: int,
        currency: str,
        user_id: str,
        payment_method_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a mock hold."""
        # Simulate API latency
        await asyncio.sleep(0.1)
        
        # Check for decline conditions
        if self._decline_all:
            return {
                "success": False,
                "error": "card_declined",
                "message": "Card was declined (mock: decline_all=True)",
            }
        
        if user_id in self._declined_users:
            return {
                "success": False,
                "error": "card_declined",
                "message": f"Card was declined (mock: user {user_id} in decline list)",
            }
        
        if payment_method_id in self._declined_methods:
            return {
                "success": False,
                "error": "card_declined",
                "message": "Payment method declined",
            }
        
        # Create successful hold
        payment_intent_id = f"pi_mock_{uuid.uuid4().hex[:16]}"
        client_secret = f"{payment_intent_id}_secret_{uuid.uuid4().hex[:24]}"
        
        hold_data = {
            "id": payment_intent_id,
            "object": "payment_intent",
            "amount": amount_cents,
            "currency": currency,
            "capture_method": "manual",
            "status": "requires_capture",  # Authorized
            "client_secret": client_secret,
            "metadata": metadata or {},
            "user_id": user_id,
            "payment_method_id": payment_method_id,
            "created": datetime.utcnow().timestamp(),
            "captured": False,
            "canceled": False,
        }
        
        self._holds[payment_intent_id] = hold_data
        
        logger.info(f"Mock hold created: {payment_intent_id} for ${amount_cents/100:.2f}")
        
        return {
            "success": True,
            "payment_intent_id": payment_intent_id,
            "client_secret": client_secret,
            "status": "requires_capture",
            "amount": amount_cents,
        }
    
    async def capture_hold(
        self,
        payment_intent_id: str,
        amount_cents: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Capture a mock hold."""
        await asyncio.sleep(0.05)
        
        if payment_intent_id not in self._holds:
            return {
                "success": False,
                "error": "not_found",
                "message": f"PaymentIntent {payment_intent_id} not found",
            }
        
        hold = self._holds[payment_intent_id]
        
        if hold["captured"]:
            return {
                "success": False,
                "error": "already_captured",
                "message": "PaymentIntent already captured",
            }
        
        if hold["canceled"]:
            return {
                "success": False,
                "error": "canceled",
                "message": "PaymentIntent was canceled",
            }
        
        capture_amount = amount_cents or hold["amount"]
        hold["captured"] = True
        hold["captured_amount"] = capture_amount
        hold["status"] = "succeeded"
        
        logger.info(f"Mock hold captured: {payment_intent_id} for ${capture_amount/100:.2f}")
        
        return {
            "success": True,
            "status": "succeeded",
            "amount_captured": capture_amount,
        }
    
    async def release_hold(
        self,
        payment_intent_id: str,
    ) -> Dict[str, Any]:
        """Release (cancel) a mock hold."""
        await asyncio.sleep(0.05)
        
        if payment_intent_id not in self._holds:
            return {
                "success": False,
                "error": "not_found",
                "message": f"PaymentIntent {payment_intent_id} not found",
            }
        
        hold = self._holds[payment_intent_id]
        
        if hold["captured"]:
            return {
                "success": False,
                "error": "already_captured",
                "message": "Cannot release captured PaymentIntent",
            }
        
        hold["canceled"] = True
        hold["status"] = "canceled"
        
        logger.info(f"Mock hold released: {payment_intent_id}")
        
        return {
            "success": True,
            "status": "canceled",
        }
    
    async def get_hold_status(
        self,
        payment_intent_id: str,
    ) -> Dict[str, Any]:
        """Get status of a mock hold."""
        await asyncio.sleep(0.02)
        
        if payment_intent_id not in self._holds:
            return {
                "success": False,
                "error": "not_found",
            }
        
        hold = self._holds[payment_intent_id]
        
        return {
            "success": True,
            "payment_intent_id": payment_intent_id,
            "status": hold["status"],
            "amount": hold["amount"],
            "captured": hold["captured"],
            "canceled": hold["canceled"],
        }


class StripeGateway(PaymentGateway):
    """
    Production Stripe Payment Gateway.
    
    Uses Stripe's PaymentIntent API with manual capture for holds.
    
    Requirements:
        - Stripe API key (secret key)
        - Customer with saved payment method
        
    Flow:
        1. Create PaymentIntent with capture_method='manual'
        2. Confirm with payment method -> creates hold
        3. Later: capture (charge) or cancel (release)
    """
    
    BASE_URL = "https://api.stripe.com/v1"
    
    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 10.0,
    ):
        """
        Initialize Stripe gateway.
        
        Args:
            api_key: Stripe secret API key
            timeout_seconds: Request timeout
        """
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=self._timeout,
                auth=(self._api_key, ""),
            )
        return self._client
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def create_hold(
        self,
        amount_cents: int,
        currency: str,
        user_id: str,
        payment_method_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a hold using Stripe PaymentIntent."""
        try:
            client = await self._get_client()
            
            # Create PaymentIntent with manual capture
            data = {
                "amount": amount_cents,
                "currency": currency,
                "capture_method": "manual",
                "confirm": "true",
                "payment_method": payment_method_id,
                "description": "FleetSentinel Security Deposit",
            }
            
            if metadata:
                for key, value in metadata.items():
                    data[f"metadata[{key}]"] = str(value)
            
            response = await client.post("/payment_intents", data=data)
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "success": True,
                "payment_intent_id": result["id"],
                "client_secret": result["client_secret"],
                "status": result["status"],
                "amount": result["amount"],
            }
            
        except httpx.HTTPStatusError as e:
            error_body = e.response.json() if e.response else {}
            error = error_body.get("error", {})
            
            return {
                "success": False,
                "error": error.get("code", "unknown"),
                "message": error.get("message", str(e)),
            }
        except Exception as e:
            logger.error(f"Stripe API error: {e}")
            return {
                "success": False,
                "error": "api_error",
                "message": str(e),
            }
    
    async def capture_hold(
        self,
        payment_intent_id: str,
        amount_cents: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Capture a previously authorized hold."""
        try:
            client = await self._get_client()
            
            data = {}
            if amount_cents:
                data["amount_to_capture"] = amount_cents
            
            response = await client.post(
                f"/payment_intents/{payment_intent_id}/capture",
                data=data,
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "success": True,
                "status": result["status"],
                "amount_captured": result.get("amount_received", result["amount"]),
            }
            
        except Exception as e:
            logger.error(f"Failed to capture hold: {e}")
            return {
                "success": False,
                "error": "capture_failed",
                "message": str(e),
            }
    
    async def release_hold(
        self,
        payment_intent_id: str,
    ) -> Dict[str, Any]:
        """Release (cancel) a hold."""
        try:
            client = await self._get_client()
            
            response = await client.post(
                f"/payment_intents/{payment_intent_id}/cancel"
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "success": True,
                "status": result["status"],
            }
            
        except Exception as e:
            logger.error(f"Failed to release hold: {e}")
            return {
                "success": False,
                "error": "release_failed",
                "message": str(e),
            }
    
    async def get_hold_status(
        self,
        payment_intent_id: str,
    ) -> Dict[str, Any]:
        """Get status of a hold."""
        try:
            client = await self._get_client()
            
            response = await client.get(f"/payment_intents/{payment_intent_id}")
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "success": True,
                "payment_intent_id": result["id"],
                "status": result["status"],
                "amount": result["amount"],
                "captured": result["status"] == "succeeded",
                "canceled": result["status"] == "canceled",
            }
            
        except Exception as e:
            logger.error(f"Failed to get hold status: {e}")
            return {
                "success": False,
                "error": "api_error",
                "message": str(e),
            }


class LiabilityEnforcementMiddleware:
    """
    The Liability Gate Middleware.
    
    This middleware intercepts ride requests in the "grey zone" 
    (confidence 0.4-0.7) and requires a security deposit before proceeding.
    
    Why $500?
        - High enough to deter casual pranksters
        - Low enough that legitimate users in emergencies will accept
        - Standard pre-auth amount that cards can handle
    
    Flow:
        1. Evaluate threat level
        2. If MEDIUM threat (0.4-0.7):
           a. Check user's payment method
           b. Create $500 pre-authorization hold
           c. If successful -> proceed with ride
           d. If failed -> block request
        3. Store hold for later resolution
    
    Resolution:
        - Normal completion -> release hold
        - Vehicle damage -> capture hold
        - Hold expires after 24 hours
    """
    
    def __init__(
        self,
        gateway: Optional[PaymentGateway] = None,
        hold_amount_cents: int = 50000,  # $500.00
        currency: str = "usd",
        hold_duration_hours: int = 24,
        confidence_threshold_low: float = 0.4,
        confidence_threshold_high: float = 0.7,
    ):
        """
        Initialize the Liability Gate.
        
        Args:
            gateway: Payment gateway (defaults to mock)
            hold_amount_cents: Amount to hold (default $500)
            currency: Currency code
            hold_duration_hours: How long holds last
            confidence_threshold_low: Lower bound for gate (default 0.4)
            confidence_threshold_high: Upper bound for gate (default 0.7)
        """
        self._gateway = gateway or MockStripeGateway()
        self._hold_amount = hold_amount_cents
        self._currency = currency
        self._hold_duration = timedelta(hours=hold_duration_hours)
        self._threshold_low = confidence_threshold_low
        self._threshold_high = confidence_threshold_high
        
        # Active holds storage
        self._active_holds: Dict[str, LiabilityHold] = {}
    
    def requires_liability(self, confidence_score: float) -> bool:
        """
        Check if a confidence score requires liability enforcement.
        
        Args:
            confidence_score: Threat confidence (0.0-1.0)
            
        Returns:
            True if liability hold is required
        """
        return self._threshold_low <= confidence_score <= self._threshold_high
    
    async def enforce(
        self,
        request_id: str,
        user_id: str,
        payment_method_id: str,
        confidence_score: float,
        threat_level: str = "",
    ) -> LiabilityResult:
        """
        Enforce liability gate for a ride request.
        
        This creates a pre-authorization hold on the user's payment method.
        
        Args:
            request_id: The ride request ID
            user_id: User's account ID
            payment_method_id: User's stored payment method
            confidence_score: Current threat confidence
            threat_level: Categorical threat level (for logging)
            
        Returns:
            LiabilityResult indicating success/failure
        """
        logger.info(
            f"Enforcing liability gate for request {request_id}: "
            f"confidence={confidence_score:.2f}, amount=${self._hold_amount/100:.2f}"
        )
        
        # Create hold record
        hold = LiabilityHold(
            hold_id=f"hold_{uuid.uuid4().hex[:16]}",
            request_id=request_id,
            user_id=user_id,
            amount_cents=self._hold_amount,
            currency=self._currency,
            status=HoldStatus.PENDING,
            expires_at=datetime.utcnow() + self._hold_duration,
            confidence_score=confidence_score,
            threat_level=threat_level,
        )
        
        # Attempt to create hold with payment gateway
        result = await self._gateway.create_hold(
            amount_cents=self._hold_amount,
            currency=self._currency,
            user_id=user_id,
            payment_method_id=payment_method_id,
            metadata={
                "request_id": request_id,
                "hold_id": hold.hold_id,
                "confidence_score": str(confidence_score),
                "threat_level": threat_level,
                "service": "FleetSentinel",
            },
        )
        
        if result.get("success"):
            # Hold created successfully
            hold.status = HoldStatus.AUTHORIZED
            hold.payment_intent_id = result["payment_intent_id"]
            hold.authorized_at = datetime.utcnow()
            
            self._active_holds[hold.hold_id] = hold
            
            logger.info(f"Liability hold authorized: {hold.hold_id}")
            
            return LiabilityResult(
                success=True,
                hold=hold,
                requires_user_action=False,
                client_secret=result.get("client_secret"),
            )
        else:
            # Hold failed (card declined, etc.)
            hold.status = HoldStatus.FAILED
            
            logger.warning(
                f"Liability hold failed for request {request_id}: "
                f"{result.get('error')} - {result.get('message')}"
            )
            
            return LiabilityResult(
                success=False,
                hold=hold,
                requires_user_action=False,
                error_message=result.get("message", "Payment authorization failed"),
            )
    
    async def release(self, hold_id: str) -> bool:
        """
        Release a liability hold (normal ride completion).
        
        Args:
            hold_id: ID of the hold to release
            
        Returns:
            True if released successfully
        """
        if hold_id not in self._active_holds:
            logger.warning(f"Hold not found: {hold_id}")
            return False
        
        hold = self._active_holds[hold_id]
        
        if hold.status != HoldStatus.AUTHORIZED:
            logger.warning(f"Cannot release hold in status: {hold.status}")
            return False
        
        if not hold.payment_intent_id:
            logger.warning(f"Hold {hold_id} has no payment_intent_id")
            return False
        
        result = await self._gateway.release_hold(hold.payment_intent_id)
        
        if result.get("success"):
            hold.status = HoldStatus.RELEASED
            hold.resolved_at = datetime.utcnow()
            
            logger.info(f"Liability hold released: {hold_id}")
            return True
        else:
            logger.error(f"Failed to release hold {hold_id}: {result}")
            return False
    
    async def capture(
        self, hold_id: str, amount_cents: Optional[int] = None
    ) -> bool:
        """
        Capture a liability hold (vehicle damage confirmed).
        
        Args:
            hold_id: ID of the hold to capture
            amount_cents: Amount to capture (None = full amount)
            
        Returns:
            True if captured successfully
        """
        if hold_id not in self._active_holds:
            logger.warning(f"Hold not found: {hold_id}")
            return False
        
        hold = self._active_holds[hold_id]
        
        if hold.status != HoldStatus.AUTHORIZED:
            logger.warning(f"Cannot capture hold in status: {hold.status}")
            return False
        
        if not hold.payment_intent_id:
            return False
        
        result = await self._gateway.capture_hold(
            hold.payment_intent_id,
            amount_cents=amount_cents,
        )
        
        if result.get("success"):
            hold.status = HoldStatus.CAPTURED
            hold.resolved_at = datetime.utcnow()
            
            logger.info(
                f"Liability hold captured: {hold_id} "
                f"for ${(amount_cents or hold.amount_cents)/100:.2f}"
            )
            return True
        else:
            logger.error(f"Failed to capture hold {hold_id}: {result}")
            return False
    
    def get_hold(self, hold_id: str) -> Optional[LiabilityHold]:
        """Get a hold by ID."""
        return self._active_holds.get(hold_id)
    
    def get_hold_by_request(self, request_id: str) -> Optional[LiabilityHold]:
        """Get hold by request ID."""
        for hold in self._active_holds.values():
            if hold.request_id == request_id:
                return hold
        return None
    
    def get_active_holds(self) -> List[LiabilityHold]:
        """Get all active (authorized) holds."""
        return [
            h for h in self._active_holds.values()
            if h.status == HoldStatus.AUTHORIZED
        ]
    
    async def cleanup_expired_holds(self) -> int:
        """
        Clean up expired holds.
        
        Returns number of holds cleaned up.
        """
        now = datetime.utcnow()
        expired = []
        
        for hold_id, hold in self._active_holds.items():
            if hold.expires_at and hold.expires_at < now:
                if hold.status == HoldStatus.AUTHORIZED:
                    # Release the hold
                    await self.release(hold_id)
                hold.status = HoldStatus.EXPIRED
                expired.append(hold_id)
        
        for hold_id in expired:
            del self._active_holds[hold_id]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired holds")
        
        return len(expired)
