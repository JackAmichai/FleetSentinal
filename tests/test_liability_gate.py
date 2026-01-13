"""
FleetSentinel Test - Liability Gate

Tests for the Layer 3 Liability Enforcement Middleware.
Validates:
- $500 hold authorization for medium-risk requests
- Proper Stripe PaymentIntent flow
- Hold expiration and capture logic
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Optional

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import (
    RideRequest,
    GeoLocation,
    ThreatAssessment,
    ThreatLevel,
    LiabilityHold,
)
from gates.liability import (
    LiabilityEnforcementMiddleware,
    PaymentGateway,
    MockStripeGateway,
    LiabilityConfig,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def gateway():
    """Create a mock Stripe gateway."""
    return MockStripeGateway(api_key="sk_test_mock")


@pytest.fixture
def middleware(gateway):
    """Create a liability middleware with mock gateway."""
    config = LiabilityConfig(
        hold_amount_cents=50000,  # $500
        min_confidence_for_hold=0.4,
        max_confidence_for_hold=0.7,
        hold_expiration_minutes=60,
    )
    return LiabilityEnforcementMiddleware(
        payment_gateway=gateway,
        config=config,
    )


@pytest.fixture
def sample_request():
    """Create a sample ride request."""
    return RideRequest(
        request_id=f"req_{uuid.uuid4().hex[:12]}",
        user_id="user_123",
        timestamp=datetime.utcnow(),
        pickup_location=GeoLocation(latitude=37.7749, longitude=-122.4194),
        user_ip_hash="ip_hash_abc",
        device_fingerprint="device_xyz",
        account_age_days=30,
        historical_rides=10,
        payment_method_verified=True,
        payment_method_id="pm_card_visa",
    )


def create_assessment(confidence: float) -> ThreatAssessment:
    """Create a threat assessment with given confidence."""
    if confidence < 0.3:
        level = ThreatLevel.NONE
    elif confidence < 0.5:
        level = ThreatLevel.LOW
    elif confidence < 0.7:
        level = ThreatLevel.MEDIUM
    elif confidence < 0.9:
        level = ThreatLevel.HIGH
    else:
        level = ThreatLevel.CRITICAL
    
    return ThreatAssessment(
        confidence_score=confidence,
        threat_level=level,
        contributing_factors=["test"],
        explanation=f"Test assessment with confidence {confidence}",
    )


# =============================================================================
# UNIT TESTS
# =============================================================================

class TestMockStripeGateway:
    """Tests for the mock Stripe gateway."""
    
    @pytest.mark.asyncio
    async def test_create_hold_success(self, gateway):
        """Test successful hold creation."""
        hold = await gateway.create_hold(
            customer_id="cus_123",
            amount_cents=50000,
            payment_method_id="pm_card_visa",
            metadata={"request_id": "req_123"},
        )
        
        assert hold is not None
        assert hold.hold_id.startswith("pi_mock_")
        assert hold.amount_cents == 50000
        assert hold.status == "requires_capture"
        assert hold.customer_id == "cus_123"
    
    @pytest.mark.asyncio
    async def test_capture_hold(self, gateway):
        """Test capturing a hold."""
        # Create hold
        hold = await gateway.create_hold(
            customer_id="cus_123",
            amount_cents=50000,
            payment_method_id="pm_card_visa",
        )
        
        # Capture it
        captured = await gateway.capture_hold(hold.hold_id)
        
        assert captured is True
    
    @pytest.mark.asyncio
    async def test_release_hold(self, gateway):
        """Test releasing a hold."""
        # Create hold
        hold = await gateway.create_hold(
            customer_id="cus_123",
            amount_cents=50000,
            payment_method_id="pm_card_visa",
        )
        
        # Release it
        released = await gateway.release_hold(hold.hold_id)
        
        assert released is True
    
    @pytest.mark.asyncio
    async def test_capture_nonexistent_hold(self, gateway):
        """Test capturing a hold that doesn't exist."""
        captured = await gateway.capture_hold("pi_nonexistent_123")
        
        assert captured is False
    
    @pytest.mark.asyncio
    async def test_simulate_card_decline(self):
        """Test simulating a declined card."""
        gateway = MockStripeGateway(
            api_key="sk_test_mock",
            simulate_decline=True,
        )
        
        hold = await gateway.create_hold(
            customer_id="cus_123",
            amount_cents=50000,
            payment_method_id="pm_card_visa",
        )
        
        assert hold is None  # Declined


class TestLiabilityMiddleware:
    """Tests for the liability enforcement middleware."""
    
    @pytest.mark.asyncio
    async def test_no_hold_for_low_confidence(self, middleware, sample_request):
        """Test that low confidence requests don't get holds."""
        assessment = create_assessment(confidence=0.2)  # Below 0.4
        
        result = await middleware.evaluate(sample_request, assessment)
        
        assert result.requires_hold is False
        assert result.hold is None
        assert result.decision == "PROCEED"
    
    @pytest.mark.asyncio
    async def test_hold_required_for_medium_confidence(self, middleware, sample_request):
        """Test that medium confidence (0.4-0.7) requires hold."""
        assessment = create_assessment(confidence=0.55)  # Between 0.4 and 0.7
        
        result = await middleware.evaluate(sample_request, assessment)
        
        assert result.requires_hold is True
        assert result.hold is not None
        assert result.hold.amount_cents == 50000  # $500
        assert result.decision == "PROCEED_WITH_HOLD"
    
    @pytest.mark.asyncio
    async def test_block_for_high_confidence(self, middleware, sample_request):
        """Test that high confidence (>0.7) results in block recommendation."""
        assessment = create_assessment(confidence=0.85)  # Above 0.7
        
        result = await middleware.evaluate(sample_request, assessment)
        
        # High confidence should recommend blocking, not holding
        assert result.requires_hold is False
        assert result.decision == "RECOMMEND_BLOCK"
    
    @pytest.mark.asyncio
    async def test_hold_without_payment_method(self, middleware):
        """Test that missing payment method fails gracefully."""
        request = RideRequest(
            request_id="req_no_payment",
            user_id="user_456",
            timestamp=datetime.utcnow(),
            pickup_location=GeoLocation(latitude=37.7749, longitude=-122.4194),
            user_ip_hash="ip_hash_def",
            device_fingerprint="device_abc",
            account_age_days=5,
            historical_rides=0,
            payment_method_verified=False,
            payment_method_id=None,  # No payment method
        )
        
        assessment = create_assessment(confidence=0.55)
        
        result = await middleware.evaluate(request, assessment)
        
        # Should recommend blocking if can't place hold
        assert result.requires_hold is True
        assert result.hold is None  # Failed to create
        assert result.decision == "RECOMMEND_BLOCK"  # Can't proceed without hold
    
    @pytest.mark.asyncio
    async def test_boundary_confidence_lower(self, middleware, sample_request):
        """Test behavior at lower boundary (0.4)."""
        assessment = create_assessment(confidence=0.4)
        
        result = await middleware.evaluate(sample_request, assessment)
        
        # At boundary, should require hold
        assert result.requires_hold is True
    
    @pytest.mark.asyncio
    async def test_boundary_confidence_upper(self, middleware, sample_request):
        """Test behavior at upper boundary (0.7)."""
        assessment = create_assessment(confidence=0.7)
        
        result = await middleware.evaluate(sample_request, assessment)
        
        # At upper boundary, should still allow hold
        assert result.requires_hold is True


class TestHoldLifecycle:
    """Tests for the full hold lifecycle."""
    
    @pytest.mark.asyncio
    async def test_full_hold_cycle_with_capture(self, middleware, sample_request, gateway):
        """Test creating and capturing a hold."""
        assessment = create_assessment(confidence=0.5)
        
        # Create hold
        result = await middleware.evaluate(sample_request, assessment)
        assert result.hold is not None
        
        hold_id = result.hold.hold_id
        
        # Simulate malicious behavior detected - capture the hold
        captured = await gateway.capture_hold(hold_id)
        assert captured is True
    
    @pytest.mark.asyncio
    async def test_full_hold_cycle_with_release(self, middleware, sample_request, gateway):
        """Test creating and releasing a hold (normal completion)."""
        assessment = create_assessment(confidence=0.5)
        
        # Create hold
        result = await middleware.evaluate(sample_request, assessment)
        assert result.hold is not None
        
        hold_id = result.hold.hold_id
        
        # Ride completed normally - release the hold
        released = await gateway.release_hold(hold_id)
        assert released is True
    
    @pytest.mark.asyncio
    async def test_hold_metadata(self, middleware, sample_request):
        """Test that hold contains proper metadata."""
        assessment = create_assessment(confidence=0.5)
        
        result = await middleware.evaluate(sample_request, assessment)
        
        assert result.hold is not None
        assert result.hold.metadata is not None
        assert "request_id" in result.hold.metadata
        assert "confidence_score" in result.hold.metadata


class TestLiabilityConfig:
    """Tests for liability configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LiabilityConfig()
        
        assert config.hold_amount_cents == 50000  # $500
        assert config.min_confidence_for_hold == 0.4
        assert config.max_confidence_for_hold == 0.7
        assert config.hold_expiration_minutes == 60
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = LiabilityConfig(
            hold_amount_cents=100000,  # $1000
            min_confidence_for_hold=0.3,
            max_confidence_for_hold=0.8,
            hold_expiration_minutes=120,
        )
        
        assert config.hold_amount_cents == 100000
        assert config.min_confidence_for_hold == 0.3
        assert config.max_confidence_for_hold == 0.8
        assert config.hold_expiration_minutes == 120
    
    @pytest.mark.asyncio
    async def test_custom_hold_amount(self):
        """Test middleware with custom hold amount."""
        gateway = MockStripeGateway(api_key="sk_test_mock")
        config = LiabilityConfig(hold_amount_cents=100000)  # $1000
        
        middleware = LiabilityEnforcementMiddleware(
            payment_gateway=gateway,
            config=config,
        )
        
        request = RideRequest(
            request_id="req_custom_hold",
            user_id="user_789",
            timestamp=datetime.utcnow(),
            pickup_location=GeoLocation(latitude=37.7749, longitude=-122.4194),
            user_ip_hash="ip_hash_ghi",
            device_fingerprint="device_jkl",
            account_age_days=5,
            historical_rides=0,
            payment_method_verified=True,
            payment_method_id="pm_card_visa",
        )
        
        assessment = create_assessment(confidence=0.5)
        result = await middleware.evaluate(request, assessment)
        
        assert result.hold is not None
        assert result.hold.amount_cents == 100000  # $1000


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
