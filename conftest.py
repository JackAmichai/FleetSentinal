"""
FleetSentinel - pytest Configuration

Shared fixtures and configuration for all tests.
"""

import pytest
import asyncio
import sys
import os
from typing import Generator, AsyncGenerator

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires external services)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    # Skip integration tests by default unless explicitly requested
    if not config.getoption("--run-integration", default=False):
        skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests that require external services"
    )


# =============================================================================
# ASYNC FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_redis():
    """Create a mock Redis client for testing."""
    from tests.test_scenario_dead_end_attack import MockRedis
    return MockRedis()


@pytest.fixture
def sample_location():
    """Create a sample GeoLocation for testing."""
    from src.models import GeoLocation
    return GeoLocation(
        latitude=37.7749,
        longitude=-122.4194,
    )


@pytest.fixture
def sample_dead_end_location():
    """Create a dead-end location for testing."""
    from src.models import GeoLocation
    return GeoLocation(
        latitude=37.7850,
        longitude=-122.4050,
    )


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def test_config():
    """Create a test configuration."""
    from src.config import FleetSentinelConfig, Environment
    
    return FleetSentinelConfig(
        environment=Environment.DEVELOPMENT,
    )


@pytest.fixture(autouse=True)
def reset_config_fixture():
    """Reset global configuration before each test."""
    from src.config import reset_config
    reset_config()
    yield
    reset_config()
