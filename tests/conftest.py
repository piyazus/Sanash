"""
Shared pytest fixtures for Sanash test suite.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_detection():
    """Typical detection result payload from Jetson edge node."""
    return {
        "count": 35,
        "occupancy_ratio": 0.583,
        "bus_id": "bus_001",
        "route_id": "36A",
        "timestamp": "2025-03-15T08:30:00",
        "camera_id": "cam_front",
    }


@pytest.fixture
def bus_capacity():
    """Standard Almaty bus capacity."""
    return 60


@pytest.fixture
def mock_db_session():
    """Mock SQLAlchemy database session."""
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = None
    session.query.return_value.all.return_value = []
    session.add.return_value = None
    session.commit.return_value = None
    return session


@pytest.fixture
def mock_redis():
    """Mock Redis cache client."""
    client = MagicMock()
    client.get.return_value = None
    client.set.return_value = True
    client.delete.return_value = 1
    client.exists.return_value = False
    return client
