from datetime import datetime, timedelta

import pytest
from fastapi import BackgroundTasks

from app.main import emergency_override
from app.models.traffic_models import EmergencyAlert, EmergencyType, LaneDirection, TrafficSignalState
from app.services.adaptive_traffic_manager import AdaptiveTrafficManager


def make_alert(**overrides) -> EmergencyAlert:
    payload = {
        "alert_id": "emergency-test",
        "emergency_type": EmergencyType.AMBULANCE,
        "detected_lane": LaneDirection.NORTH,
        "vehicle_location": {"x": 0.5, "y": 0.2},
        "priority_level": 5,
        "override_duration": 60,
    }
    payload.update(overrides)
    return EmergencyAlert(**payload)


@pytest.mark.asyncio
async def test_emergency_endpoint_accepts_validated_alert():
    manager = AdaptiveTrafficManager()
    alert = make_alert()

    response = await emergency_override(alert, BackgroundTasks(), manager)

    assert response["alert_id"] == alert.alert_id
    assert manager.intersection_status.emergency_mode_active is True
    assert manager.intersection_status.traffic_signals[LaneDirection.NORTH].current_state == TrafficSignalState.GREEN
    await manager.cleanup()


@pytest.mark.asyncio
async def test_expired_emergency_alert_is_resolved_without_stopping_manager():
    manager = AdaptiveTrafficManager()
    alert = make_alert(created_at=datetime.utcnow() - timedelta(seconds=61))
    manager.emergency_alerts[alert.alert_id] = alert
    manager.intersection_status.emergency_mode_active = True

    await manager._check_emergency_overrides()

    assert alert.alert_id not in manager.emergency_alerts
    assert manager.intersection_status.emergency_mode_active is False
    await manager.cleanup()
