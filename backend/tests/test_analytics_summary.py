import pytest
from app.models.ai_system import AISystem, RiskLevel
from app.models.user import User


def test_analytics_summary_counts(client, db_session):
    # Register and login a test user
    email = "analytics_user@example.com"
    password = "TestPass123!"
    client.post("/api/v1/auth/register", json={"email": email, "password": password, "full_name": "Analytics User"})
    resp = client.post("/api/v1/auth/login", data={"username": email, "password": password})
    token = resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    user = db_session.query(User).filter(User.email == email).first()

    # Create AI systems with different risk levels
    systems = [
        AISystem(owner_id=user.id, name="sys1", risk_level=RiskLevel.MINIMAL),
        AISystem(owner_id=user.id, name="sys2", risk_level=RiskLevel.MINIMAL),
        AISystem(owner_id=user.id, name="sys3", risk_level=RiskLevel.LIMITED),
        AISystem(owner_id=user.id, name="sys4", risk_level=RiskLevel.UNACCEPTABLE),
    ]

    db_session.add_all(systems)
    db_session.commit()

    response = client.get("/api/v1/analytics/summary", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "counts" in data
    counts = data["counts"]
    assert counts["minimal"] == 2
    assert counts["limited"] == 1
    assert counts["high"] == 0
    assert counts["unacceptable"] == 1


def test_analytics_summary_with_unclassified_systems(client, db_session):
    """Systems with risk_level=None are not counted but do not cause KeyError."""
    email = "unclassified@example.com"
    password = "TestPass123!"
    client.post("/api/v1/auth/register", json={"email": email, "password": password, "full_name": "Test"})
    resp = client.post("/api/v1/auth/login", data={"username": email, "password": password})
    token = resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    user = db_session.query(User).filter(User.email == email).first()

    # Create system with risk_level=None (unclassified)
    unclassified_system = AISystem(owner_id=user.id, name="unclassified", risk_level=None)
    classified_system = AISystem(owner_id=user.id, name="classified", risk_level=RiskLevel.MINIMAL)
    db_session.add_all([unclassified_system, classified_system])
    db_session.commit()

    response = client.get("/api/v1/analytics/summary", headers=headers)
    assert response.status_code == 200
    data = response.json()
    counts = data["counts"]
    # Minimal should be 1, others 0
    assert counts["minimal"] == 1
    assert counts["limited"] == 0
    assert counts["high"] == 0
    assert counts["unacceptable"] == 0
    # No KeyError means None systems are safely ignored
