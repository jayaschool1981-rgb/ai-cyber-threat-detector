from src.database.models import Alert

def test_predict_requires_authentication(client):
    payload = {
        "Destination Port": 80,
        "Flow Duration": 1000,
        "Total Fwd Packets": 1,
        "Total Backward Packets": 2
    }
    response = client.post("/api/v1/predict", json=payload)
    # Check that access is denied (FastAPI security can return 401 or 403 depending on configuration)
    assert response.status_code in [401, 403]

def test_predict_invalid_auth(client):
    payload = {
        "Destination Port": 80,
        "Flow Duration": 1000
    }
    headers = {"Authorization": "Bearer invalidtoken123"}
    response = client.post("/api/v1/predict", json=payload, headers=headers)
    assert response.status_code == 401
    assert "Invalid or expired" in response.json()["detail"]

def test_predict_success_and_db_logging(client, db_session):
    # Register and login user to get token
    client.post(
        "/api/v1/auth/register",
        json={"username": "user1", "password": "password123"}
    )
    login_res = client.post(
        "/api/v1/auth/login",
        json={"username": "user1", "password": "password123"}
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "Destination Port": 80,
        "Flow Duration": 12000,
        "Total Fwd Packets": 2,
        "Total Backward Packets": 3
    }
    
    response = client.post("/api/v1/predict", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert "prediction" in data[0]
    assert "confidence" in data[0]

    # Verify that an Alert record was written to the DB
    alerts = db_session.query(Alert).all()
    assert len(alerts) == 1
    db_alert = alerts[0]
    assert db_alert.prediction == data[0]["prediction"]
    assert db_alert.confidence == data[0]["confidence"]
    assert db_alert.destination_port == 80
    assert db_alert.flow_duration == 12000

def test_predict_validation_errors(client):
    client.post(
        "/api/v1/auth/register",
        json={"username": "user2", "password": "password123"}
    )
    login_res = client.post(
        "/api/v1/auth/login",
        json={"username": "user2", "password": "password123"}
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Invalid port (outside range 0-65535)
    payload = {
        "Destination Port": 999999,
        "Flow Duration": 1000
    }
    response = client.post("/api/v1/predict", json=payload, headers=headers)
    assert response.status_code == 422
