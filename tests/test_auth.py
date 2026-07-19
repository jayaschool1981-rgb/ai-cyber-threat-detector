from src.services.auth_service import hash_password, verify_password, decode_access_token

def test_password_hashing():
    password = "secretpassword"
    hashed = hash_password(password)
    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("wrongpassword", hashed) is False

def test_register_user(client):
    response = client.post(
        "/api/v1/auth/register",
        json={"username": "testuser", "password": "securepassword123"}
    )
    assert response.status_code == 210 or response.status_code == 201  # HTTP_201_CREATED
    data = response.json()
    assert data["username"] == "testuser"
    assert "hashed_password" not in data
    assert data["is_active"] is True

def test_register_duplicate_user(client):
    # Register once
    client.post(
        "/api/v1/auth/register",
        json={"username": "testuser", "password": "securepassword123"}
    )
    # Register again
    response = client.post(
        "/api/v1/auth/register",
        json={"username": "testuser", "password": "anotherpassword"}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Username already registered"

def test_login_success(client):
    # Register user
    client.post(
        "/api/v1/auth/register",
        json={"username": "loginuser", "password": "mypassword123"}
    )
    # Login
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "loginuser", "password": "mypassword123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    
    # Decode token
    decoded = decode_access_token(data["access_token"])
    assert decoded == "loginuser"

def test_login_invalid_credentials(client):
    # Try logging in with non-existent user
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "nonexistent", "password": "password"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"
