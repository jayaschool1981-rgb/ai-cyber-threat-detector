from fastapi import FastAPI, Response
from fastapi.testclient import TestClient
from api.middleware.rate_limit import RateLimitMiddleware

def test_rate_limiting_middleware():
    test_app = FastAPI()
    
    @test_app.get("/test")
    def test_endpoint():
        return {"status": "ok"}
        
    # Configure rate limiting with a low limit of 2 requests
    test_app.add_middleware(RateLimitMiddleware, limit=2, window=10)
    
    client = TestClient(test_app)
    
    # 1st request -> Allowed
    res1 = client.get("/test")
    assert res1.status_code == 200
    
    # 2nd request -> Allowed
    res2 = client.get("/test")
    assert res2.status_code == 200
    
    # 3rd request -> Rate limited (429)
    res3 = client.get("/test")
    assert res3.status_code == 429
    assert "Rate limit exceeded" in res3.text
