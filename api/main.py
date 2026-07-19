import logging
import uvicorn
import contextlib
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import settings
from src.services.inference_service import inference_service
from src.database.session import init_db
from api.v1.endpoints.predict import router as predict_router
from api.v1.endpoints.auth import router as auth_router
from api.middleware.rate_limit import RateLimitMiddleware

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("main")

# Lifespan for startup/shutdown actions
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up threat detection API...")
    try:
        init_db()
    except Exception as e:
        logger.critical(f"Database initialization failed: {str(e)}")
    yield
    logger.info("Shutting down threat detection API...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Enterprise Network Cyber Threat Detection Engine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to trusted domains in production settings
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Rate Limiting Middleware
app.add_middleware(RateLimitMiddleware, limit=100, window=60)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception occurred: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."}
    )

# Root Welcome Endpoint
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the AI-Powered Cyber Threat Detection API.",
        "docs": "/docs",
        "health": "/health"
    }

# Healthcheck Router
@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    return {
        "status": "ok",
        "env": settings.ENV,
        "model_loaded": (inference_service.model is not None or inference_service.session is not None),
        "n_expected_cols": len(inference_service.expected_cols)
    }

# Register V1 Endpoints
app.include_router(predict_router, prefix=settings.API_V1_STR)
app.include_router(auth_router, prefix=settings.API_V1_STR + "/auth", tags=["Authentication"])

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=(settings.ENV == "development")
    )

