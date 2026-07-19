from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Generator
import logging

from src.core.config import settings
from src.database.base import Base

logger = logging.getLogger("database_session")

# Connection pool configurations
db_url = settings.get_database_url()

# If using sqlite, disable pool parameters not supported by it
if db_url.startswith("sqlite"):
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(
        db_url,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator:
    # Resolves SessionLocal dynamically so fallback updates are reflected
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db() -> None:
    global engine, SessionLocal
    try:
        # Import models here to register them with Base.metadata before creating
        import src.database.models  # noqa: F401
        logger.info("Initializing database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        if not db_url.startswith("sqlite"):
            logger.warning(f"Primary PostgreSQL connection failed: {str(e)}. Falling back to local SQLite database (threats.db)...")
            fallback_url = "sqlite:///threats.db"
            engine = create_engine(
                fallback_url,
                connect_args={"check_same_thread": False}
            )
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            # Re-attempt table creation
            Base.metadata.create_all(bind=engine)
            logger.info("Local SQLite fallback database tables initialized successfully.")
        else:
            logger.error(f"Error initializing database: {str(e)}", exc_info=True)
            raise e
