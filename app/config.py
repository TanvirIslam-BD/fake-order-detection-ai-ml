import os

class Config:
    """Base configuration class with default settings."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "your_default_secret_key")
    # SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "sqlite:///site.db")
    # SQLALCHEMY_TRACK_MODIFICATIONS = False
    DEBUG = True
    TESTING = False

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    SQLALCHEMY_ECHO = True  # Show SQL queries in console for debugging

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///test.db"  # Use a separate test database
    WTF_CSRF_ENABLED = False  # Disable CSRF protection for easier testing

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "postgresql://user:password@localhost/prod_db")

# Dictionary to help easily retrieve configurations
config = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}
