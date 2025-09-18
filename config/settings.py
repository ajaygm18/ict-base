"""
Configuration settings for ICT Trading System
Centralized configuration management with environment variable support
"""

import os
from typing import List, Optional
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseSettings(BaseSettings):
    """Database configuration"""
    url: str = Field(default="sqlite:///./ict_trading.db", env="DATABASE_URL")
    echo: bool = Field(default=False, env="DATABASE_ECHO")
    pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")

class RedisSettings(BaseSettings):
    """Redis configuration"""
    url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT")

class DataSourceSettings(BaseSettings):
    """Data source API configurations"""
    alpha_vantage_api_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    finnhub_api_key: Optional[str] = Field(default=None, env="FINNHUB_API_KEY")
    polygon_api_key: Optional[str] = Field(default=None, env="POLYGON_API_KEY")
    iex_api_token: Optional[str] = Field(default=None, env="IEX_API_TOKEN")
    
    # Rate limiting
    requests_per_minute: int = Field(default=60, env="REQUESTS_PER_MINUTE")
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")

class TradingSettings(BaseSettings):
    """Trading configuration"""
    paper_trading: bool = Field(default=True, env="PAPER_TRADING")
    broker_api_key: Optional[str] = Field(default=None, env="BROKER_API_KEY")
    broker_secret_key: Optional[str] = Field(default=None, env="BROKER_SECRET_KEY")
    
    # Trading hours (ET)
    market_open_hour: int = Field(default=9, env="MARKET_OPEN_HOUR")
    market_open_minute: int = Field(default=30, env="MARKET_OPEN_MINUTE")
    market_close_hour: int = Field(default=16, env="MARKET_CLOSE_HOUR")
    market_close_minute: int = Field(default=0, env="MARKET_CLOSE_MINUTE")

class RiskManagementSettings(BaseSettings):
    """Risk management configuration"""
    max_portfolio_risk: float = Field(default=0.02, env="MAX_PORTFOLIO_RISK")
    max_position_risk: float = Field(default=0.01, env="MAX_POSITION_RISK")
    max_drawdown: float = Field(default=0.15, env="MAX_DRAWDOWN")
    max_concurrent_trades: int = Field(default=5, env="MAX_CONCURRENT_TRADES")
    min_risk_reward_ratio: float = Field(default=1.5, env="MIN_RISK_REWARD_RATIO")
    
    # Position sizing
    kelly_fraction: float = Field(default=0.25, env="KELLY_FRACTION")
    fixed_fractional: float = Field(default=0.01, env="FIXED_FRACTIONAL")

class MLSettings(BaseSettings):
    """Machine Learning configuration"""
    model_dir: str = Field(default="data/models", env="MODEL_DIR")
    feature_store_path: str = Field(default="data/features", env="FEATURE_STORE_PATH")
    min_training_samples: int = Field(default=10000, env="MIN_TRAINING_SAMPLES")
    validation_split: float = Field(default=0.2, env="VALIDATION_SPLIT")
    
    # Model parameters
    lstm_hidden_size: int = Field(default=128, env="LSTM_HIDDEN_SIZE")
    lstm_num_layers: int = Field(default=2, env="LSTM_NUM_LAYERS")
    transformer_d_model: int = Field(default=256, env="TRANSFORMER_D_MODEL")
    transformer_nhead: int = Field(default=8, env="TRANSFORMER_NHEAD")

class MonitoringSettings(BaseSettings):
    """Monitoring and logging configuration"""
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    prometheus_port: int = Field(default=8090, env="PROMETHEUS_PORT")
    
    # Performance monitoring
    enable_profiling: bool = Field(default=False, env="ENABLE_PROFILING")
    max_memory_usage_mb: int = Field(default=4096, env="MAX_MEMORY_USAGE_MB")

class PerformanceSettings(BaseSettings):
    """Performance optimization settings"""
    cache_ttl: int = Field(default=300, env="CACHE_TTL")
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    batch_size: int = Field(default=1000, env="BATCH_SIZE")
    
    # Pattern detection optimization
    pattern_detection_batch_size: int = Field(default=100, env="PATTERN_DETECTION_BATCH_SIZE")
    feature_computation_workers: int = Field(default=2, env="FEATURE_COMPUTATION_WORKERS")

class Settings(BaseSettings):
    """Main settings class"""
    debug: bool = Field(default=False, env="DEBUG")
    secret_key: str = Field(default="development-key", env="SECRET_KEY")
    app_name: str = Field(default="ICT Trading System", env="APP_NAME")
    version: str = Field(default="1.0.0", env="VERSION")
    
    # Sub-configurations
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    data_sources: DataSourceSettings = DataSourceSettings()
    trading: TradingSettings = TradingSettings()
    risk_management: RiskManagementSettings = RiskManagementSettings()
    ml: MLSettings = MLSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    performance: PerformanceSettings = PerformanceSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()

# Create necessary directories
Path(settings.ml.model_dir).mkdir(parents=True, exist_ok=True)
Path(settings.ml.feature_store_path).mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(exist_ok=True)
Path("data/cache").mkdir(parents=True, exist_ok=True)