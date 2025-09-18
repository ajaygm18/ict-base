"""
Stock data models for storing OHLCV and related market data
"""
from sqlalchemy import Column, Integer, String, Float, BigInteger, DateTime, Index
from sqlalchemy.sql import func
from app.database import Base

class StockData(Base):
    __tablename__ = "stock_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    
    # OHLCV data
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)  
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)
    
    # Additional stock data
    bid_price = Column(Float)
    ask_price = Column(Float)
    spread = Column(Float)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_symbol_timestamp_timeframe', 'symbol', 'timestamp', 'timeframe'),
    )

class StockFundamentals(Base):
    __tablename__ = "stock_fundamentals"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, unique=True, index=True)
    
    # Fundamental data
    market_cap = Column(BigInteger)
    pe_ratio = Column(Float)
    eps = Column(Float)
    dividend_yield = Column(Float)
    beta = Column(Float)
    
    # Company info
    sector = Column(String(100))
    industry = Column(String(100))
    employees = Column(Integer)
    
    # Analyst data
    analyst_target_price = Column(Float)
    analyst_recommendation = Column(String(20))
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class MarketHours(Base):
    __tablename__ = "market_hours"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, nullable=False, unique=True, index=True)
    
    # Market session times (all in ET)
    premarket_start = Column(DateTime)  # 4:00 AM ET
    market_open = Column(DateTime)      # 9:30 AM ET
    market_close = Column(DateTime)     # 4:00 PM ET
    afterhours_end = Column(DateTime)   # 8:00 PM ET
    
    # Market status
    is_trading_day = Column(String(10), default="true")  # true/false/holiday
    holiday_name = Column(String(100))
    
    # Metadata
    created_at = Column(DateTime, default=func.now())