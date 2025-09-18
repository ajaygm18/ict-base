"""
ICT Patterns models for storing all 65 ICT concept detections
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Index, Text
from sqlalchemy.sql import func
from app.database import Base

class ICTPatterns(Base):
    __tablename__ = "ict_patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    pattern_type = Column(String(50), nullable=False, index=True)  # All 65 ICT concepts
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    
    # Pattern-specific data (JSON field for flexibility)
    pattern_data = Column(JSON, nullable=False)
    
    # Pattern location and measurement
    start_timestamp = Column(DateTime)
    end_timestamp = Column(DateTime)
    price_level = Column(Float)
    price_range_high = Column(Float)
    price_range_low = Column(Float)
    
    # Performance tracking
    success_rate = Column(Float)
    avg_return = Column(Float)
    total_occurrences = Column(Integer, default=1)
    
    # ICT specific fields
    market_structure_context = Column(String(50))  # HH, HL, LH, LL
    session_context = Column(String(20))  # premarket, market_hours, afterhours
    volume_context = Column(String(20))   # high, normal, low
    
    # Notes and description
    description = Column(Text)
    notes = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_pattern_symbol_type', 'symbol', 'pattern_type', 'timestamp'),
        Index('idx_pattern_confidence', 'confidence'),
    )

class LiquidityLevels(Base):
    __tablename__ = "liquidity_levels"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Liquidity level details
    level_type = Column(String(30), nullable=False)  # buyside, sellside, equal_highs, equal_lows
    price_level = Column(Float, nullable=False)
    strength = Column(Float, nullable=False)  # 0.0 to 1.0
    
    # Context
    touches = Column(Integer, default=1)
    last_test_timestamp = Column(DateTime)
    breached = Column(String(10), default="false")  # true/false
    breached_timestamp = Column(DateTime)
    
    # Volume analysis
    avg_volume_at_level = Column(BigInteger)
    last_volume_at_level = Column(BigInteger)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class OrderBlocks(Base):
    __tablename__ = "order_blocks"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Order block details
    block_type = Column(String(20), nullable=False)  # bullish, bearish, breaker
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    formation_candle_open = Column(Float, nullable=False)
    formation_candle_close = Column(Float, nullable=False)
    
    # Validity and status
    is_active = Column(String(10), default="true")  # true/false
    mitigation_percentage = Column(Float, default=0.0)  # 0.0 to 100.0
    last_interaction = Column(DateTime)
    
    # Context
    market_structure = Column(String(50))
    impulse_strength = Column(Float)  # Strength of the move that created the OB
    volume_confirmation = Column(String(10))  # high/normal/low
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())