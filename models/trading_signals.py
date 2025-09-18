"""
Trading signals models for storing generated signals and performance tracking
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Index, Text
from sqlalchemy.sql import func
from app.database import Base

class TradingSignals(Base):
    __tablename__ = "trading_signals"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    strategy_name = Column(String(100), nullable=False, index=True)  # One of 15 strategies (concepts 51-65)
    signal_type = Column(String(10), nullable=False)  # BUY/SELL/HOLD
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    
    # Entry/exit data
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    risk_reward_ratio = Column(Float)
    
    # Position sizing
    position_size = Column(Float)
    risk_percentage = Column(Float)
    dollar_risk = Column(Float)
    
    # Signal context
    market_bias = Column(String(20))  # bullish/bearish/neutral
    session_context = Column(String(20))  # premarket/market_hours/afterhours
    pattern_confluence = Column(JSON)  # List of supporting ICT patterns
    
    # Performance tracking
    outcome = Column(String(20))  # WIN/LOSS/PENDING/SCRATCHED
    actual_entry_price = Column(Float)
    actual_exit_price = Column(Float)
    actual_pnl = Column(Float)
    actual_pnl_percentage = Column(Float)
    
    # Execution details
    signal_status = Column(String(20), default="PENDING")  # PENDING/ACTIVE/CLOSED/CANCELLED
    entry_timestamp = Column(DateTime)
    exit_timestamp = Column(DateTime)
    duration_minutes = Column(Integer)
    
    # Notes and analysis
    entry_reason = Column(Text)
    exit_reason = Column(Text)
    notes = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_signal_symbol_strategy', 'symbol', 'strategy_name', 'timestamp'),
        Index('idx_signal_outcome', 'outcome'),
        Index('idx_signal_status', 'signal_status'),
    )

class BacktestResults(Base):
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_name = Column(String(100), nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    
    # Backtest period
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    total_days = Column(Integer, nullable=False)
    
    # Performance metrics
    total_trades = Column(Integer, nullable=False)
    winning_trades = Column(Integer, nullable=False)
    losing_trades = Column(Integer, nullable=False)
    win_rate = Column(Float, nullable=False)
    
    # P&L metrics
    total_pnl = Column(Float, nullable=False)
    total_pnl_percentage = Column(Float, nullable=False)
    avg_win = Column(Float)
    avg_loss = Column(Float)
    profit_factor = Column(Float)
    
    # Risk metrics
    max_drawdown = Column(Float)
    max_drawdown_percentage = Column(Float)
    sharpe_ratio = Column(Float)
    calmar_ratio = Column(Float)
    
    # Additional metrics
    avg_trade_duration_minutes = Column(Integer)
    largest_win = Column(Float)
    largest_loss = Column(Float)
    consecutive_wins = Column(Integer)
    consecutive_losses = Column(Integer)
    
    # Configuration
    backtest_config = Column(JSON)  # Parameters used for the backtest
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_backtest_strategy_symbol', 'strategy_name', 'symbol'),
    )

class StrategyPerformance(Base):
    __tablename__ = "strategy_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_name = Column(String(100), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    
    # Daily performance metrics
    daily_trades = Column(Integer, default=0)
    daily_wins = Column(Integer, default=0)
    daily_losses = Column(Integer, default=0)
    daily_pnl = Column(Float, default=0.0)
    daily_pnl_percentage = Column(Float, default=0.0)
    
    # Running totals
    cumulative_trades = Column(Integer, default=0)
    cumulative_pnl = Column(Float, default=0.0)
    running_win_rate = Column(Float, default=0.0)
    running_profit_factor = Column(Float, default=0.0)
    
    # Risk metrics
    current_drawdown = Column(Float, default=0.0)
    current_drawdown_percentage = Column(Float, default=0.0)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_performance_strategy_date', 'strategy_name', 'date'),
    )