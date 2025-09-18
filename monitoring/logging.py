"""
Comprehensive logging and monitoring system for ICT Trading System
Production-ready logging with structured formats and monitoring integration
"""

import sys
import logging
import structlog
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path
import json
from config.settings import settings

class ICTLogger:
    """Enhanced logging system with structured logging"""
    
    def __init__(self, name: str = "ict_trading"):
        self.name = name
        self._setup_logging()
        self.logger = structlog.get_logger(name)
        
    def _setup_logging(self):
        """Setup structured logging configuration"""
        
        # Configure standard library logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, settings.monitoring.log_level.upper())
        )
        
        # Configure structlog
        timestamper = structlog.processors.TimeStamper(fmt="ISO")
        
        shared_processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
        
        if settings.monitoring.log_format == "json":
            # JSON formatter for production
            structlog.configure(
                processors=shared_processors + [
                    structlog.processors.JSONRenderer()
                ],
                wrapper_class=structlog.stdlib.BoundLogger,
                logger_factory=structlog.stdlib.LoggerFactory(),
                context_class=dict,
                cache_logger_on_first_use=True,
            )
        else:
            # Human-readable formatter for development
            structlog.configure(
                processors=shared_processors + [
                    structlog.dev.ConsoleRenderer()
                ],
                wrapper_class=structlog.stdlib.BoundLogger,
                logger_factory=structlog.stdlib.LoggerFactory(),
                context_class=dict,
                cache_logger_on_first_use=True,
            )
    
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self.logger.info(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context"""
        self.logger.error(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self.logger.warning(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self.logger.debug(message, **kwargs)
    
    def trade_event(self, event_type: str, symbol: str, **kwargs):
        """Log trading-specific events"""
        self.logger.info(
            "trade_event",
            event_type=event_type,
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            **kwargs
        )
    
    def pattern_detection(self, symbol: str, patterns_found: int, execution_time: float, **kwargs):
        """Log pattern detection events"""
        self.logger.info(
            "pattern_detection",
            symbol=symbol,
            patterns_found=patterns_found,
            execution_time_ms=execution_time,
            timestamp=datetime.now().isoformat(),
            **kwargs
        )
    
    def risk_event(self, event_type: str, risk_level: str, **kwargs):
        """Log risk management events"""
        self.logger.warning(
            "risk_event",
            event_type=event_type,
            risk_level=risk_level,
            timestamp=datetime.now().isoformat(),
            **kwargs
        )
    
    def performance_metric(self, metric_name: str, value: float, **kwargs):
        """Log performance metrics"""
        self.logger.info(
            "performance_metric",
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now().isoformat(),
            **kwargs
        )

class PerformanceMonitor:
    """Performance monitoring and metrics collection"""
    
    def __init__(self):
        self.logger = ICTLogger("performance_monitor")
        self.metrics = {}
        
    def record_execution_time(self, operation: str, duration_ms: float, **context):
        """Record execution time for operations"""
        self.logger.performance_metric(
            metric_name=f"{operation}_execution_time",
            value=duration_ms,
            operation=operation,
            **context
        )
        
        # Store in metrics
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration_ms)
    
    def record_memory_usage(self, operation: str, memory_mb: float, **context):
        """Record memory usage"""
        self.logger.performance_metric(
            metric_name=f"{operation}_memory_usage",
            value=memory_mb,
            operation=operation,
            **context
        )
    
    def record_pattern_detection_stats(self, symbol: str, stats: Dict[str, Any]):
        """Record pattern detection statistics"""
        self.logger.pattern_detection(
            symbol=symbol,
            patterns_found=stats.get('total_patterns', 0),
            execution_time=stats.get('execution_time_ms', 0),
            high_confidence_patterns=stats.get('high_confidence_patterns', 0),
            confidence_distribution=stats.get('confidence_distribution', {})
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics"""
        summary = {}
        for operation, times in self.metrics.items():
            if times:
                summary[operation] = {
                    'count': len(times),
                    'avg_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'total_ms': sum(times)
                }
        return summary

class TradingEventLogger:
    """Specialized logger for trading events"""
    
    def __init__(self):
        self.logger = ICTLogger("trading_events")
    
    def log_signal_generation(self, signal: Dict[str, Any]):
        """Log signal generation events"""
        self.logger.trade_event(
            event_type="signal_generated",
            symbol=signal.get('symbol'),
            signal_type=signal.get('type'),
            confidence=signal.get('confidence'),
            patterns=signal.get('supporting_patterns', []),
            timeframe=signal.get('timeframe')
        )
    
    def log_trade_entry(self, trade: Dict[str, Any]):
        """Log trade entry events"""
        self.logger.trade_event(
            event_type="trade_entry",
            symbol=trade.get('symbol'),
            direction=trade.get('direction'),
            entry_price=trade.get('entry_price'),
            position_size=trade.get('position_size'),
            stop_loss=trade.get('stop_loss'),
            take_profit=trade.get('take_profit'),
            risk_amount=trade.get('risk_amount'),
            setup_type=trade.get('setup_type')
        )
    
    def log_trade_exit(self, trade: Dict[str, Any]):
        """Log trade exit events"""
        self.logger.trade_event(
            event_type="trade_exit",
            symbol=trade.get('symbol'),
            exit_price=trade.get('exit_price'),
            pnl=trade.get('pnl'),
            pnl_percentage=trade.get('pnl_percentage'),
            duration_minutes=trade.get('duration_minutes'),
            exit_reason=trade.get('exit_reason')
        )
    
    def log_risk_breach(self, breach_type: str, details: Dict[str, Any]):
        """Log risk management breaches"""
        self.logger.risk_event(
            event_type="risk_breach",
            risk_level="HIGH",
            breach_type=breach_type,
            **details
        )

# Global logger instances
logger = ICTLogger()
performance_monitor = PerformanceMonitor()
trading_logger = TradingEventLogger()

__all__ = ['logger', 'performance_monitor', 'trading_logger', 'ICTLogger', 'PerformanceMonitor', 'TradingEventLogger']