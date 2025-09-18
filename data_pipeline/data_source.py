"""
Robust Data Pipeline for Real-time Market Data
Handles data ingestion, validation, cleaning, and storage with multiple sources
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import yfinance as yf
from pathlib import Path
import redis
import json
import time
from concurrent.futures import ThreadPoolExecutor

from config.settings import settings
from monitoring.logging import logger, performance_monitor

@dataclass
class MarketData:
    """Standardized market data structure"""
    symbol: str
    timestamp: datetime
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    source: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""
    completeness: float  # % of non-null values
    consistency: float   # % of consistent data points
    timeliness: float   # % of timely data
    accuracy: float     # % of accurate data (within expected ranges)
    duplicates: int     # Number of duplicate records
    outliers: int       # Number of outliers detected
    
class DataValidator:
    """Validates market data quality and integrity"""
    
    def __init__(self):
        self.logger = logger
        
    def validate_ohlcv(self, data: MarketData) -> Dict[str, bool]:
        """Validate OHLCV data integrity"""
        validations = {
            'price_consistency': self._validate_price_consistency(data),
            'volume_validity': self._validate_volume(data),
            'timestamp_validity': self._validate_timestamp(data),
            'price_range': self._validate_price_range(data),
            'no_nulls': self._validate_no_nulls(data)
        }
        return validations
    
    def _validate_price_consistency(self, data: MarketData) -> bool:
        """Validate OHLC price relationships"""
        try:
            # High should be >= max(open, close)
            if data.high < max(data.open, data.close):
                return False
            
            # Low should be <= min(open, close)
            if data.low > min(data.open, data.close):
                return False
            
            # High >= Low
            if data.high < data.low:
                return False
                
            return True
        except Exception:
            return False
    
    def _validate_volume(self, data: MarketData) -> bool:
        """Validate volume data"""
        return data.volume >= 0
    
    def _validate_timestamp(self, data: MarketData) -> bool:
        """Validate timestamp"""
        now = datetime.now()
        # Data should not be more than 1 hour in the future
        return data.timestamp <= now + timedelta(hours=1)
    
    def _validate_price_range(self, data: MarketData) -> bool:
        """Validate price is within reasonable range"""
        prices = [data.open, data.high, data.low, data.close]
        return all(0.01 <= price <= 100000 for price in prices)
    
    def _validate_no_nulls(self, data: MarketData) -> bool:
        """Check for null values in critical fields"""
        critical_fields = [data.open, data.high, data.low, data.close, data.volume]
        return all(value is not None for value in critical_fields)
    
    def calculate_data_quality(self, data_batch: List[MarketData]) -> DataQualityMetrics:
        """Calculate overall data quality metrics for a batch"""
        if not data_batch:
            return DataQualityMetrics(0, 0, 0, 0, 0, 0)
        
        total_records = len(data_batch)
        valid_records = 0
        outliers = 0
        duplicates = 0
        
        # Check for duplicates
        seen_timestamps = set()
        for record in data_batch:
            timestamp_key = (record.symbol, record.timestamp, record.timeframe)
            if timestamp_key in seen_timestamps:
                duplicates += 1
            seen_timestamps.add(timestamp_key)
        
        # Validate each record
        for record in data_batch:
            validations = self.validate_ohlcv(record)
            if all(validations.values()):
                valid_records += 1
            
            # Check for outliers (simplified)
            if self._is_outlier(record, data_batch):
                outliers += 1
        
        return DataQualityMetrics(
            completeness=valid_records / total_records,
            consistency=valid_records / total_records,
            timeliness=1.0,  # Simplified for now
            accuracy=valid_records / total_records,
            duplicates=duplicates,
            outliers=outliers
        )
    
    def _is_outlier(self, record: MarketData, batch: List[MarketData]) -> bool:
        """Simple outlier detection"""
        # Get prices from same symbol
        symbol_prices = [r.close for r in batch if r.symbol == record.symbol]
        if len(symbol_prices) < 5:
            return False
        
        # Use IQR method
        q75, q25 = np.percentile(symbol_prices, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        return record.close < lower_bound or record.close > upper_bound

class DataSource(ABC):
    """Abstract base class for data sources"""
    
    @abstractmethod
    async def get_real_time_data(self, symbols: List[str]) -> List[MarketData]:
        """Get real-time market data"""
        pass
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, start_date: str, end_date: str, timeframe: str) -> List[MarketData]:
        """Get historical market data"""
        pass

class YFinanceDataSource(DataSource):
    """Yahoo Finance data source"""
    
    def __init__(self):
        self.source_name = "yfinance"
        self.rate_limiter = asyncio.Semaphore(settings.data_sources.max_concurrent_requests)
    
    async def get_real_time_data(self, symbols: List[str]) -> List[MarketData]:
        """Get real-time data from Yahoo Finance"""
        async with self.rate_limiter:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                data = await loop.run_in_executor(executor, self._fetch_real_time, symbols)
            return data
    
    def _fetch_real_time(self, symbols: List[str]) -> List[MarketData]:
        """Fetch real-time data synchronously"""
        market_data = []
        try:
            # Get current data for all symbols
            tickers = yf.Tickers(' '.join(symbols))
            
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    hist = ticker.history(period="1d", interval="1m")
                    
                    if not hist.empty:
                        latest = hist.iloc[-1]
                        data = MarketData(
                            symbol=symbol,
                            timestamp=latest.name.to_pydatetime(),
                            timeframe="1m",
                            open=float(latest['Open']),
                            high=float(latest['High']),
                            low=float(latest['Low']),
                            close=float(latest['Close']),
                            volume=int(latest['Volume']),
                            source=self.source_name
                        )
                        market_data.append(data)
                
                except Exception as e:
                    logger.error(f"Error fetching real-time data for {symbol}", error=str(e))
        
        except Exception as e:
            logger.error(f"Error in YFinance real-time fetch", error=str(e))
        
        return market_data
    
    async def get_historical_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1d") -> List[MarketData]:
        """Get historical data from Yahoo Finance"""
        async with self.rate_limiter:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                data = await loop.run_in_executor(
                    executor, self._fetch_historical, symbol, start_date, end_date, timeframe
                )
            return data
    
    def _fetch_historical(self, symbol: str, start_date: str, end_date: str, timeframe: str) -> List[MarketData]:
        """Fetch historical data synchronously"""
        market_data = []
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date, interval=timeframe)
            
            for timestamp, row in hist.iterrows():
                data = MarketData(
                    symbol=symbol,
                    timestamp=timestamp.to_pydatetime(),
                    timeframe=timeframe,
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume']),
                    source=self.source_name
                )
                market_data.append(data)
        
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}", 
                        symbol=symbol, start_date=start_date, end_date=end_date, error=str(e))
        
        return market_data

class DataCache:
    """Redis-based data caching system"""
    
    def __init__(self):
        try:
            self.redis_client = redis.from_url(settings.redis.url)
            self.default_ttl = settings.performance.cache_ttl
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            self.redis_client = None
    
    async def get_cached_data(self, key: str) -> Optional[List[MarketData]]:
        """Get cached market data"""
        if not self.redis_client:
            return None
        
        try:
            cached = self.redis_client.get(key)
            if cached:
                data_list = json.loads(cached)
                return [MarketData.from_dict(d) for d in data_list]
        except Exception as e:
            logger.error("Error retrieving cached data", key=key, error=str(e))
        
        return None
    
    async def cache_data(self, key: str, data: List[MarketData], ttl: int = None):
        """Cache market data"""
        if not self.redis_client or not data:
            return
        
        try:
            data_dicts = [d.to_dict() for d in data]
            self.redis_client.setex(
                key, 
                ttl or self.default_ttl, 
                json.dumps(data_dicts, default=str)
            )
        except Exception as e:
            logger.error("Error caching data", key=key, error=str(e))

class DataPipeline:
    """Main data pipeline orchestrator"""
    
    def __init__(self):
        self.data_sources = {
            'yfinance': YFinanceDataSource()
        }
        self.validator = DataValidator()
        self.cache = DataCache()
        self.processed_data = {}
        
    async def get_real_time_data(self, symbols: List[str], use_cache: bool = True) -> Dict[str, List[MarketData]]:
        """Get real-time data with caching and validation"""
        start_time = time.time()
        
        result = {}
        
        for symbol in symbols:
            cache_key = f"realtime:{symbol}:1m"
            
            # Check cache first
            if use_cache:
                cached_data = await self.cache.get_cached_data(cache_key)
                if cached_data:
                    result[symbol] = cached_data
                    continue
            
            # Fetch from data sources
            data = []
            for source_name, source in self.data_sources.items():
                try:
                    source_data = await source.get_real_time_data([symbol])
                    data.extend(source_data)
                except Exception as e:
                    logger.error(f"Error fetching from {source_name}", symbol=symbol, error=str(e))
            
            # Validate data
            if data:
                valid_data = []
                for record in data:
                    validations = self.validator.validate_ohlcv(record)
                    if all(validations.values()):
                        valid_data.append(record)
                    else:
                        logger.warning("Invalid data record", symbol=symbol, validations=validations)
                
                if valid_data:
                    result[symbol] = valid_data
                    # Cache valid data
                    await self.cache.cache_data(cache_key, valid_data, ttl=60)  # Cache for 1 minute
        
        execution_time = (time.time() - start_time) * 1000
        performance_monitor.record_execution_time("real_time_data_fetch", execution_time, symbols=symbols)
        
        return result
    
    async def get_historical_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1d") -> List[MarketData]:
        """Get historical data with validation and caching"""
        start_time = time.time()
        
        cache_key = f"historical:{symbol}:{start_date}:{end_date}:{timeframe}"
        
        # Check cache first
        cached_data = await self.cache.get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        # Fetch from data sources
        all_data = []
        for source_name, source in self.data_sources.items():
            try:
                source_data = await source.get_historical_data(symbol, start_date, end_date, timeframe)
                all_data.extend(source_data)
            except Exception as e:
                logger.error(f"Error fetching historical from {source_name}", 
                           symbol=symbol, error=str(e))
        
        # Remove duplicates and validate
        if all_data:
            # Sort by timestamp
            all_data.sort(key=lambda x: x.timestamp)
            
            # Remove duplicates
            seen_timestamps = set()
            unique_data = []
            for record in all_data:
                timestamp_key = record.timestamp
                if timestamp_key not in seen_timestamps:
                    seen_timestamps.add(timestamp_key)
                    
                    # Validate record
                    validations = self.validator.validate_ohlcv(record)
                    if all(validations.values()):
                        unique_data.append(record)
            
            # Cache for longer period (1 hour for historical data)
            await self.cache.cache_data(cache_key, unique_data, ttl=3600)
            
            execution_time = (time.time() - start_time) * 1000
            performance_monitor.record_execution_time("historical_data_fetch", execution_time, symbol=symbol)
            
            return unique_data
        
        return []
    
    def get_data_quality_report(self, data: List[MarketData]) -> DataQualityMetrics:
        """Get data quality assessment"""
        return self.validator.calculate_data_quality(data)
    
    async def start_real_time_stream(self, symbols: List[str], callback=None):
        """Start real-time data streaming"""
        logger.info("Starting real-time data stream", symbols=symbols)
        
        while True:
            try:
                data = await self.get_real_time_data(symbols, use_cache=False)
                
                if callback and data:
                    await callback(data)
                
                # Wait before next fetch (respect rate limits)
                await asyncio.sleep(60)  # 1 minute intervals
                
            except Exception as e:
                logger.error("Error in real-time stream", error=str(e))
                await asyncio.sleep(5)  # Wait before retry

# Global data pipeline instance
data_pipeline = DataPipeline()

__all__ = ['DataPipeline', 'MarketData', 'DataValidator', 'DataQualityMetrics', 'data_pipeline']