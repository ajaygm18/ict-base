"""
Feature Engineering for ICT Pattern Recognition
Creates 200+ technical indicators for stock market analysis
"""

import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class FeatureSet:
    """Container for engineered features"""
    timeframe: str
    features: pd.DataFrame
    feature_names: List[str]
    creation_timestamp: datetime
    symbol: str

class TechnicalIndicatorEngine:
    """Creates comprehensive technical indicators for stock analysis"""
    
    def __init__(self):
        self.indicator_count = 0
        self.feature_cache = {}
        
    def create_price_based_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based technical indicators"""
        indicators = pd.DataFrame(index=df.index)
        
        # Ensure column names are lowercase
        df_cols = {col.lower(): col for col in df.columns}
        close_col = df_cols.get('close', 'close')
        high_col = df_cols.get('high', 'high')
        low_col = df_cols.get('low', 'low')
        open_col = df_cols.get('open', 'open')
        volume_col = df_cols.get('volume', 'volume')
        
        # Moving averages (20 indicators)
        periods = [5, 10, 20, 50, 100, 200]
        for period in periods:
            if len(df) >= period:
                indicators[f'SMA_{period}'] = talib.SMA(df[close_col].values, timeperiod=period)
                indicators[f'EMA_{period}'] = talib.EMA(df[close_col].values, timeperiod=period)
                indicators[f'WMA_{period}'] = talib.WMA(df[close_col].values, timeperiod=period)
            
        # Bollinger Bands (9 indicators)
        for period in [20, 50]:
            if len(df) >= period:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(df[close_col].values, timeperiod=period)
                indicators[f'BB_Upper_{period}'] = bb_upper
                indicators[f'BB_Middle_{period}'] = bb_middle
                indicators[f'BB_Lower_{period}'] = bb_lower
                indicators[f'BB_Width_{period}'] = (bb_upper - bb_lower) / np.where(bb_middle != 0, bb_middle, 1)
            
        # Price channels and ranges (12 indicators)
        for period in [20, 50]:
            if len(df) >= period:
                high_roll = df[high_col].rolling(period)
                low_roll = df[low_col].rolling(period)
                indicators[f'Highest_{period}'] = high_roll.max()
                indicators[f'Lowest_{period}'] = low_roll.min()
                indicators[f'Channel_Width_{period}'] = indicators[f'Highest_{period}'] - indicators[f'Lowest_{period}']
                channel_width = indicators[f'Channel_Width_{period}']
                indicators[f'Price_Position_{period}'] = np.where(
                    channel_width != 0,
                    (df[close_col] - indicators[f'Lowest_{period}']) / channel_width,
                    0.5
                )
            
        # Pivot points (7 indicators)
        pivot = (df[high_col] + df[low_col] + df[close_col]) / 3
        indicators['Pivot'] = pivot
        indicators['R1'] = 2 * pivot - df[low_col]
        indicators['R2'] = pivot + (df[high_col] - df[low_col])
        indicators['S1'] = 2 * pivot - df[high_col]
        indicators['S2'] = pivot - (df[high_col] - df[low_col])
        indicators['R3'] = df[high_col] + 2 * (pivot - df[low_col])
        indicators['S3'] = df[low_col] - 2 * (df[high_col] - pivot)
        
        # Fibonacci levels (8 indicators)
        if len(df) >= 20:
            high_20 = df[high_col].rolling(20).max()
            low_20 = df[low_col].rolling(20).min()
            fib_range = high_20 - low_20
            indicators['Fib_23.6'] = high_20 - 0.236 * fib_range
            indicators['Fib_38.2'] = high_20 - 0.382 * fib_range
            indicators['Fib_50.0'] = high_20 - 0.500 * fib_range
            indicators['Fib_61.8'] = high_20 - 0.618 * fib_range
            indicators['Fib_78.6'] = high_20 - 0.786 * fib_range
        
        # Price patterns (5 indicators)
        indicators['Doji'] = abs(df[close_col] - df[open_col]) / (df[high_col] - df[low_col] + 1e-10)
        indicators['Upper_Shadow'] = df[high_col] - np.maximum(df[open_col], df[close_col])
        indicators['Lower_Shadow'] = np.minimum(df[open_col], df[close_col]) - df[low_col]
        indicators['Body_Size'] = abs(df[close_col] - df[open_col])
        indicators['True_Range'] = talib.TRANGE(df[high_col].values, df[low_col].values, df[close_col].values)
        
        self.indicator_count += indicators.shape[1]
        return indicators
        
    def create_volume_based_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based technical indicators"""
        indicators = pd.DataFrame(index=df.index)
        
        # Ensure column names are lowercase
        df_cols = {col.lower(): col for col in df.columns}
        close_col = df_cols.get('close', 'close')
        high_col = df_cols.get('high', 'high')
        low_col = df_cols.get('low', 'low')
        volume_col = df_cols.get('volume', 'volume')
        
        # Volume moving averages (6 indicators)
        for period in [10, 20, 50]:
            if len(df) >= period:
                vol_sma = df[volume_col].rolling(period).mean()
                indicators[f'Volume_SMA_{period}'] = vol_sma
                indicators[f'Volume_Ratio_{period}'] = df[volume_col] / np.where(vol_sma != 0, vol_sma, 1)
            
        # Volume trend indicators (10 indicators)
        try:
            indicators['OBV'] = talib.OBV(df[close_col].values, df[volume_col].values)
            indicators['AD'] = talib.AD(df[high_col].values, df[low_col].values, df[close_col].values, df[volume_col].values)
            indicators['ADOSC'] = talib.ADOSC(df[high_col].values, df[low_col].values, df[close_col].values, df[volume_col].values)
        except Exception as e:
            logger.warning(f"Error creating volume indicators: {e}")
            # Set default values if calculation fails
            indicators['OBV'] = 0
            indicators['AD'] = 0
            indicators['ADOSC'] = 0
        
        # Volume-weighted price indicators (8 indicators)
        for period in [10, 20]:
            indicators[f'VWAP_{period}'] = (df['Close'] * df['Volume']).rolling(period).sum() / df['Volume'].rolling(period).sum()
            indicators[f'VWAP_Distance_{period}'] = (df['Close'] - indicators[f'VWAP_{period}']) / indicators[f'VWAP_{period}']
            
        # Volume oscillators (6 indicators)
        for period in [14, 21]:
            vol_ema = df['Volume'].ewm(span=period).mean()
            indicators[f'Volume_EMA_{period}'] = vol_ema
            indicators[f'Volume_Oscillator_{period}'] = (df['Volume'] - vol_ema) / vol_ema
            
        # Money flow indicators (4 indicators)
        indicators['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'])
        indicators['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'])
        # Calculate EMV manually since pandas-ta might not have it
        high_low_diff = df['High'] - df['Low']
        volume_adj = df['Volume'] / (high_low_diff + 1e-10)
        indicators['EMV'] = ((df['High'] + df['Low'])/2 - (df['High'].shift() + df['Low'].shift())/2) * volume_adj
        
        self.indicator_count += indicators.shape[1]
        return indicators
        
    def create_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum-based technical indicators"""
        indicators = pd.DataFrame(index=df.index)
        
        # RSI family (12 indicators)
        for period in [7, 14, 21]:
            indicators[f'RSI_{period}'] = talib.RSI(df['Close'], timeperiod=period)
            # Calculate StochRSI manually using pandas-ta
            stochrsi_result = ta.stochrsi(df['Close'], length=period)
            if isinstance(stochrsi_result, pd.DataFrame) and not stochrsi_result.empty:
                # Get the first column (usually STOCHRSIk)
                indicators[f'StochRSI_{period}'] = stochrsi_result.iloc[:, 0]
            else:
                # Fallback: calculate manually
                rsi = talib.RSI(df['Close'], timeperiod=period)
                rsi_min = rsi.rolling(period).min()
                rsi_max = rsi.rolling(period).max()
                indicators[f'StochRSI_{period}'] = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
            
        # Stochastic oscillators (9 indicators)
        for k_period, d_period in [(14, 3), (21, 5)]:
            slowk, slowd = talib.STOCH(df['High'], df['Low'], df['Close'], 
                                     fastk_period=k_period, slowk_period=3, slowd_period=d_period)
            indicators[f'Stoch_K_{k_period}_{d_period}'] = slowk
            indicators[f'Stoch_D_{k_period}_{d_period}'] = slowd
            indicators[f'Stoch_Signal_{k_period}_{d_period}'] = slowk - slowd
            
        # MACD family (12 indicators)
        macd_combos = [(12, 26, 9), (5, 35, 5)]
        for fast, slow, signal in macd_combos:
            macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=fast, slowperiod=slow, signalperiod=signal)
            indicators[f'MACD_{fast}_{slow}_{signal}'] = macd
            indicators[f'MACD_Signal_{fast}_{slow}_{signal}'] = macdsignal
            indicators[f'MACD_Hist_{fast}_{slow}_{signal}'] = macdhist
            
        # Rate of change indicators (8 indicators)
        for period in [1, 5, 10, 20]:
            indicators[f'ROC_{period}'] = talib.ROC(df['Close'], timeperiod=period)
            indicators[f'Momentum_{period}'] = talib.MOM(df['Close'], timeperiod=period)
            
        # Williams %R (4 indicators)
        for period in [14, 21]:
            indicators[f'Williams_R_{period}'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=period)
            
        # CCI - Commodity Channel Index (3 indicators)
        for period in [14, 20]:
            indicators[f'CCI_{period}'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=period)
            
        self.indicator_count += indicators.shape[1]
        return indicators
        
    def create_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based technical indicators"""
        indicators = pd.DataFrame(index=df.index)
        
        # Average True Range family (6 indicators)
        for period in [7, 14, 21]:
            indicators[f'ATR_{period}'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=period)
            indicators[f'ATR_Percent_{period}'] = indicators[f'ATR_{period}'] / df['Close']
            
        # Volatility measures (8 indicators)
        for period in [10, 20]:
            returns = df['Close'].pct_change()
            indicators[f'Volatility_{period}'] = returns.rolling(period).std()
            indicators[f'Parkinson_{period}'] = np.sqrt((1/(4*np.log(2))) * np.log(df['High']/df['Low'])**2).rolling(period).mean()
            indicators[f'Garman_Klass_{period}'] = (0.5 * np.log(df['High']/df['Low'])**2 - 
                                                  (2*np.log(2)-1) * np.log(df['Close']/df['Open'])**2).rolling(period).mean()
            
        # Keltner Channels (6 indicators)
        for period in [20, 50]:
            ema = df['Close'].ewm(span=period).mean()
            atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=period)
            indicators[f'Keltner_Upper_{period}'] = ema + 2 * atr
            indicators[f'Keltner_Lower_{period}'] = ema - 2 * atr
            indicators[f'Keltner_Width_{period}'] = 4 * atr / ema
            
        self.indicator_count += indicators.shape[1]
        return indicators
        
    def create_ict_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ICT-specific features for pattern recognition"""
        indicators = pd.DataFrame(index=df.index)
        
        # Fair Value Gap features (10 indicators)
        indicators['FVG_Bull_Gap'] = np.where(
            (df['Low'].shift(1) > df['High'].shift(-1)) & 
            (df['Close'] > df['Close'].shift(1)), 1, 0
        )
        indicators['FVG_Bear_Gap'] = np.where(
            (df['High'].shift(1) < df['Low'].shift(-1)) & 
            (df['Close'] < df['Close'].shift(1)), 1, 0
        )
        indicators['Gap_Size'] = np.maximum(
            df['Low'].shift(1) - df['High'].shift(-1),
            df['High'].shift(1) - df['Low'].shift(-1)
        )
        indicators['Gap_Size_Percent'] = indicators['Gap_Size'] / df['Close']
        
        # Order Block features (8 indicators)
        # Bullish Order Block: Last down candle before bullish impulse
        down_candle = df['Close'] < df['Open']
        up_impulse = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) > 0.02
        indicators['Bull_OB_Signal'] = (down_candle & up_impulse.shift(-1)).astype(int)
        
        # Bearish Order Block: Last up candle before bearish impulse  
        up_candle = df['Close'] > df['Open']
        down_impulse = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) < -0.02
        indicators['Bear_OB_Signal'] = (up_candle & down_impulse.shift(-1)).astype(int)
        
        # Liquidity features (12 indicators)
        # Equal highs/lows detection
        high_roll = df['High'].rolling(20)
        low_roll = df['Low'].rolling(20)
        indicators['Equal_Highs'] = (abs(df['High'] - high_roll.max().shift(1)) / df['High'] < 0.001).astype(int)
        indicators['Equal_Lows'] = (abs(df['Low'] - low_roll.min().shift(1)) / df['Low'] < 0.001).astype(int)
        
        # Liquidity sweeps
        indicators['High_Sweep'] = (df['High'] > df['High'].rolling(20).max().shift(1)).astype(int)
        indicators['Low_Sweep'] = (df['Low'] < df['Low'].rolling(20).min().shift(1)).astype(int)
        
        # Premium/Discount zones (6 indicators)
        high_20 = df['High'].rolling(20).max()
        low_20 = df['Low'].rolling(20).min()
        range_20 = high_20 - low_20
        price_position = (df['Close'] - low_20) / range_20
        indicators['Premium_Zone'] = (price_position > 0.7).astype(int)
        indicators['Discount_Zone'] = (price_position < 0.3).astype(int)
        indicators['Equilibrium_Zone'] = ((price_position >= 0.4) & (price_position <= 0.6)).astype(int)
        indicators['OTE_Zone'] = ((price_position >= 0.62) & (price_position <= 0.79)).astype(int)
        
        # Market structure features (8 indicators)
        # Higher highs, higher lows, lower highs, lower lows
        hh = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
        hl = (df['Low'] > df['Low'].shift(1)) & (df['Low'].shift(1) > df['Low'].shift(2))
        lh = (df['High'] < df['High'].shift(1)) & (df['High'].shift(1) < df['High'].shift(2))
        ll = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
        
        indicators['Higher_High'] = hh.astype(int)
        indicators['Higher_Low'] = hl.astype(int)
        indicators['Lower_High'] = lh.astype(int)
        indicators['Lower_Low'] = ll.astype(int)
        
        # Trend structure score
        indicators['Bull_Structure'] = (indicators['Higher_High'] + indicators['Higher_Low']).rolling(10).sum()
        indicators['Bear_Structure'] = (indicators['Lower_High'] + indicators['Lower_Low']).rolling(10).sum()
        
        self.indicator_count += indicators.shape[1]
        return indicators
        
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features for ML models"""
        indicators = pd.DataFrame(index=df.index)
        
        # Rolling correlations (6 indicators)
        price_change = df['Close'].pct_change()
        volume_change = df['Volume'].pct_change()
        
        for period in [10, 20, 50]:
            indicators[f'Price_Volume_Corr_{period}'] = price_change.rolling(period).corr(volume_change)
            
        # Z-scores (8 indicators)
        for period in [20, 50]:
            price_mean = df['Close'].rolling(period).mean()
            price_std = df['Close'].rolling(period).std()
            indicators[f'Price_ZScore_{period}'] = (df['Close'] - price_mean) / price_std
            
            volume_mean = df['Volume'].rolling(period).mean()
            volume_std = df['Volume'].rolling(period).std()
            indicators[f'Volume_ZScore_{period}'] = (df['Volume'] - volume_mean) / volume_std
            
        # Skewness and Kurtosis (8 indicators)
        for period in [20, 50]:
            returns = df['Close'].pct_change()
            indicators[f'Returns_Skew_{period}'] = returns.rolling(period).skew()
            indicators[f'Returns_Kurt_{period}'] = returns.rolling(period).kurt()
            
        # Entropy measures (4 indicators)
        for period in [20, 50]:
            # Price dispersion entropy
            price_range = df['High'] - df['Low']
            indicators[f'Price_Entropy_{period}'] = -((price_range / price_range.rolling(period).sum()).rolling(period).apply(
                lambda x: np.sum(x * np.log(x + 1e-10))
            ))
            
        self.indicator_count += indicators.shape[1]
        return indicators
        
    def create_comprehensive_features(self, df: pd.DataFrame, symbol: str, timeframe: str) -> FeatureSet:
        """Create comprehensive feature set with 200+ indicators"""
        
        logger.info(f"Creating comprehensive features for {symbol} on {timeframe} timeframe")
        
        # Reset indicator count
        self.indicator_count = 0
        
        # Create all indicator categories
        price_features = self.create_price_based_indicators(df)
        volume_features = self.create_volume_based_indicators(df)
        momentum_features = self.create_momentum_indicators(df)
        volatility_features = self.create_volatility_indicators(df)
        ict_features = self.create_ict_specific_features(df)
        statistical_features = self.create_statistical_features(df)
        
        # Combine all features
        all_features = pd.concat([
            price_features,
            volume_features, 
            momentum_features,
            volatility_features,
            ict_features,
            statistical_features
        ], axis=1)
        
        # Remove any columns with all NaN values
        all_features = all_features.dropna(axis=1, how='all')
        
        # Forward fill and backward fill remaining NaN values
        all_features = all_features.ffill().bfill()
        
        feature_names = all_features.columns.tolist()
        
        logger.info(f"Created {len(feature_names)} features for {symbol}")
        
        return FeatureSet(
            timeframe=timeframe,
            features=all_features,
            feature_names=feature_names,
            creation_timestamp=datetime.now(),
            symbol=symbol
        )

class MultiTimeframeFeatureEngine:
    """Handles feature creation across multiple timeframes"""
    
    def __init__(self):
        self.indicator_engine = TechnicalIndicatorEngine()
        self.feature_cache = {}
        
    def create_multi_timeframe_features(self, symbol: str, base_data: Dict[str, pd.DataFrame]) -> Dict[str, FeatureSet]:
        """Create features across multiple timeframes"""
        
        timeframes = ['1m', '5m', '15m', '1h', '1d']
        feature_sets = {}
        
        for timeframe in timeframes:
            if timeframe in base_data and not base_data[timeframe].empty:
                feature_set = self.indicator_engine.create_comprehensive_features(
                    base_data[timeframe], symbol, timeframe
                )
                feature_sets[timeframe] = feature_set
                
        return feature_sets
        
    def align_timeframe_features(self, feature_sets: Dict[str, FeatureSet], target_timeframe: str = '5m') -> pd.DataFrame:
        """Align features from different timeframes to target timeframe"""
        
        if target_timeframe not in feature_sets:
            raise ValueError(f"Target timeframe {target_timeframe} not found in feature sets")
            
        target_features = feature_sets[target_timeframe].features.copy()
        
        # Add features from higher timeframes
        for timeframe, feature_set in feature_sets.items():
            if timeframe != target_timeframe:
                # Resample higher timeframe features to target timeframe
                if timeframe in ['1h', '1d']:
                    resampled_features = feature_set.features.resample('5T').ffill()
                    
                    # Add suffix to distinguish timeframe
                    resampled_features.columns = [f"{col}_{timeframe}" for col in resampled_features.columns]
                    
                    # Align with target timeframe
                    aligned_features = resampled_features.reindex(target_features.index, method='ffill')
                    
                    # Concatenate with target features
                    target_features = pd.concat([target_features, aligned_features], axis=1)
                    
        return target_features

# Export main classes
__all__ = ['TechnicalIndicatorEngine', 'MultiTimeframeFeatureEngine', 'FeatureSet']