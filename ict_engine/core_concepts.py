"""
ICT Core Concepts Implementation (Concepts 1-20)
Market Structure, Liquidity, Order Blocks, Fair Value Gaps, and more
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MarketStructure(Enum):
    HIGHER_HIGH = "HH"
    HIGHER_LOW = "HL" 
    LOWER_HIGH = "LH"
    LOWER_LOW = "LL"
    EQUAL_HIGH = "EH"
    EQUAL_LOW = "EL"

@dataclass
class SwingPoint:
    timestamp: datetime
    price: float
    swing_type: str  # 'high' or 'low'
    index: int
    strength: float = 0.0

@dataclass
class LiquidityPool:
    timestamp: datetime
    price_level: float
    pool_type: str  # 'buyside', 'sellside', 'equal_highs', 'equal_lows'
    strength: float
    volume: int = 0
    touches: int = 1

@dataclass
class OrderBlock:
    timestamp: datetime
    high_price: float
    low_price: float
    block_type: str  # 'bullish', 'bearish'
    formation_candle_index: int
    strength: float
    is_breaker: bool = False

@dataclass
class FairValueGap:
    timestamp: datetime
    gap_high: float
    gap_low: float
    gap_type: str  # 'bullish', 'bearish'
    gap_size: float
    mitigation_level: float
    is_mitigated: bool = False

class StockMarketStructureAnalyzer:
    def __init__(self):
        self.swing_lookback = 5  # Periods to look back for swing identification
        self.liquidity_threshold = 0.001  # 0.1% for equal levels
        self.min_volume_ratio = 1.5  # Minimum volume ratio for significance
        
    def concept_1_market_structure_hh_hl_lh_ll(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 1: Market Structure (HH, HL, LH, LL)
        - Higher Highs (HH) detection with stock-specific logic
        - Higher Lows (HL) pattern recognition for stock trends
        - Lower Highs (LH) identification in stock downtrends  
        - Lower Lows (LL) pattern detection
        """
        if stock_data.empty or len(stock_data) < 10:
            return {'error': 'Insufficient data for market structure analysis'}
        
        try:
            # Identify swing points
            swing_highs = self._find_swing_highs(stock_data)
            swing_lows = self._find_swing_lows(stock_data)
            
            # Classify market structure
            structure_analysis = {
                'swing_highs': swing_highs,
                'swing_lows': swing_lows,
                'current_structure': self._classify_current_structure(swing_highs, swing_lows),
                'structure_breaks': self._detect_structure_breaks(swing_highs, swing_lows),
                'trend_direction': self._determine_trend_direction(swing_highs, swing_lows),
                'confidence': self._calculate_structure_confidence(swing_highs, swing_lows)
            }
            
            return structure_analysis
            
        except Exception as e:
            logger.error(f"Error in market structure analysis: {e}")
            return {'error': str(e)}
    
    def concept_2_liquidity_buyside_sellside(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 2: Liquidity (buy-side & sell-side)
        - Buy-side liquidity above stock resistance levels
        - Sell-side liquidity below stock support levels
        - Institutional liquidity pools in stocks
        """
        try:
            swing_highs = self._find_swing_highs(stock_data)
            swing_lows = self._find_swing_lows(stock_data)
            
            buyside_liquidity = []
            sellside_liquidity = []
            
            # Identify buy-side liquidity (above recent highs)
            for high in swing_highs[-5:]:  # Last 5 swing highs
                if self._is_significant_level(stock_data, high.price):
                    buyside_liquidity.append({
                        'price_level': high.price,
                        'timestamp': high.timestamp,
                        'strength': high.strength,
                        'type': 'buyside',
                        'distance_from_current': abs(high.price - stock_data['close'].iloc[-1]) / stock_data['close'].iloc[-1]
                    })
            
            # Identify sell-side liquidity (below recent lows)
            for low in swing_lows[-5:]:  # Last 5 swing lows
                if self._is_significant_level(stock_data, low.price):
                    sellside_liquidity.append({
                        'price_level': low.price,
                        'timestamp': low.timestamp,
                        'strength': low.strength,
                        'type': 'sellside',
                        'distance_from_current': abs(low.price - stock_data['close'].iloc[-1]) / stock_data['close'].iloc[-1]
                    })
            
            return {
                'buyside_liquidity': sorted(buyside_liquidity, key=lambda x: x['strength'], reverse=True),
                'sellside_liquidity': sorted(sellside_liquidity, key=lambda x: x['strength'], reverse=True),
                'nearest_buyside': min(buyside_liquidity, key=lambda x: x['distance_from_current']) if buyside_liquidity else None,
                'nearest_sellside': min(sellside_liquidity, key=lambda x: x['distance_from_current']) if sellside_liquidity else None,
                'liquidity_balance': len(buyside_liquidity) - len(sellside_liquidity)
            }
            
        except Exception as e:
            logger.error(f"Error in liquidity analysis: {e}")
            return {'error': str(e)}
    
    def concept_3_liquidity_pools(self, stock_data: pd.DataFrame) -> List[LiquidityPool]:
        """
        CONCEPT 3: Liquidity Pools (equal highs/lows, trendline liquidity)
        - Equal highs/lows in stock price action
        - Trendline liquidity for stock breakouts
        - Relative Equal Highs/Lows (REH/REL) in stocks
        """
        try:
            pools = []
            
            # Find equal highs
            swing_highs = self._find_swing_highs(stock_data)
            equal_highs = self._find_equal_levels(swing_highs, 'high')
            
            for level_group in equal_highs:
                if len(level_group) >= 2:
                    avg_price = np.mean([point.price for point in level_group])
                    pools.append(LiquidityPool(
                        timestamp=level_group[-1].timestamp,
                        price_level=avg_price,
                        pool_type='equal_highs',
                        strength=len(level_group) * 0.3,  # More touches = stronger
                        touches=len(level_group)
                    ))
            
            # Find equal lows
            swing_lows = self._find_swing_lows(stock_data)
            equal_lows = self._find_equal_levels(swing_lows, 'low')
            
            for level_group in equal_lows:
                if len(level_group) >= 2:
                    avg_price = np.mean([point.price for point in level_group])
                    pools.append(LiquidityPool(
                        timestamp=level_group[-1].timestamp,
                        price_level=avg_price,
                        pool_type='equal_lows',
                        strength=len(level_group) * 0.3,
                        touches=len(level_group)
                    ))
            
            # Find trendline liquidity
            trendline_pools = self._find_trendline_liquidity(stock_data)
            pools.extend(trendline_pools)
            
            return sorted(pools, key=lambda x: x.strength, reverse=True)
            
        except Exception as e:
            logger.error(f"Error in liquidity pools analysis: {e}")
            return []
    
    def concept_4_order_blocks_bullish_bearish(self, stock_data: pd.DataFrame) -> List[OrderBlock]:
        """
        CONCEPT 4: Order Blocks (Bullish & Bearish)
        - Institutional order blocks in stock movements
        - Last down candle before stock impulse up
        - Last up candle before stock impulse down
        """
        try:
            order_blocks = []
            
            if len(stock_data) < 10:
                return order_blocks
            
            # Look for bullish order blocks (last down candle before impulse up)
            for i in range(3, len(stock_data) - 3):
                # Check if current candle is bearish and followed by bullish impulse
                current_candle = stock_data.iloc[i]
                
                if current_candle['close'] < current_candle['open']:  # Bearish candle
                    # Check for bullish impulse in next 3 candles
                    next_candles = stock_data.iloc[i+1:i+4]
                    impulse_strength = (next_candles['close'].max() - current_candle['low']) / current_candle['low']
                    
                    if impulse_strength > 0.02:  # 2% impulse move
                        order_blocks.append(OrderBlock(
                            timestamp=current_candle['timestamp'] if 'timestamp' in current_candle else current_candle.name,
                            high_price=current_candle['high'],
                            low_price=current_candle['low'],
                            block_type='bullish',
                            formation_candle_index=i,
                            strength=impulse_strength
                        ))
            
            # Look for bearish order blocks (last up candle before impulse down)
            for i in range(3, len(stock_data) - 3):
                current_candle = stock_data.iloc[i]
                
                if current_candle['close'] > current_candle['open']:  # Bullish candle
                    # Check for bearish impulse in next 3 candles
                    next_candles = stock_data.iloc[i+1:i+4]
                    impulse_strength = (current_candle['high'] - next_candles['close'].min()) / current_candle['high']
                    
                    if impulse_strength > 0.02:  # 2% impulse move
                        order_blocks.append(OrderBlock(
                            timestamp=current_candle['timestamp'] if 'timestamp' in current_candle else current_candle.name,
                            high_price=current_candle['high'],
                            low_price=current_candle['low'],
                            block_type='bearish',
                            formation_candle_index=i,
                            strength=impulse_strength
                        ))
            
            return sorted(order_blocks, key=lambda x: x.strength, reverse=True)
            
        except Exception as e:
            logger.error(f"Error in order blocks analysis: {e}")
            return []
    
    def concept_5_breaker_blocks(self, stock_data: pd.DataFrame) -> List[OrderBlock]:
        """
        CONCEPT 5: Breaker Blocks
        - Former resistance becomes support in stocks
        - Former support becomes resistance in stocks
        - Polarity switch detection
        """
        try:
            breaker_blocks = []
            order_blocks = self.concept_4_order_blocks_bullish_bearish(stock_data)
            
            for ob in order_blocks:
                # Check if order block has been broken and is now acting as support/resistance
                ob_index = ob.formation_candle_index
                
                if ob_index < len(stock_data) - 5:  # Need some candles after formation
                    future_data = stock_data.iloc[ob_index+1:]
                    
                    if ob.block_type == 'bullish':
                        # Check if price broke below and then found support
                        broken_below = future_data['low'].min() < ob.low_price
                        if broken_below:
                            # Check for subsequent support
                            support_tests = future_data[future_data['low'] <= ob.high_price * 1.01]
                            support_tests = support_tests[support_tests['low'] >= ob.low_price * 0.99]
                            
                            if len(support_tests) >= 2:
                                breaker = OrderBlock(
                                    timestamp=ob.timestamp,
                                    high_price=ob.high_price,
                                    low_price=ob.low_price,
                                    block_type='bullish_breaker',
                                    formation_candle_index=ob.formation_candle_index,
                                    strength=ob.strength * 1.2,  # Breakers are stronger
                                    is_breaker=True
                                )
                                breaker_blocks.append(breaker)
                    
                    elif ob.block_type == 'bearish':
                        # Check if price broke above and then found resistance
                        broken_above = future_data['high'].max() > ob.high_price
                        if broken_above:
                            # Check for subsequent resistance
                            resistance_tests = future_data[future_data['high'] >= ob.low_price * 0.99]
                            resistance_tests = resistance_tests[resistance_tests['high'] <= ob.high_price * 1.01]
                            
                            if len(resistance_tests) >= 2:
                                breaker = OrderBlock(
                                    timestamp=ob.timestamp,
                                    high_price=ob.high_price,
                                    low_price=ob.low_price,
                                    block_type='bearish_breaker',
                                    formation_candle_index=ob.formation_candle_index,
                                    strength=ob.strength * 1.2,
                                    is_breaker=True
                                )
                                breaker_blocks.append(breaker)
            
            return breaker_blocks
            
        except Exception as e:
            logger.error(f"Error in breaker blocks analysis: {e}")
            return []
    
    def concept_6_fair_value_gaps_fvg_imbalances(self, stock_data: pd.DataFrame) -> List[FairValueGap]:
        """
        CONCEPT 6: Fair Value Gaps (FVG) / Imbalances
        - 3-candle gap patterns in stock charts
        - Bullish FVG: gap up in stock price
        - Bearish FVG: gap down in stock price
        - 50% mitigation rule for stocks
        """
        try:
            fvgs = []
            
            if len(stock_data) < 3:
                return fvgs
            
            for i in range(1, len(stock_data) - 1):
                prev_candle = stock_data.iloc[i-1]
                current_candle = stock_data.iloc[i]
                next_candle = stock_data.iloc[i+1]
                
                # Bullish FVG: previous high < next low (gap up)
                if prev_candle['high'] < next_candle['low']:
                    gap_size = next_candle['low'] - prev_candle['high']
                    gap_percentage = gap_size / prev_candle['high']
                    
                    if gap_percentage > 0.001:  # Minimum 0.1% gap
                        fvg = FairValueGap(
                            timestamp=current_candle['timestamp'] if 'timestamp' in current_candle else current_candle.name,
                            gap_high=next_candle['low'],
                            gap_low=prev_candle['high'],
                            gap_type='bullish',
                            gap_size=gap_size,
                            mitigation_level=(next_candle['low'] + prev_candle['high']) / 2
                        )
                        fvgs.append(fvg)
                
                # Bearish FVG: previous low > next high (gap down)
                elif prev_candle['low'] > next_candle['high']:
                    gap_size = prev_candle['low'] - next_candle['high']
                    gap_percentage = gap_size / prev_candle['low']
                    
                    if gap_percentage > 0.001:  # Minimum 0.1% gap
                        fvg = FairValueGap(
                            timestamp=current_candle['timestamp'] if 'timestamp' in current_candle else current_candle.name,
                            gap_high=prev_candle['low'],
                            gap_low=next_candle['high'],
                            gap_type='bearish',
                            gap_size=gap_size,
                            mitigation_level=(prev_candle['low'] + next_candle['high']) / 2
                        )
                        fvgs.append(fvg)
            
            # Check for mitigation
            for fvg in fvgs:
                fvg.is_mitigated = self._check_fvg_mitigation(stock_data, fvg)
            
            return fvgs
            
        except Exception as e:
            logger.error(f"Error in FVG analysis: {e}")
            return []
    
    # Helper methods
    def _find_swing_highs(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Find swing high points"""
        swing_highs = []
        
        for i in range(self.swing_lookback, len(df) - self.swing_lookback):
            current_high = df['high'].iloc[i]
            is_swing_high = True
            
            # Check if current point is higher than surrounding points
            for j in range(i - self.swing_lookback, i + self.swing_lookback + 1):
                if j != i and df['high'].iloc[j] >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                strength = self._calculate_swing_strength(df, i, 'high')
                swing_highs.append(SwingPoint(
                    timestamp=df.iloc[i]['timestamp'] if 'timestamp' in df.columns else df.index[i],
                    price=current_high,
                    swing_type='high',
                    index=i,
                    strength=strength
                ))
        
        return swing_highs
    
    def _find_swing_lows(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Find swing low points"""
        swing_lows = []
        
        for i in range(self.swing_lookback, len(df) - self.swing_lookback):
            current_low = df['low'].iloc[i]
            is_swing_low = True
            
            # Check if current point is lower than surrounding points
            for j in range(i - self.swing_lookback, i + self.swing_lookback + 1):
                if j != i and df['low'].iloc[j] <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                strength = self._calculate_swing_strength(df, i, 'low')
                swing_lows.append(SwingPoint(
                    timestamp=df.iloc[i]['timestamp'] if 'timestamp' in df.columns else df.index[i],
                    price=current_low,
                    swing_type='low',
                    index=i,
                    strength=strength
                ))
        
        return swing_lows
    
    def _calculate_swing_strength(self, df: pd.DataFrame, index: int, swing_type: str) -> float:
        """Calculate the strength of a swing point"""
        try:
            if swing_type == 'high':
                # Volume and range at the swing high
                volume_ratio = df['volume'].iloc[index] / df['volume'].rolling(20).mean().iloc[index]
                price_range = (df['high'].iloc[index] - df['low'].iloc[index]) / df['close'].iloc[index]
            else:
                # Volume and range at the swing low
                volume_ratio = df['volume'].iloc[index] / df['volume'].rolling(20).mean().iloc[index]
                price_range = (df['high'].iloc[index] - df['low'].iloc[index]) / df['close'].iloc[index]
            
            # Combine volume and range for strength
            strength = min(volume_ratio * price_range * 10, 1.0)  # Cap at 1.0
            return max(strength, 0.1)  # Minimum strength
            
        except:
            return 0.5  # Default strength
    
    def _classify_current_structure(self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]) -> str:
        """Classify current market structure"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'insufficient_data'
        
        recent_highs = swing_highs[-2:]
        recent_lows = swing_lows[-2:]
        
        # Compare recent swing points
        if recent_highs[-1].price > recent_highs[-2].price and recent_lows[-1].price > recent_lows[-2].price:
            return 'bullish_structure'  # HH and HL
        elif recent_highs[-1].price < recent_highs[-2].price and recent_lows[-1].price < recent_lows[-2].price:
            return 'bearish_structure'  # LH and LL
        else:
            return 'mixed_structure'
    
    def _detect_structure_breaks(self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]) -> List[Dict]:
        """Detect market structure breaks"""
        breaks = []
        
        # Detect breaks in bullish structure (break of recent swing low)
        if len(swing_lows) >= 2:
            for i in range(1, len(swing_lows)):
                if swing_lows[i].price < swing_lows[i-1].price:
                    breaks.append({
                        'type': 'bearish_break',
                        'timestamp': swing_lows[i].timestamp,
                        'price': swing_lows[i].price,
                        'previous_low': swing_lows[i-1].price
                    })
        
        # Detect breaks in bearish structure (break of recent swing high)
        if len(swing_highs) >= 2:
            for i in range(1, len(swing_highs)):
                if swing_highs[i].price > swing_highs[i-1].price:
                    breaks.append({
                        'type': 'bullish_break',
                        'timestamp': swing_highs[i].timestamp,
                        'price': swing_highs[i].price,
                        'previous_high': swing_highs[i-1].price
                    })
        
        return breaks
    
    def _determine_trend_direction(self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]) -> str:
        """Determine overall trend direction"""
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return 'sideways'
        
        # Analyze last 3 swing points
        recent_highs = [h.price for h in swing_highs[-3:]]
        recent_lows = [l.price for l in swing_lows[-3:]]
        
        highs_trend = 'up' if recent_highs[-1] > recent_highs[0] else 'down'
        lows_trend = 'up' if recent_lows[-1] > recent_lows[0] else 'down'
        
        if highs_trend == 'up' and lows_trend == 'up':
            return 'uptrend'
        elif highs_trend == 'down' and lows_trend == 'down':
            return 'downtrend'
        else:
            return 'sideways'
    
    def _calculate_structure_confidence(self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]) -> float:
        """Calculate confidence in market structure analysis"""
        if not swing_highs or not swing_lows:
            return 0.0
        
        # Base confidence on number of swing points and their strength
        total_points = len(swing_highs) + len(swing_lows)
        avg_strength = np.mean([h.strength for h in swing_highs] + [l.strength for l in swing_lows])
        
        confidence = min((total_points / 10) * avg_strength, 1.0)
        return confidence
    
    def _is_significant_level(self, df: pd.DataFrame, price: float) -> bool:
        """Check if a price level is significant"""
        current_price = df['close'].iloc[-1]
        distance = abs(price - current_price) / current_price
        
        # Level should be within reasonable distance but not too close
        return 0.005 < distance < 0.1  # Between 0.5% and 10%
    
    def _find_equal_levels(self, swing_points: List[SwingPoint], point_type: str) -> List[List[SwingPoint]]:
        """Find equal levels among swing points"""
        if len(swing_points) < 2:
            return []
        
        equal_levels = []
        used_points = set()
        
        for i, point1 in enumerate(swing_points):
            if i in used_points:
                continue
                
            level_group = [point1]
            used_points.add(i)
            
            for j, point2 in enumerate(swing_points[i+1:], i+1):
                if j in used_points:
                    continue
                    
                # Check if points are at equal levels
                price_diff = abs(point1.price - point2.price) / point1.price
                if price_diff <= self.liquidity_threshold:
                    level_group.append(point2)
                    used_points.add(j)
            
            if len(level_group) >= 2:
                equal_levels.append(level_group)
        
        return equal_levels
    
    def _find_trendline_liquidity(self, df: pd.DataFrame) -> List[LiquidityPool]:
        """Find trendline liquidity pools"""
        # Simplified trendline liquidity detection
        # This would need more sophisticated trendline analysis in production
        pools = []
        
        try:
            # Find upward trendlines (connecting swing lows)
            swing_lows = self._find_swing_lows(df)
            if len(swing_lows) >= 2:
                for i in range(len(swing_lows) - 1):
                    low1, low2 = swing_lows[i], swing_lows[i+1]
                    
                    # Calculate trendline level at current time
                    time_diff = (df.index[-1] - low2.index) if hasattr(df.index[-1], '__sub__') else 1
                    price_diff = low2.price - low1.price
                    time_span = low2.index - low1.index if time_span != 0 else 1
                    
                    if time_span > 0:
                        trendline_price = low2.price + (price_diff / time_span) * time_diff
                        
                        pools.append(LiquidityPool(
                            timestamp=datetime.now(),
                            price_level=trendline_price,
                            pool_type='upward_trendline',
                            strength=0.6,
                            touches=2
                        ))
            
        except Exception as e:
            logger.error(f"Error in trendline liquidity: {e}")
        
        return pools
    
    def _check_fvg_mitigation(self, df: pd.DataFrame, fvg: FairValueGap) -> bool:
        """Check if FVG has been mitigated"""
        try:
            # Find data after FVG formation
            fvg_timestamp = fvg.timestamp
            
            if hasattr(df, 'timestamp'):
                future_data = df[df['timestamp'] > fvg_timestamp]
            else:
                # If no timestamp column, assume chronological order
                fvg_index = df[df.index == fvg_timestamp].index
                if len(fvg_index) > 0:
                    future_data = df.iloc[fvg_index[0]+1:]
                else:
                    return False
            
            if future_data.empty:
                return False
            
            # Check if price has returned to 50% of the gap
            if fvg.gap_type == 'bullish':
                # Price should come back down to mitigation level
                return future_data['low'].min() <= fvg.mitigation_level
            else:
                # Price should come back up to mitigation level
                return future_data['high'].max() >= fvg.mitigation_level
                
        except Exception:
            return False

    def concept_7_rejection_blocks(self, stock_data: pd.DataFrame) -> List[Dict]:
        """
        CONCEPT 7: Rejection Blocks
        - Strong rejection candles in stocks
        - Wick analysis for institutional rejection
        - Volume confirmation for rejection validity
        """
        try:
            rejection_blocks = []
            
            if len(stock_data) < 5:
                return rejection_blocks
            
            for i in range(2, len(stock_data) - 2):
                candle = stock_data.iloc[i]
                
                # Calculate wick sizes
                upper_wick = candle['high'] - max(candle['open'], candle['close'])
                lower_wick = min(candle['open'], candle['close']) - candle['low']
                body_size = abs(candle['close'] - candle['open'])
                total_range = candle['high'] - candle['low']
                
                # Check for significant rejection (wick > 2x body size)
                if total_range > 0:
                    upper_wick_ratio = upper_wick / total_range
                    lower_wick_ratio = lower_wick / total_range
                    
                    # Upper rejection (bearish)
                    if upper_wick_ratio > 0.6 and upper_wick > body_size * 2:
                        volume_strength = self._calculate_volume_strength(stock_data, i)
                        rejection_blocks.append({
                            'timestamp': candle['timestamp'] if 'timestamp' in candle else candle.name,
                            'rejection_type': 'bearish',
                            'rejection_price': candle['high'],
                            'support_level': candle['low'],
                            'wick_ratio': upper_wick_ratio,
                            'volume_strength': volume_strength,
                            'strength': upper_wick_ratio * volume_strength
                        })
                    
                    # Lower rejection (bullish)
                    if lower_wick_ratio > 0.6 and lower_wick > body_size * 2:
                        volume_strength = self._calculate_volume_strength(stock_data, i)
                        rejection_blocks.append({
                            'timestamp': candle['timestamp'] if 'timestamp' in candle else candle.name,
                            'rejection_type': 'bullish',
                            'rejection_price': candle['low'],
                            'resistance_level': candle['high'],
                            'wick_ratio': lower_wick_ratio,
                            'volume_strength': volume_strength,
                            'strength': lower_wick_ratio * volume_strength
                        })
            
            return sorted(rejection_blocks, key=lambda x: x['strength'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error in rejection blocks analysis: {e}")
            return []

    def concept_8_mitigation_blocks(self, stock_data: pd.DataFrame) -> List[Dict]:
        """
        CONCEPT 8: Mitigation Blocks
        - Price returning to mitigate inefficiencies
        - Partial vs full mitigation in stocks
        - Time-based mitigation analysis
        """
        try:
            mitigation_blocks = []
            fvgs = self.concept_6_fair_value_gaps_fvg_imbalances(stock_data)
            order_blocks = self.concept_4_order_blocks_bullish_bearish(stock_data)
            
            # Check FVG mitigation
            for fvg in fvgs:
                mitigation_info = self._analyze_fvg_mitigation(stock_data, fvg)
                if mitigation_info:
                    mitigation_blocks.append(mitigation_info)
            
            # Check Order Block mitigation
            for ob in order_blocks:
                mitigation_info = self._analyze_ob_mitigation(stock_data, ob)
                if mitigation_info:
                    mitigation_blocks.append(mitigation_info)
            
            return sorted(mitigation_blocks, key=lambda x: x.get('mitigation_percentage', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Error in mitigation blocks analysis: {e}")
            return []

    def concept_9_supply_demand_zones(self, stock_data: pd.DataFrame) -> List[Dict]:
        """
        CONCEPT 9: Supply & Demand Zones
        - Fresh vs tested zones in stocks
        - Zone strength classification
        - Institutional supply/demand levels
        """
        try:
            zones = []
            
            # Find supply zones (areas where price dropped from)
            swing_highs = self._find_swing_highs(stock_data)
            for high in swing_highs:
                # Look for the base/zone before the high
                zone_info = self._identify_supply_zone(stock_data, high)
                if zone_info:
                    zones.append(zone_info)
            
            # Find demand zones (areas where price rallied from)
            swing_lows = self._find_swing_lows(stock_data)
            for low in swing_lows:
                # Look for the base/zone before the low
                zone_info = self._identify_demand_zone(stock_data, low)
                if zone_info:
                    zones.append(zone_info)
            
            return sorted(zones, key=lambda x: x.get('strength', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Error in supply/demand zones analysis: {e}")
            return []

    def concept_10_premium_discount_ote(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 10: Premium & Discount (Optimal Trade Entry - OTE)
        - Premium zones (above 50% of range)
        - Discount zones (below 50% of range)
        - Optimal Trade Entry calculations
        """
        try:
            if len(stock_data) < 20:
                return {'error': 'Insufficient data'}
            
            # Calculate daily/weekly range
            recent_data = stock_data.tail(20)  # Last 20 periods
            range_high = recent_data['high'].max()
            range_low = recent_data['low'].min()
            current_price = stock_data['close'].iloc[-1]
            
            range_size = range_high - range_low
            price_position = (current_price - range_low) / range_size if range_size > 0 else 0.5
            
            # Calculate OTE levels (62%-79% retracement zones)
            bullish_ote_high = range_low + (range_size * 0.79)
            bullish_ote_low = range_low + (range_size * 0.62)
            
            bearish_ote_high = range_high - (range_size * 0.62)
            bearish_ote_low = range_high - (range_size * 0.79)
            
            return {
                'range_high': range_high,
                'range_low': range_low,
                'current_price': current_price,
                'price_position_percentage': price_position * 100,
                'market_bias': 'premium' if price_position > 0.5 else 'discount',
                'premium_zone': price_position > 0.7,
                'discount_zone': price_position < 0.3,
                'optimal_trade_entry': {
                    'bullish_ote_high': bullish_ote_high,
                    'bullish_ote_low': bullish_ote_low,
                    'bearish_ote_high': bearish_ote_high,
                    'bearish_ote_low': bearish_ote_low,
                    'in_bullish_ote': bullish_ote_low <= current_price <= bullish_ote_high,
                    'in_bearish_ote': bearish_ote_low <= current_price <= bearish_ote_high
                }
            }
            
        except Exception as e:
            logger.error(f"Error in premium/discount analysis: {e}")
            return {'error': str(e)}

    def concept_11_dealing_ranges(self, stock_data: pd.DataFrame) -> List[Dict]:
        """
        CONCEPT 11: Dealing Ranges
        - Consolidation ranges in stocks
        - Range high/low identification
        - Breakout vs fakeout analysis
        """
        try:
            ranges = []
            
            # Use rolling window to identify consolidation periods
            window_size = 20
            
            for i in range(window_size, len(stock_data) - window_size):
                window_data = stock_data.iloc[i-window_size:i+window_size]
                
                # Calculate range statistics
                range_high = window_data['high'].max()
                range_low = window_data['low'].min()
                range_size = range_high - range_low
                avg_price = window_data['close'].mean()
                
                # Check if it's a consolidation (low volatility)
                price_std = window_data['close'].std()
                volatility_ratio = price_std / avg_price
                
                if volatility_ratio < 0.02:  # Low volatility threshold
                    range_info = {
                        'start_timestamp': window_data.iloc[0]['timestamp'] if 'timestamp' in window_data.columns else window_data.index[0],
                        'end_timestamp': window_data.iloc[-1]['timestamp'] if 'timestamp' in window_data.columns else window_data.index[-1],
                        'range_high': range_high,
                        'range_low': range_low,
                        'range_size': range_size,
                        'range_size_percentage': range_size / avg_price,
                        'consolidation_strength': 1 - volatility_ratio,
                        'volume_profile': self._analyze_range_volume(window_data)
                    }
                    ranges.append(range_info)
            
            return ranges
            
        except Exception as e:
            logger.error(f"Error in dealing ranges analysis: {e}")
            return []

    def concept_12_swing_highs_swing_lows(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 12: Swing Highs & Swing Lows
        - Fractal-based swing identification
        - Multi-timeframe swing analysis
        - Swing failure patterns
        """
        try:
            swing_highs = self._find_swing_highs(stock_data)
            swing_lows = self._find_swing_lows(stock_data)
            
            # Analyze swing patterns
            swing_analysis = {
                'swing_highs': [
                    {
                        'timestamp': sh.timestamp,
                        'price': sh.price,
                        'strength': sh.strength,
                        'index': sh.index
                    } for sh in swing_highs
                ],
                'swing_lows': [
                    {
                        'timestamp': sl.timestamp,
                        'price': sl.price,
                        'strength': sl.strength,
                        'index': sl.index
                    } for sl in swing_lows
                ],
                'swing_failures': self._detect_swing_failures(swing_highs, swing_lows),
                'double_tops': self._detect_double_tops(swing_highs),
                'double_bottoms': self._detect_double_bottoms(swing_lows),
                'swing_count': {
                    'highs': len(swing_highs),
                    'lows': len(swing_lows),
                    'total': len(swing_highs) + len(swing_lows)
                }
            }
            
            return swing_analysis
            
        except Exception as e:
            logger.error(f"Error in swing analysis: {e}")
            return {'error': str(e)}

    # Additional helper methods for concepts 7-12
    def _calculate_volume_strength(self, df: pd.DataFrame, index: int) -> float:
        """Calculate volume strength at given index"""
        try:
            current_volume = df['volume'].iloc[index]
            avg_volume = df['volume'].rolling(20).mean().iloc[index]
            return min(current_volume / avg_volume, 3.0) / 3.0 if avg_volume > 0 else 0.5
        except:
            return 0.5

    def _analyze_fvg_mitigation(self, df: pd.DataFrame, fvg: FairValueGap) -> Optional[Dict]:
        """Analyze FVG mitigation details"""
        try:
            # Find future price action after FVG
            fvg_timestamp = fvg.timestamp
            future_data = df[df.index > fvg_timestamp] if hasattr(df.index[0], '__lt__') else df.tail(10)
            
            if future_data.empty:
                return None
            
            mitigation_percentage = 0
            mitigation_type = 'none'
            
            if fvg.gap_type == 'bullish':
                lowest_price = future_data['low'].min()
                if lowest_price <= fvg.gap_low:
                    mitigation_percentage = 100
                    mitigation_type = 'full'
                elif lowest_price <= fvg.mitigation_level:
                    mitigation_percentage = 50
                    mitigation_type = 'partial'
            else:
                highest_price = future_data['high'].max()
                if highest_price >= fvg.gap_high:
                    mitigation_percentage = 100
                    mitigation_type = 'full'
                elif highest_price >= fvg.mitigation_level:
                    mitigation_percentage = 50
                    mitigation_type = 'partial'
            
            if mitigation_percentage > 0:
                return {
                    'type': 'fvg_mitigation',
                    'original_pattern': 'fair_value_gap',
                    'timestamp': fvg.timestamp,
                    'mitigation_percentage': mitigation_percentage,
                    'mitigation_type': mitigation_type,
                    'gap_type': fvg.gap_type,
                    'strength': mitigation_percentage / 100
                }
            
            return None
            
        except Exception:
            return None

    def _analyze_ob_mitigation(self, df: pd.DataFrame, ob: OrderBlock) -> Optional[Dict]:
        """Analyze Order Block mitigation details"""
        try:
            ob_index = ob.formation_candle_index
            if ob_index >= len(df) - 1:
                return None
            
            future_data = df.iloc[ob_index+1:]
            if future_data.empty:
                return None
            
            mitigation_percentage = 0
            mitigation_type = 'none'
            
            if ob.block_type == 'bullish':
                # Check if price returned to OB zone
                touches = future_data[(future_data['low'] <= ob.high_price) & (future_data['low'] >= ob.low_price)]
                if len(touches) > 0:
                    lowest_in_zone = touches['low'].min()
                    zone_size = ob.high_price - ob.low_price
                    penetration = (ob.high_price - lowest_in_zone) / zone_size if zone_size > 0 else 0
                    mitigation_percentage = min(penetration * 100, 100)
                    mitigation_type = 'full' if penetration > 0.8 else 'partial'
            
            else:  # bearish
                touches = future_data[(future_data['high'] >= ob.low_price) & (future_data['high'] <= ob.high_price)]
                if len(touches) > 0:
                    highest_in_zone = touches['high'].max()
                    zone_size = ob.high_price - ob.low_price
                    penetration = (highest_in_zone - ob.low_price) / zone_size if zone_size > 0 else 0
                    mitigation_percentage = min(penetration * 100, 100)
                    mitigation_type = 'full' if penetration > 0.8 else 'partial'
            
            if mitigation_percentage > 0:
                return {
                    'type': 'order_block_mitigation',
                    'original_pattern': 'order_block',
                    'timestamp': ob.timestamp,
                    'mitigation_percentage': mitigation_percentage,
                    'mitigation_type': mitigation_type,
                    'block_type': ob.block_type,
                    'strength': mitigation_percentage / 100
                }
            
            return None
            
        except Exception:
            return None

    def _identify_supply_zone(self, df: pd.DataFrame, swing_high: SwingPoint) -> Optional[Dict]:
        """Identify supply zone before swing high"""
        try:
            high_index = swing_high.index
            
            # Look back for the base/consolidation before the high
            lookback_data = df.iloc[max(0, high_index-10):high_index]
            
            if len(lookback_data) < 3:
                return None
            
            # Find the consolidation area
            zone_high = lookback_data['high'].max()
            zone_low = lookback_data['low'].min()
            zone_volume = lookback_data['volume'].sum()
            
            return {
                'zone_type': 'supply',
                'zone_high': zone_high,
                'zone_low': zone_low,
                'swing_high': swing_high.price,
                'timestamp': swing_high.timestamp,
                'strength': swing_high.strength,
                'zone_volume': zone_volume,
                'is_fresh': True,  # Would need to check for subsequent tests
                'distance_from_current': abs(zone_high - df['close'].iloc[-1]) / df['close'].iloc[-1]
            }
            
        except Exception:
            return None

    def _identify_demand_zone(self, df: pd.DataFrame, swing_low: SwingPoint) -> Optional[Dict]:
        """Identify demand zone before swing low"""
        try:
            low_index = swing_low.index
            
            # Look back for the base/consolidation before the low
            lookback_data = df.iloc[max(0, low_index-10):low_index]
            
            if len(lookback_data) < 3:
                return None
            
            # Find the consolidation area
            zone_high = lookback_data['high'].max()
            zone_low = lookback_data['low'].min()
            zone_volume = lookback_data['volume'].sum()
            
            return {
                'zone_type': 'demand',
                'zone_high': zone_high,
                'zone_low': zone_low,
                'swing_low': swing_low.price,
                'timestamp': swing_low.timestamp,
                'strength': swing_low.strength,
                'zone_volume': zone_volume,
                'is_fresh': True,  # Would need to check for subsequent tests
                'distance_from_current': abs(zone_low - df['close'].iloc[-1]) / df['close'].iloc[-1]
            }
            
        except Exception:
            return None

    def _analyze_range_volume(self, df: pd.DataFrame) -> Dict:
        """Analyze volume profile within a range"""
        try:
            return {
                'total_volume': df['volume'].sum(),
                'avg_volume': df['volume'].mean(),
                'volume_distribution': 'uniform',  # Simplified
                'poc_price': df.loc[df['volume'].idxmax(), 'close']  # Point of Control
            }
        except:
            return {'total_volume': 0, 'avg_volume': 0}

    def _detect_swing_failures(self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]) -> List[Dict]:
        """Detect swing failure patterns"""
        failures = []
        
        # Failed swing highs (price makes higher high but fails to sustain)
        if len(swing_highs) >= 2:
            for i in range(1, len(swing_highs)):
                current_high = swing_highs[i]
                prev_high = swing_highs[i-1]
                
                if current_high.price > prev_high.price and current_high.strength < prev_high.strength:
                    failures.append({
                        'type': 'failed_swing_high',
                        'timestamp': current_high.timestamp,
                        'price': current_high.price,
                        'previous_high': prev_high.price
                    })
        
        # Failed swing lows
        if len(swing_lows) >= 2:
            for i in range(1, len(swing_lows)):
                current_low = swing_lows[i]
                prev_low = swing_lows[i-1]
                
                if current_low.price < prev_low.price and current_low.strength < prev_low.strength:
                    failures.append({
                        'type': 'failed_swing_low',
                        'timestamp': current_low.timestamp,
                        'price': current_low.price,
                        'previous_low': prev_low.price
                    })
        
        return failures

    def _detect_double_tops(self, swing_highs: List[SwingPoint]) -> List[Dict]:
        """Detect double top patterns"""
        double_tops = []
        
        for i in range(1, len(swing_highs)):
            current_high = swing_highs[i]
            prev_high = swing_highs[i-1]
            
            price_diff = abs(current_high.price - prev_high.price) / prev_high.price
            
            if price_diff < 0.02:  # Within 2% of each other
                double_tops.append({
                    'first_top': {
                        'timestamp': prev_high.timestamp,
                        'price': prev_high.price
                    },
                    'second_top': {
                        'timestamp': current_high.timestamp,
                        'price': current_high.price
                    },
                    'pattern_strength': min(prev_high.strength, current_high.strength)
                })
        
        return double_tops

    def _detect_double_bottoms(self, swing_lows: List[SwingPoint]) -> List[Dict]:
        """Detect double bottom patterns"""
        double_bottoms = []
        
        for i in range(1, len(swing_lows)):
            current_low = swing_lows[i]
            prev_low = swing_lows[i-1]
            
            price_diff = abs(current_low.price - prev_low.price) / prev_low.price
            
            if price_diff < 0.02:  # Within 2% of each other
                double_bottoms.append({
                    'first_bottom': {
                        'timestamp': prev_low.timestamp,
                        'price': prev_low.price
                    },
                    'second_bottom': {
                        'timestamp': current_low.timestamp,
                        'price': current_low.price
                    },
                    'pattern_strength': min(prev_low.strength, current_low.strength)
                })
        
        return double_bottoms

    def concept_13_market_maker_buy_sell_models(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 13: Market Maker Buy & Sell Models
        - Institutional buying patterns
        - Market maker selling patterns
        - Smart money footprints in stocks
        """
        try:
            if len(stock_data) < 20:
                return {'error': 'Insufficient data'}
            
            # Analyze volume and price action patterns
            df = stock_data.copy()
            
            # Calculate volume metrics
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['price_change'] = df['close'].pct_change()
            
            buy_models = []
            sell_models = []
            
            # Market Maker Buy Model characteristics:
            # - High volume, small price moves up (accumulation)
            # - Multiple touches of support with increasing volume
            for i in range(10, len(df)):
                window = df.iloc[i-10:i+1]
                
                # Check for accumulation pattern
                avg_volume_ratio = window['volume_ratio'].mean()
                price_stability = window['price_change'].std()
                net_change = (window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0]
                
                if (avg_volume_ratio > 1.5 and  # High volume
                    price_stability < 0.02 and  # Low volatility
                    -0.02 <= net_change <= 0.05):  # Slight up or sideways
                    
                    buy_models.append({
                        'timestamp': df.iloc[i]['timestamp'] if 'timestamp' in df.columns else df.index[i],
                        'model_type': 'accumulation',
                        'price_level': window['close'].mean(),
                        'volume_strength': avg_volume_ratio,
                        'confidence': min(avg_volume_ratio * (1 - price_stability) * 10, 1.0)
                    })
            
            # Market Maker Sell Model characteristics:
            # - High volume, small price moves down (distribution)
            # - Multiple touches of resistance with increasing volume
            for i in range(10, len(df)):
                window = df.iloc[i-10:i+1]
                
                avg_volume_ratio = window['volume_ratio'].mean()
                price_stability = window['price_change'].std()
                net_change = (window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0]
                
                if (avg_volume_ratio > 1.5 and  # High volume
                    price_stability < 0.02 and  # Low volatility
                    -0.05 <= net_change <= 0.02):  # Slight down or sideways
                    
                    sell_models.append({
                        'timestamp': df.iloc[i]['timestamp'] if 'timestamp' in df.columns else df.index[i],
                        'model_type': 'distribution',
                        'price_level': window['close'].mean(),
                        'volume_strength': avg_volume_ratio,
                        'confidence': min(avg_volume_ratio * (1 - price_stability) * 10, 1.0)
                    })
            
            return {
                'buy_models': sorted(buy_models, key=lambda x: x['confidence'], reverse=True)[:5],
                'sell_models': sorted(sell_models, key=lambda x: x['confidence'], reverse=True)[:5],
                'current_bias': self._determine_mm_bias(df),
                'smart_money_flow': self._calculate_smart_money_flow(df)
            }
            
        except Exception as e:
            logger.error(f"Error in market maker models analysis: {e}")
            return {'error': str(e)}

    def concept_14_market_maker_programs(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 14: Market Maker Sell Programs & Buy Programs
        - Automated institutional programs
        - Program trading detection
        - Volume signature analysis
        """
        try:
            if len(stock_data) < 30:
                return {'error': 'Insufficient data'}
            
            df = stock_data.copy()
            
            # Calculate indicators for program detection
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_spike'] = df['volume'] / df['volume_ma']
            df['price_change'] = df['close'].pct_change()
            df['true_range'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df['atr'] = df['true_range'].rolling(14).mean()
            df['volatility_ratio'] = df['true_range'] / df['atr']
            
            programs = []
            
            # Detect sell programs: High volume + sustained selling pressure
            for i in range(20, len(df)):
                window = df.iloc[i-5:i+1]  # 6-period window
                
                # Characteristics of sell program:
                volume_increase = window['volume_spike'].mean() > 2.0
                consistent_selling = (window['price_change'] < 0).sum() >= 4
                low_volatility = window['volatility_ratio'].mean() < 0.8
                
                if volume_increase and consistent_selling and low_volatility:
                    programs.append({
                        'timestamp': df.iloc[i]['timestamp'] if 'timestamp' in df.columns else df.index[i],
                        'program_type': 'sell_program',
                        'duration_periods': 6,
                        'avg_volume_spike': window['volume_spike'].mean(),
                        'price_impact': window['price_change'].sum(),
                        'strength': window['volume_spike'].mean() * abs(window['price_change'].sum())
                    })
            
            # Detect buy programs: High volume + sustained buying pressure
            for i in range(20, len(df)):
                window = df.iloc[i-5:i+1]
                
                volume_increase = window['volume_spike'].mean() > 2.0
                consistent_buying = (window['price_change'] > 0).sum() >= 4
                low_volatility = window['volatility_ratio'].mean() < 0.8
                
                if volume_increase and consistent_buying and low_volatility:
                    programs.append({
                        'timestamp': df.iloc[i]['timestamp'] if 'timestamp' in df.columns else df.index[i],
                        'program_type': 'buy_program',
                        'duration_periods': 6,
                        'avg_volume_spike': window['volume_spike'].mean(),
                        'price_impact': window['price_change'].sum(),
                        'strength': window['volume_spike'].mean() * window['price_change'].sum()
                    })
            
            return {
                'detected_programs': sorted(programs, key=lambda x: x['strength'], reverse=True),
                'program_summary': {
                    'total_programs': len(programs),
                    'buy_programs': len([p for p in programs if p['program_type'] == 'buy_program']),
                    'sell_programs': len([p for p in programs if p['program_type'] == 'sell_program'])
                },
                'current_program_activity': self._assess_current_program_activity(df)
            }
            
        except Exception as e:
            logger.error(f"Error in market maker programs analysis: {e}")
            return {'error': str(e)}

    def concept_15_judas_swing(self, stock_data: pd.DataFrame) -> List[Dict]:
        """
        CONCEPT 15: Judas Swing (false breakout at sessions open)
        - False breakout at market open
        - Pre-market vs regular hours analysis
        - Stop hunt identification
        """
        try:
            judas_swings = []
            
            if 'session' not in stock_data.columns:
                # Add session detection if not present
                stock_data = self.detect_market_sessions(stock_data)
            
            # Look for false breakouts at session opens
            session_opens = stock_data[stock_data['session'] == 'market_open']
            
            for i, open_candle in session_opens.iterrows():
                # Get pre-market data
                premarket_data = stock_data[
                    (stock_data['session'] == 'premarket') & 
                    (stock_data.index < i)
                ].tail(10)
                
                if len(premarket_data) < 3:
                    continue
                
                # Identify pre-market range
                pm_high = premarket_data['high'].max()
                pm_low = premarket_data['low'].min()
                
                # Check for false breakout above pre-market high
                if open_candle['high'] > pm_high:
                    # Look for subsequent failure and reversal
                    future_data = stock_data[stock_data.index > i].head(10)
                    
                    if len(future_data) > 0:
                        failed_breakout = future_data['low'].min() < pm_high
                        reversal_strength = (open_candle['high'] - future_data['close'].iloc[-1]) / open_candle['high']
                        
                        if failed_breakout and reversal_strength > 0.01:
                            judas_swings.append({
                                'timestamp': open_candle['timestamp'] if 'timestamp' in open_candle else i,
                                'swing_type': 'bearish_judas',
                                'false_breakout_price': open_candle['high'],
                                'premarket_high': pm_high,
                                'reversal_target': future_data['low'].min(),
                                'strength': reversal_strength,
                                'session': 'market_open'
                            })
                
                # Check for false breakout below pre-market low
                if open_candle['low'] < pm_low:
                    future_data = stock_data[stock_data.index > i].head(10)
                    
                    if len(future_data) > 0:
                        failed_breakout = future_data['high'].max() > pm_low
                        reversal_strength = (future_data['close'].iloc[-1] - open_candle['low']) / open_candle['low']
                        
                        if failed_breakout and reversal_strength > 0.01:
                            judas_swings.append({
                                'timestamp': open_candle['timestamp'] if 'timestamp' in open_candle else i,
                                'swing_type': 'bullish_judas',
                                'false_breakout_price': open_candle['low'],
                                'premarket_low': pm_low,
                                'reversal_target': future_data['high'].max(),
                                'strength': reversal_strength,
                                'session': 'market_open'
                            })
            
            return sorted(judas_swings, key=lambda x: x['strength'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error in Judas swing analysis: {e}")
            return []

    def concept_16_turtle_soup(self, stock_data: pd.DataFrame) -> List[Dict]:
        """
        CONCEPT 16: Turtle Soup (stop-hunt strategy)
        - 20-day high/low breakout failures
        - Stop hunting patterns
        - Reversal after false breakout
        """
        try:
            turtle_soup_signals = []
            
            if len(stock_data) < 25:
                return turtle_soup_signals
            
            # Calculate 20-day high/low
            stock_data['high_20'] = stock_data['high'].rolling(20).max()
            stock_data['low_20'] = stock_data['low'].rolling(20).min()
            
            for i in range(20, len(stock_data) - 5):
                current = stock_data.iloc[i]
                
                # Turtle Soup Long: Break below 20-day low then reverse
                if current['low'] < current['low_20']:
                    # Look for reversal in next few periods
                    future_data = stock_data.iloc[i+1:i+6]
                    
                    if len(future_data) > 0:
                        reversal_high = future_data['high'].max()
                        breakout_size = current['low_20'] - current['low']
                        reversal_size = reversal_high - current['low']
                        
                        # Check if reversal is significant
                        if reversal_size > breakout_size * 2:  # Reversal should be at least 2x breakout
                            turtle_soup_signals.append({
                                'timestamp': current['timestamp'] if 'timestamp' in current else current.name,
                                'signal_type': 'turtle_soup_long',
                                'breakout_price': current['low'],
                                'twenty_day_level': current['low_20'],
                                'reversal_target': reversal_high,
                                'breakout_size': breakout_size,
                                'reversal_size': reversal_size,
                                'strength': reversal_size / breakout_size,
                                'stop_loss': current['low'] - (breakout_size * 0.5),
                                'take_profit': current['low'] + (reversal_size * 0.618)
                            })
                
                # Turtle Soup Short: Break above 20-day high then reverse
                if current['high'] > current['high_20']:
                    future_data = stock_data.iloc[i+1:i+6]
                    
                    if len(future_data) > 0:
                        reversal_low = future_data['low'].min()
                        breakout_size = current['high'] - current['high_20']
                        reversal_size = current['high'] - reversal_low
                        
                        if reversal_size > breakout_size * 2:
                            turtle_soup_signals.append({
                                'timestamp': current['timestamp'] if 'timestamp' in current else current.name,
                                'signal_type': 'turtle_soup_short',
                                'breakout_price': current['high'],
                                'twenty_day_level': current['high_20'],
                                'reversal_target': reversal_low,
                                'breakout_size': breakout_size,
                                'reversal_size': reversal_size,
                                'strength': reversal_size / breakout_size,
                                'stop_loss': current['high'] + (breakout_size * 0.5),
                                'take_profit': current['high'] - (reversal_size * 0.618)
                            })
            
            return sorted(turtle_soup_signals, key=lambda x: x['strength'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error in turtle soup analysis: {e}")
            return []

    def concept_17_power_of_3(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 17: Power of 3 (Accumulation – Manipulation – Distribution)
        - Accumulation phase detection
        - Manipulation (stop hunts)
        - Distribution phase identification
        """
        try:
            if len(stock_data) < 50:
                return {'error': 'Insufficient data for Power of 3 analysis'}
            
            df = stock_data.copy()
            
            # Calculate indicators for phase detection
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['price_change'] = df['close'].pct_change()
            df['volatility'] = df['price_change'].rolling(10).std()
            df['true_range'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            
            phases = []
            
            # Detect phases using 30-period windows
            window_size = 30
            
            for i in range(window_size, len(df) - window_size, 10):  # Step by 10 to avoid overlap
                window = df.iloc[i-window_size:i]
                
                avg_volume_ratio = window['volume_ratio'].mean()
                price_range = (window['high'].max() - window['low'].min()) / window['close'].mean()
                net_change = (window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0]
                avg_volatility = window['volatility'].mean()
                
                # Accumulation: High volume, low price movement, slight upward bias
                if (avg_volume_ratio > 1.2 and price_range < 0.1 and 
                    -0.02 <= net_change <= 0.05 and avg_volatility < 0.02):
                    phases.append({
                        'phase': 'accumulation',
                        'start_timestamp': window.iloc[0]['timestamp'] if 'timestamp' in window.columns else window.index[0],
                        'end_timestamp': window.iloc[-1]['timestamp'] if 'timestamp' in window.columns else window.index[-1],
                        'avg_volume_ratio': avg_volume_ratio,
                        'price_range': price_range,
                        'net_change': net_change,
                        'strength': avg_volume_ratio * (1 - abs(net_change))
                    })
                
                # Distribution: High volume, low price movement, slight downward bias
                elif (avg_volume_ratio > 1.2 and price_range < 0.1 and 
                      -0.05 <= net_change <= 0.02 and avg_volatility < 0.02):
                    phases.append({
                        'phase': 'distribution',
                        'start_timestamp': window.iloc[0]['timestamp'] if 'timestamp' in window.columns else window.index[0],
                        'end_timestamp': window.iloc[-1]['timestamp'] if 'timestamp' in window.columns else window.index[-1],
                        'avg_volume_ratio': avg_volume_ratio,
                        'price_range': price_range,
                        'net_change': net_change,
                        'strength': avg_volume_ratio * abs(net_change)
                    })
                
                # Manipulation: Sudden volume spikes with quick reversals
                elif (avg_volume_ratio > 2.0 and avg_volatility > 0.03):
                    # Check for quick reversal
                    reversal_detected = self._check_manipulation_reversal(window)
                    if reversal_detected:
                        phases.append({
                            'phase': 'manipulation',
                            'start_timestamp': window.iloc[0]['timestamp'] if 'timestamp' in window.columns else window.index[0],
                            'end_timestamp': window.iloc[-1]['timestamp'] if 'timestamp' in window.columns else window.index[-1],
                            'avg_volume_ratio': avg_volume_ratio,
                            'price_range': price_range,
                            'net_change': net_change,
                            'strength': avg_volume_ratio * avg_volatility
                        })
            
            # Analyze current phase
            current_phase = self._determine_current_power_of_3_phase(df.tail(30))
            
            return {
                'detected_phases': phases,
                'current_phase': current_phase,
                'phase_cycle_analysis': self._analyze_phase_cycles(phases),
                'next_expected_phase': self._predict_next_phase(phases)
            }
            
        except Exception as e:
            logger.error(f"Error in Power of 3 analysis: {e}")
            return {'error': str(e)}

    def concept_18_optimal_trade_entry(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 18: Optimal Trade Entry (retracement into 62%-79% zone)
        - Fibonacci retracement levels
        - Golden zone entries (62%-79%)
        - Confluence with other ICT concepts
        """
        try:
            if len(stock_data) < 20:
                return {'error': 'Insufficient data'}
            
            # Find recent significant moves for retracement analysis
            swing_highs = self._find_swing_highs(stock_data)
            swing_lows = self._find_swing_lows(stock_data)
            
            ote_opportunities = []
            
            # Analyze bullish OTE (retracement in uptrend)
            if len(swing_lows) >= 1 and len(swing_highs) >= 1:
                recent_low = swing_lows[-1]
                recent_high = swing_highs[-1]
                
                if recent_high.timestamp > recent_low.timestamp:  # Uptrend
                    move_size = recent_high.price - recent_low.price
                    
                    # Calculate Fibonacci levels
                    fib_levels = {
                        '23.6%': recent_high.price - (move_size * 0.236),
                        '38.2%': recent_high.price - (move_size * 0.382),
                        '50%': recent_high.price - (move_size * 0.5),
                        '61.8%': recent_high.price - (move_size * 0.618),
                        '78.6%': recent_high.price - (move_size * 0.786)
                    }
                    
                    # OTE zone (62%-79%)
                    ote_high = fib_levels['61.8%']
                    ote_low = fib_levels['78.6%']
                    
                    current_price = stock_data['close'].iloc[-1]
                    
                    ote_opportunities.append({
                        'direction': 'bullish',
                        'swing_low': recent_low.price,
                        'swing_high': recent_high.price,
                        'ote_zone_high': ote_high,
                        'ote_zone_low': ote_low,
                        'fibonacci_levels': fib_levels,
                        'current_price': current_price,
                        'in_ote_zone': ote_low <= current_price <= ote_high,
                        'distance_to_ote': min(abs(current_price - ote_high), abs(current_price - ote_low)),
                        'confluence_score': self._calculate_ote_confluence(stock_data, ote_high, ote_low, 'bullish')
                    })
            
            # Analyze bearish OTE (retracement in downtrend)
            if len(swing_highs) >= 1 and len(swing_lows) >= 1:
                recent_high = swing_highs[-1]
                recent_low = swing_lows[-1]
                
                if recent_low.timestamp > recent_high.timestamp:  # Downtrend
                    move_size = recent_high.price - recent_low.price
                    
                    # Calculate Fibonacci levels
                    fib_levels = {
                        '23.6%': recent_low.price + (move_size * 0.236),
                        '38.2%': recent_low.price + (move_size * 0.382),
                        '50%': recent_low.price + (move_size * 0.5),
                        '61.8%': recent_low.price + (move_size * 0.618),
                        '78.6%': recent_low.price + (move_size * 0.786)
                    }
                    
                    # OTE zone (62%-79%)
                    ote_low = fib_levels['61.8%']
                    ote_high = fib_levels['78.6%']
                    
                    current_price = stock_data['close'].iloc[-1]
                    
                    ote_opportunities.append({
                        'direction': 'bearish',
                        'swing_high': recent_high.price,
                        'swing_low': recent_low.price,
                        'ote_zone_high': ote_high,
                        'ote_zone_low': ote_low,
                        'fibonacci_levels': fib_levels,
                        'current_price': current_price,
                        'in_ote_zone': ote_low <= current_price <= ote_high,
                        'distance_to_ote': min(abs(current_price - ote_high), abs(current_price - ote_low)),
                        'confluence_score': self._calculate_ote_confluence(stock_data, ote_high, ote_low, 'bearish')
                    })
            
            return {
                'ote_opportunities': ote_opportunities,
                'best_opportunity': max(ote_opportunities, key=lambda x: x['confluence_score']) if ote_opportunities else None,
                'active_ote_zones': [op for op in ote_opportunities if op['in_ote_zone']]
            }
            
        except Exception as e:
            logger.error(f"Error in OTE analysis: {e}")
            return {'error': str(e)}

    def concept_19_smt_divergence(self, stock_data: pd.DataFrame, correlated_stocks: List[str] = None) -> List[Dict]:
        """
        CONCEPT 19: SMT Divergence (Smart Money Divergence across correlated pairs)
        - Cross-asset divergence analysis
        - Sector rotation signals
        - Relative strength/weakness
        """
        try:
            divergences = []
            
            # If no correlated stocks provided, use sector ETFs as proxy
            if not correlated_stocks:
                correlated_stocks = ['SPY', 'QQQ', 'IWM']  # Major market indices
            
            # This is a simplified implementation
            # In practice, you'd fetch data for correlated assets
            
            current_symbol_highs = self._find_swing_highs(stock_data)
            current_symbol_lows = self._find_swing_lows(stock_data)
            
            if len(current_symbol_highs) >= 2 and len(current_symbol_lows) >= 2:
                # Analyze recent swing points
                recent_highs = current_symbol_highs[-2:]
                recent_lows = current_symbol_lows[-2:]
                
                # Check for divergence patterns
                # Higher high in price but lower high in correlated asset = bearish divergence
                if recent_highs[-1].price > recent_highs[-2].price:
                    divergences.append({
                        'divergence_type': 'potential_bearish_smt',
                        'timestamp': recent_highs[-1].timestamp,
                        'price_level': recent_highs[-1].price,
                        'pattern': 'higher_high_in_symbol',
                        'requires_correlation_check': True,
                        'strength': recent_highs[-1].strength,
                        'note': 'Check if correlated assets failed to make new highs'
                    })
                
                # Lower low in price but higher low in correlated asset = bullish divergence
                if recent_lows[-1].price < recent_lows[-2].price:
                    divergences.append({
                        'divergence_type': 'potential_bullish_smt',
                        'timestamp': recent_lows[-1].timestamp,
                        'price_level': recent_lows[-1].price,
                        'pattern': 'lower_low_in_symbol',
                        'requires_correlation_check': True,
                        'strength': recent_lows[-1].strength,
                        'note': 'Check if correlated assets failed to make new lows'
                    })
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error in SMT divergence analysis: {e}")
            return []

    def concept_20_liquidity_voids_inefficiencies(self, stock_data: pd.DataFrame) -> List[Dict]:
        """
        CONCEPT 20: Liquidity Voids / Inefficiencies
        - Price gaps in stock movements
        - Unfilled gap analysis
        - Inefficiency targeting
        """
        try:
            voids = []
            
            if len(stock_data) < 5:
                return voids
            
            # Detect price gaps (inefficiencies)
            for i in range(1, len(stock_data)):
                current = stock_data.iloc[i]
                previous = stock_data.iloc[i-1]
                
                # Gap up (bullish inefficiency)
                if current['low'] > previous['high']:
                    gap_size = current['low'] - previous['high']
                    gap_percentage = gap_size / previous['high']
                    
                    if gap_percentage > 0.002:  # Minimum 0.2% gap
                        # Check if gap has been filled
                        future_data = stock_data.iloc[i+1:] if i+1 < len(stock_data) else pd.DataFrame()
                        is_filled = not future_data.empty and future_data['low'].min() <= previous['high']
                        
                        voids.append({
                            'timestamp': current['timestamp'] if 'timestamp' in current else current.name,
                            'void_type': 'gap_up_inefficiency',
                            'gap_high': current['low'],
                            'gap_low': previous['high'],
                            'gap_size': gap_size,
                            'gap_percentage': gap_percentage,
                            'is_filled': is_filled,
                            'fill_target': previous['high'],
                            'strength': gap_percentage * 10,  # Larger gaps are stronger
                            'volume_context': current['volume'] / stock_data['volume'].rolling(20).mean().iloc[i] if i >= 20 else 1.0
                        })
                
                # Gap down (bearish inefficiency)
                elif current['high'] < previous['low']:
                    gap_size = previous['low'] - current['high']
                    gap_percentage = gap_size / previous['low']
                    
                    if gap_percentage > 0.002:  # Minimum 0.2% gap
                        future_data = stock_data.iloc[i+1:] if i+1 < len(stock_data) else pd.DataFrame()
                        is_filled = not future_data.empty and future_data['high'].max() >= previous['low']
                        
                        voids.append({
                            'timestamp': current['timestamp'] if 'timestamp' in current else current.name,
                            'void_type': 'gap_down_inefficiency',
                            'gap_high': previous['low'],
                            'gap_low': current['high'],
                            'gap_size': gap_size,
                            'gap_percentage': gap_percentage,
                            'is_filled': is_filled,
                            'fill_target': previous['low'],
                            'strength': gap_percentage * 10,
                            'volume_context': current['volume'] / stock_data['volume'].rolling(20).mean().iloc[i] if i >= 20 else 1.0
                        })
            
            # Detect single-print areas (areas where price moved quickly with little time spent)
            single_prints = self._detect_single_print_areas(stock_data)
            voids.extend(single_prints)
            
            return sorted(voids, key=lambda x: x['strength'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error in liquidity voids analysis: {e}")
            return []

    # Additional helper methods for concepts 13-20
    def _determine_mm_bias(self, df: pd.DataFrame) -> str:
        """Determine current market maker bias"""
        try:
            recent_data = df.tail(10)
            avg_volume_ratio = recent_data['volume_ratio'].mean()
            net_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            if avg_volume_ratio > 1.5:
                if net_change > 0.02:
                    return 'accumulation'
                elif net_change < -0.02:
                    return 'distribution'
                else:
                    return 'manipulation'
            else:
                return 'neutral'
        except:
            return 'neutral'

    def _calculate_smart_money_flow(self, df: pd.DataFrame) -> Dict:
        """Calculate smart money flow indicators"""
        try:
            # Simplified smart money flow calculation
            df_temp = df.copy()
            df_temp['typical_price'] = (df_temp['high'] + df_temp['low'] + df_temp['close']) / 3
            df_temp['money_flow'] = df_temp['typical_price'] * df_temp['volume']
            
            recent_flow = df_temp['money_flow'].tail(10).sum()
            historical_avg = df_temp['money_flow'].rolling(50).mean().iloc[-1] * 10
            
            return {
                'recent_flow': recent_flow,
                'flow_ratio': recent_flow / historical_avg if historical_avg > 0 else 1.0,
                'flow_direction': 'inflow' if recent_flow > historical_avg else 'outflow'
            }
        except:
            return {'recent_flow': 0, 'flow_ratio': 1.0, 'flow_direction': 'neutral'}

    def _assess_current_program_activity(self, df: pd.DataFrame) -> Dict:
        """Assess current program trading activity"""
        try:
            recent_data = df.tail(5)
            
            volume_spike = recent_data['volume_spike'].mean()
            price_stability = recent_data['price_change'].std()
            
            if volume_spike > 2.0 and price_stability < 0.02:
                if recent_data['price_change'].mean() > 0:
                    return {'activity_level': 'high', 'program_type': 'buy_program'}
                else:
                    return {'activity_level': 'high', 'program_type': 'sell_program'}
            else:
                return {'activity_level': 'normal', 'program_type': 'none'}
        except:
            return {'activity_level': 'unknown', 'program_type': 'unknown'}

    def detect_market_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and label market sessions"""
        if df.empty or 'timestamp' not in df.columns:
            return df
        
        try:
            df_copy = df.copy()
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            df_copy['hour'] = df_copy['timestamp'].dt.hour
            df_copy['minute'] = df_copy['timestamp'].dt.minute
            
            conditions = [
                (df_copy['hour'] >= 4) & (df_copy['hour'] < 9),
                (df_copy['hour'] == 9) & (df_copy['minute'] >= 30),
                (df_copy['hour'] >= 10) & (df_copy['hour'] < 16),
                (df_copy['hour'] >= 16) & (df_copy['hour'] < 20),
            ]
            
            choices = ['premarket', 'market_open', 'market_hours', 'afterhours']
            df_copy['session'] = np.select(conditions, choices, default='closed')
            
            return df_copy
        except Exception:
            return df

    def _check_manipulation_reversal(self, window: pd.DataFrame) -> bool:
        """Check for manipulation reversal pattern"""
        try:
            # Look for quick spike and reversal
            high_point = window['high'].max()
            low_point = window['low'].min()
            
            high_idx = window['high'].idxmax()
            low_idx = window['low'].idxmin()
            
            # Manipulation typically shows quick spike then reversal
            if abs(high_idx - low_idx) <= 3:  # Spike and reversal within 3 periods
                spike_size = (high_point - low_point) / window['close'].mean()
                return spike_size > 0.03  # 3% spike
            
            return False
        except:
            return False

    def _determine_current_power_of_3_phase(self, recent_data: pd.DataFrame) -> Dict:
        """Determine current Power of 3 phase"""
        try:
            avg_volume_ratio = recent_data['volume_ratio'].mean()
            price_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].mean()
            net_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            avg_volatility = recent_data['volatility'].mean()
            
            if avg_volume_ratio > 1.2 and price_range < 0.1:
                if -0.02 <= net_change <= 0.05:
                    return {'phase': 'accumulation', 'confidence': 0.8}
                elif -0.05 <= net_change <= 0.02:
                    return {'phase': 'distribution', 'confidence': 0.8}
            elif avg_volume_ratio > 2.0 and avg_volatility > 0.03:
                return {'phase': 'manipulation', 'confidence': 0.7}
            else:
                return {'phase': 'transition', 'confidence': 0.5}
        except:
            return {'phase': 'unknown', 'confidence': 0.0}

    def _analyze_phase_cycles(self, phases: List[Dict]) -> Dict:
        """Analyze Power of 3 phase cycles"""
        if not phases:
            return {'cycle_count': 0, 'avg_cycle_length': 0}
        
        try:
            # Count complete cycles (accumulation -> manipulation -> distribution)
            cycles = 0
            i = 0
            
            while i < len(phases) - 2:
                if (phases[i]['phase'] == 'accumulation' and 
                    phases[i+1]['phase'] == 'manipulation' and 
                    phases[i+2]['phase'] == 'distribution'):
                    cycles += 1
                    i += 3
                else:
                    i += 1
            
            return {
                'cycle_count': cycles,
                'total_phases': len(phases),
                'phase_distribution': {
                    'accumulation': len([p for p in phases if p['phase'] == 'accumulation']),
                    'manipulation': len([p for p in phases if p['phase'] == 'manipulation']),
                    'distribution': len([p for p in phases if p['phase'] == 'distribution'])
                }
            }
        except:
            return {'cycle_count': 0, 'avg_cycle_length': 0}

    def _predict_next_phase(self, phases: List[Dict]) -> str:
        """Predict next expected phase"""
        if not phases:
            return 'unknown'
        
        last_phase = phases[-1]['phase']
        
        if last_phase == 'accumulation':
            return 'manipulation'
        elif last_phase == 'manipulation':
            return 'distribution'
        elif last_phase == 'distribution':
            return 'accumulation'
        else:
            return 'unknown'

    def _calculate_ote_confluence(self, df: pd.DataFrame, ote_high: float, ote_low: float, direction: str) -> float:
        """Calculate confluence score for OTE zone"""
        try:
            confluence_score = 0.5  # Base score
            
            # Check for order blocks in OTE zone
            order_blocks = self.concept_4_order_blocks_bullish_bearish(df)
            for ob in order_blocks:
                if ((ob.low_price <= ote_high and ob.high_price >= ote_low) and 
                    ((direction == 'bullish' and ob.block_type == 'bullish') or
                     (direction == 'bearish' and ob.block_type == 'bearish'))):
                    confluence_score += 0.2
            
            # Check for FVGs in OTE zone
            fvgs = self.concept_6_fair_value_gaps_fvg_imbalances(df)
            for fvg in fvgs:
                if ((fvg.gap_low <= ote_high and fvg.gap_high >= ote_low) and
                    ((direction == 'bullish' and fvg.gap_type == 'bullish') or
                     (direction == 'bearish' and fvg.gap_type == 'bearish'))):
                    confluence_score += 0.15
            
            # Check for liquidity levels
            liquidity = self.concept_2_liquidity_buyside_sellside(df)
            if direction == 'bullish' and liquidity.get('nearest_sellside'):
                sellside_level = liquidity['nearest_sellside']['price_level']
                if ote_low <= sellside_level <= ote_high:
                    confluence_score += 0.25
            elif direction == 'bearish' and liquidity.get('nearest_buyside'):
                buyside_level = liquidity['nearest_buyside']['price_level']
                if ote_low <= buyside_level <= ote_high:
                    confluence_score += 0.25
            
            return min(confluence_score, 1.0)
        except:
            return 0.5

    def _detect_single_print_areas(self, df: pd.DataFrame) -> List[Dict]:
        """Detect single print areas (areas with little time/volume spent)"""
        single_prints = []
        
        try:
            # Look for areas where price moved quickly with low volume
            df_temp = df.copy()
            df_temp['range_size'] = df_temp['high'] - df_temp['low']
            df_temp['volume_per_range'] = df_temp['volume'] / df_temp['range_size']
            
            # Identify low volume per range areas
            low_volume_threshold = df_temp['volume_per_range'].quantile(0.2)
            
            for i in range(len(df_temp)):
                if (df_temp['volume_per_range'].iloc[i] < low_volume_threshold and 
                    df_temp['range_size'].iloc[i] > df_temp['range_size'].mean()):
                    
                    single_prints.append({
                        'timestamp': df_temp.iloc[i]['timestamp'] if 'timestamp' in df_temp.columns else df_temp.index[i],
                        'void_type': 'single_print_area',
                        'gap_high': df_temp['high'].iloc[i],
                        'gap_low': df_temp['low'].iloc[i],
                        'gap_size': df_temp['range_size'].iloc[i],
                        'gap_percentage': df_temp['range_size'].iloc[i] / df_temp['close'].iloc[i],
                        'is_filled': False,  # Would need to check future data
                        'fill_target': (df_temp['high'].iloc[i] + df_temp['low'].iloc[i]) / 2,
                        'strength': df_temp['range_size'].iloc[i] / df_temp['close'].iloc[i],
                        'volume_context': df_temp['volume_per_range'].iloc[i]
                    })
        except Exception:
            pass
        
        return single_prints

# Global instance
market_structure_analyzer = StockMarketStructureAnalyzer()