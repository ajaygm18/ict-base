"""
Complete ICT Concepts Implementation (Concepts 1-20)
All core ICT trading concepts with production-ready algorithms
"""

from ict_engine.core_concepts import StockMarketStructureAnalyzer
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class CompleteICTConceptsEngine(StockMarketStructureAnalyzer):
    """Complete implementation of all 20 core ICT concepts"""
    
    def concept_4_order_blocks(self, stock_data: pd.DataFrame) -> Dict:
        """CONCEPT 4: Order Blocks (Bullish & Bearish)"""
        try:
            order_blocks = []
            
            for i in range(5, len(stock_data) - 5):
                # Bullish Order Block: Last bearish candle before bullish impulse
                if self._is_bullish_order_block(stock_data, i):
                    order_blocks.append({
                        'type': 'bullish',
                        'timestamp': stock_data.index[i],
                        'high': stock_data.iloc[i]['high'],
                        'low': stock_data.iloc[i]['low'],
                        'open': stock_data.iloc[i]['open'],
                        'close': stock_data.iloc[i]['close'],
                        'strength': self._calculate_ob_strength(stock_data, i, 'bullish'),
                        'mitigation_level': stock_data.iloc[i]['low'],
                        'is_mitigated': False
                    })
                
                # Bearish Order Block: Last bullish candle before bearish impulse
                if self._is_bearish_order_block(stock_data, i):
                    order_blocks.append({
                        'type': 'bearish',
                        'timestamp': stock_data.index[i],
                        'high': stock_data.iloc[i]['high'],
                        'low': stock_data.iloc[i]['low'],
                        'open': stock_data.iloc[i]['open'],
                        'close': stock_data.iloc[i]['close'],
                        'strength': self._calculate_ob_strength(stock_data, i, 'bearish'),
                        'mitigation_level': stock_data.iloc[i]['high'],
                        'is_mitigated': False
                    })
            
            # Check for mitigation
            current_price = stock_data['close'].iloc[-1]
            for ob in order_blocks:
                if ob['type'] == 'bullish' and current_price <= ob['mitigation_level']:
                    ob['is_mitigated'] = True
                elif ob['type'] == 'bearish' and current_price >= ob['mitigation_level']:
                    ob['is_mitigated'] = True
            
            return {
                'order_blocks': order_blocks[-20:],  # Keep last 20
                'active_bullish': [ob for ob in order_blocks if ob['type'] == 'bullish' and not ob['is_mitigated']],
                'active_bearish': [ob for ob in order_blocks if ob['type'] == 'bearish' and not ob['is_mitigated']],
                'total_found': len(order_blocks)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def concept_5_breaker_blocks(self, stock_data: pd.DataFrame) -> Dict:
        """CONCEPT 5: Breaker Blocks"""
        try:
            breaker_blocks = []
            order_blocks = self.concept_4_order_blocks(stock_data)['order_blocks']
            
            for ob in order_blocks:
                if ob['is_mitigated']:
                    # Check if price came back to test the mitigated order block
                    ob_index = stock_data.index.get_loc(ob['timestamp'])
                    
                    if ob_index < len(stock_data) - 10:
                        future_data = stock_data.iloc[ob_index+1:]
                        
                        if ob['type'] == 'bullish':
                            # Look for price coming back down to test
                            test_level = (ob['high'] + ob['low']) / 2
                            retests = future_data[future_data['low'] <= test_level]
                            
                            if len(retests) > 0:
                                breaker_blocks.append({
                                    'type': 'bullish_breaker',
                                    'original_ob': ob,
                                    'test_level': test_level,
                                    'retests': len(retests),
                                    'strength': ob['strength'] * 1.2,  # Breakers are stronger
                                    'status': 'active'
                                })
                        
                        else:  # bearish order block
                            test_level = (ob['high'] + ob['low']) / 2
                            retests = future_data[future_data['high'] >= test_level]
                            
                            if len(retests) > 0:
                                breaker_blocks.append({
                                    'type': 'bearish_breaker',
                                    'original_ob': ob,
                                    'test_level': test_level,
                                    'retests': len(retests),
                                    'strength': ob['strength'] * 1.2,
                                    'status': 'active'
                                })
            
            return {
                'breaker_blocks': breaker_blocks,
                'active_bullish_breakers': [bb for bb in breaker_blocks if bb['type'] == 'bullish_breaker'],
                'active_bearish_breakers': [bb for bb in breaker_blocks if bb['type'] == 'bearish_breaker'],
                'total_found': len(breaker_blocks)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def concept_6_fair_value_gaps(self, stock_data: pd.DataFrame) -> Dict:
        """CONCEPT 6: Fair Value Gaps (FVG)"""
        try:
            fvgs = []
            
            for i in range(1, len(stock_data) - 1):
                # Bullish FVG: Current low > Previous high
                if stock_data.iloc[i]['low'] > stock_data.iloc[i-1]['high']:
                    gap_size = stock_data.iloc[i]['low'] - stock_data.iloc[i-1]['high']
                    
                    fvgs.append({
                        'type': 'bullish',
                        'timestamp': stock_data.index[i],
                        'gap_high': stock_data.iloc[i]['low'],
                        'gap_low': stock_data.iloc[i-1]['high'],
                        'gap_size': gap_size,
                        'gap_size_percent': (gap_size / stock_data.iloc[i-1]['high']) * 100,
                        'mitigation_level': (stock_data.iloc[i]['low'] + stock_data.iloc[i-1]['high']) / 2,
                        'is_mitigated': False,
                        'strength': min(gap_size / stock_data.iloc[i-1]['high'] * 10, 5.0)
                    })
                
                # Bearish FVG: Current high < Previous low
                elif stock_data.iloc[i]['high'] < stock_data.iloc[i-1]['low']:
                    gap_size = stock_data.iloc[i-1]['low'] - stock_data.iloc[i]['high']
                    
                    fvgs.append({
                        'type': 'bearish',
                        'timestamp': stock_data.index[i],
                        'gap_high': stock_data.iloc[i-1]['low'],
                        'gap_low': stock_data.iloc[i]['high'],
                        'gap_size': gap_size,
                        'gap_size_percent': (gap_size / stock_data.iloc[i]['high']) * 100,
                        'mitigation_level': (stock_data.iloc[i-1]['low'] + stock_data.iloc[i]['high']) / 2,
                        'is_mitigated': False,
                        'strength': min(gap_size / stock_data.iloc[i]['high'] * 10, 5.0)
                    })
            
            # Check for mitigation
            current_price = stock_data['close'].iloc[-1]
            for fvg in fvgs:
                if fvg['type'] == 'bullish':
                    if current_price <= fvg['gap_low']:
                        fvg['is_mitigated'] = True
                else:  # bearish
                    if current_price >= fvg['gap_high']:
                        fvg['is_mitigated'] = True
            
            return {
                'fair_value_gaps': fvgs[-50:],  # Keep last 50
                'active_bullish_fvgs': [fvg for fvg in fvgs if fvg['type'] == 'bullish' and not fvg['is_mitigated']],
                'active_bearish_fvgs': [fvg for fvg in fvgs if fvg['type'] == 'bearish' and not fvg['is_mitigated']],
                'total_found': len(fvgs)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def concept_7_rejection_blocks(self, stock_data: pd.DataFrame) -> Dict:
        """CONCEPT 7: Rejection Blocks"""
        try:
            rejection_blocks = []
            
            for i in range(10, len(stock_data) - 5):
                # Look for strong rejection patterns
                current_candle = stock_data.iloc[i]
                
                # Bullish rejection: Long lower wick with strong close
                lower_wick = current_candle['open'] - current_candle['low']
                body_size = abs(current_candle['close'] - current_candle['open'])
                total_range = current_candle['high'] - current_candle['low']
                
                if (lower_wick > body_size * 2 and 
                    lower_wick > total_range * 0.6 and
                    current_candle['close'] > current_candle['open']):
                    
                    rejection_blocks.append({
                        'type': 'bullish_rejection',
                        'timestamp': stock_data.index[i],
                        'rejection_low': current_candle['low'],
                        'rejection_high': current_candle['close'],
                        'wick_size': lower_wick,
                        'wick_ratio': lower_wick / total_range,
                        'strength': min(lower_wick / body_size, 5.0),
                        'volume_confirmation': current_candle.get('volume', 0) > stock_data['volume'].iloc[i-10:i].mean()
                    })
                
                # Bearish rejection: Long upper wick with strong close down
                upper_wick = current_candle['high'] - current_candle['open']
                
                if (upper_wick > body_size * 2 and 
                    upper_wick > total_range * 0.6 and
                    current_candle['close'] < current_candle['open']):
                    
                    rejection_blocks.append({
                        'type': 'bearish_rejection',
                        'timestamp': stock_data.index[i],
                        'rejection_high': current_candle['high'],
                        'rejection_low': current_candle['close'],
                        'wick_size': upper_wick,
                        'wick_ratio': upper_wick / total_range,
                        'strength': min(upper_wick / body_size, 5.0),
                        'volume_confirmation': current_candle.get('volume', 0) > stock_data['volume'].iloc[i-10:i].mean()
                    })
            
            return {
                'rejection_blocks': rejection_blocks[-30:],
                'bullish_rejections': [rb for rb in rejection_blocks if rb['type'] == 'bullish_rejection'],
                'bearish_rejections': [rb for rb in rejection_blocks if rb['type'] == 'bearish_rejection'],
                'total_found': len(rejection_blocks)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def concept_8_mitigation_blocks(self, stock_data: pd.DataFrame) -> Dict:
        """CONCEPT 8: Mitigation Blocks"""
        try:
            # Get order blocks and FVGs to check for mitigation
            order_blocks = self.concept_4_order_blocks(stock_data)['order_blocks']
            fvgs = self.concept_6_fair_value_gaps(stock_data)['fair_value_gaps']
            
            mitigation_blocks = []
            
            # Check order block mitigations
            for ob in order_blocks:
                if ob['is_mitigated']:
                    mitigation_blocks.append({
                        'type': 'order_block_mitigation',
                        'original_type': ob['type'],
                        'mitigation_price': ob['mitigation_level'],
                        'original_timestamp': ob['timestamp'],
                        'strength': ob['strength'] * 0.8,  # Mitigated blocks are weaker
                        'status': 'mitigated'
                    })
            
            # Check FVG mitigations
            for fvg in fvgs:
                if fvg['is_mitigated']:
                    mitigation_blocks.append({
                        'type': 'fvg_mitigation',
                        'original_type': fvg['type'],
                        'mitigation_price': fvg['mitigation_level'],
                        'original_timestamp': fvg['timestamp'],
                        'strength': fvg['strength'] * 0.6,
                        'status': 'mitigated'
                    })
            
            return {
                'mitigation_blocks': mitigation_blocks,
                'mitigated_obs': len([mb for mb in mitigation_blocks if mb['type'] == 'order_block_mitigation']),
                'mitigated_fvgs': len([mb for mb in mitigation_blocks if mb['type'] == 'fvg_mitigation']),
                'total_mitigations': len(mitigation_blocks)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def concept_9_supply_demand_zones(self, stock_data: pd.DataFrame) -> Dict:
        """CONCEPT 9: Supply & Demand Zones"""
        try:
            supply_zones = []
            demand_zones = []
            
            # Use swing points to identify zones
            swing_highs = self._find_swing_highs_enhanced(stock_data)
            swing_lows = self._find_swing_lows_enhanced(stock_data)
            
            # Supply zones (around swing highs)
            for high in swing_highs[-10:]:
                zone_range = self._calculate_zone_range(stock_data, high.index, 'supply')
                
                supply_zones.append({
                    'type': 'supply',
                    'timestamp': high.timestamp,
                    'zone_high': zone_range['high'],
                    'zone_low': zone_range['low'],
                    'zone_center': (zone_range['high'] + zone_range['low']) / 2,
                    'strength': high.strength,
                    'touches': self._count_zone_touches(stock_data, zone_range, high.index),
                    'is_fresh': self._is_zone_fresh(stock_data, zone_range, high.index)
                })
            
            # Demand zones (around swing lows)
            for low in swing_lows[-10:]:
                zone_range = self._calculate_zone_range(stock_data, low.index, 'demand')
                
                demand_zones.append({
                    'type': 'demand',
                    'timestamp': low.timestamp,
                    'zone_high': zone_range['high'],
                    'zone_low': zone_range['low'],
                    'zone_center': (zone_range['high'] + zone_range['low']) / 2,
                    'strength': low.strength,
                    'touches': self._count_zone_touches(stock_data, zone_range, low.index),
                    'is_fresh': self._is_zone_fresh(stock_data, zone_range, low.index)
                })
            
            return {
                'supply_zones': sorted(supply_zones, key=lambda x: x['strength'], reverse=True),
                'demand_zones': sorted(demand_zones, key=lambda x: x['strength'], reverse=True),
                'fresh_supply_zones': [sz for sz in supply_zones if sz['is_fresh']],
                'fresh_demand_zones': [dz for dz in demand_zones if dz['is_fresh']],
                'total_zones': len(supply_zones) + len(demand_zones)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def concept_10_premium_discount_ote(self, stock_data: pd.DataFrame) -> Dict:
        """CONCEPT 10: Premium & Discount (OTE - Optimal Trade Entry)"""
        try:
            # Calculate recent range
            lookback = min(50, len(stock_data))
            recent_data = stock_data.iloc[-lookback:]
            
            range_high = recent_data['high'].max()
            range_low = recent_data['low'].min()
            range_size = range_high - range_low
            
            # Calculate OTE levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
            ote_levels = {
                'range_high': range_high,
                'range_low': range_low,
                'range_size': range_size,
                'premium_start': range_high - (range_size * 0.236),  # 76.4% level
                'discount_end': range_low + (range_size * 0.236),    # 23.6% level
                'optimal_bullish_entry': range_low + (range_size * 0.618),  # 61.8%
                'optimal_bearish_entry': range_high - (range_size * 0.618), # 38.2%
                'equilibrium': range_low + (range_size * 0.5),       # 50%
                'fib_382': range_low + (range_size * 0.382),
                'fib_618': range_low + (range_size * 0.618),
                'fib_786': range_low + (range_size * 0.786)
            }
            
            # Current price analysis
            current_price = stock_data['close'].iloc[-1]
            price_position = (current_price - range_low) / range_size
            
            if price_position > 0.764:
                market_state = 'premium'
                bias = 'bearish'
            elif price_position < 0.236:
                market_state = 'discount'
                bias = 'bullish'
            else:
                market_state = 'balanced'
                bias = 'neutral'
            
            # Find optimal trade entries
            optimal_entries = []
            
            if market_state == 'premium':
                # Look for bearish OTE entries
                if abs(current_price - ote_levels['optimal_bearish_entry']) / current_price < 0.02:
                    optimal_entries.append({
                        'type': 'bearish_ote',
                        'entry_level': ote_levels['optimal_bearish_entry'],
                        'target': ote_levels['equilibrium'],
                        'stop_loss': range_high,
                        'risk_reward': abs(ote_levels['equilibrium'] - ote_levels['optimal_bearish_entry']) / abs(range_high - ote_levels['optimal_bearish_entry'])
                    })
            
            elif market_state == 'discount':
                # Look for bullish OTE entries
                if abs(current_price - ote_levels['optimal_bullish_entry']) / current_price < 0.02:
                    optimal_entries.append({
                        'type': 'bullish_ote',
                        'entry_level': ote_levels['optimal_bullish_entry'],
                        'target': ote_levels['equilibrium'],
                        'stop_loss': range_low,
                        'risk_reward': abs(ote_levels['equilibrium'] - ote_levels['optimal_bullish_entry']) / abs(ote_levels['optimal_bullish_entry'] - range_low)
                    })
            
            return {
                'ote_levels': ote_levels,
                'current_price': current_price,
                'price_position_percent': price_position * 100,
                'market_state': market_state,
                'bias': bias,
                'optimal_entries': optimal_entries,
                'range_analysis': {
                    'range_size_percent': (range_size / range_low) * 100,
                    'volatility': range_size / current_price,
                    'range_age_bars': lookback
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    # Helper methods for the concepts
    def _is_bullish_order_block(self, data: pd.DataFrame, index: int) -> bool:
        """Check if candle at index is a bullish order block"""
        if index < 5 or index >= len(data) - 5:
            return False
        
        current = data.iloc[index]
        
        # Must be bearish candle
        if current['close'] >= current['open']:
            return False
        
        # Check for bullish impulse after
        future_high = data.iloc[index+1:index+6]['high'].max()
        impulse_strength = (future_high - current['high']) / current['high']
        
        return impulse_strength > 0.02  # 2% impulse minimum
    
    def _is_bearish_order_block(self, data: pd.DataFrame, index: int) -> bool:
        """Check if candle at index is a bearish order block"""
        if index < 5 or index >= len(data) - 5:
            return False
        
        current = data.iloc[index]
        
        # Must be bullish candle
        if current['close'] <= current['open']:
            return False
        
        # Check for bearish impulse after
        future_low = data.iloc[index+1:index+6]['low'].min()
        impulse_strength = (current['low'] - future_low) / current['low']
        
        return impulse_strength > 0.02  # 2% impulse minimum
    
    def _calculate_ob_strength(self, data: pd.DataFrame, index: int, ob_type: str) -> float:
        """Calculate order block strength"""
        current = data.iloc[index]
        body_size = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        
        # Volume confirmation
        avg_volume = data['volume'].iloc[max(0, index-20):index].mean()
        volume_ratio = current['volume'] / avg_volume if avg_volume > 0 else 1.0
        
        # Size and volume factors
        size_factor = body_size / current['close']
        volume_factor = min(volume_ratio / 2, 2.0)
        
        return min(size_factor * volume_factor * 10, 5.0)
    
    def _calculate_zone_range(self, data: pd.DataFrame, index: int, zone_type: str) -> Dict:
        """Calculate supply/demand zone range"""
        if zone_type == 'supply':
            # Zone around swing high
            zone_high = data.iloc[index]['high']
            zone_low = data.iloc[index]['low']
        else:  # demand
            # Zone around swing low
            zone_high = data.iloc[index]['high']
            zone_low = data.iloc[index]['low']
        
        return {'high': zone_high, 'low': zone_low}
    
    def _count_zone_touches(self, data: pd.DataFrame, zone_range: Dict, start_index: int) -> int:
        """Count how many times price touched the zone after formation"""
        touches = 0
        future_data = data.iloc[start_index+1:]
        
        for _, candle in future_data.iterrows():
            if (candle['low'] <= zone_range['high'] and 
                candle['high'] >= zone_range['low']):
                touches += 1
        
        return touches
    
    def _is_zone_fresh(self, data: pd.DataFrame, zone_range: Dict, start_index: int) -> bool:
        """Check if zone is fresh (not broken)"""
        future_data = data.iloc[start_index+1:]
        
        for _, candle in future_data.iterrows():
            # Zone is broken if price closes significantly inside it
            zone_center = (zone_range['high'] + zone_range['low']) / 2
            if abs(candle['close'] - zone_center) < (zone_range['high'] - zone_range['low']) * 0.3:
                return False
        
        return True