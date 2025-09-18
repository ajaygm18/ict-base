"""
Enhanced ICT Core Concepts Helper Methods
Production-ready implementations of all core ICT pattern recognition methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import scipy.stats as stats
from dataclasses import asdict

from monitoring.logging import logger

class ICTAnalysisHelpers:
    """Helper methods for ICT analysis with enhanced algorithms"""
    
    @staticmethod
    def _find_swing_highs_enhanced(stock_data: pd.DataFrame, lookback: int = 5) -> List:
        """Enhanced swing high detection with statistical significance"""
        swing_highs = []
        high_prices = stock_data['high'].values
        volumes = stock_data['volume'].values
        timestamps = stock_data.index
        
        for i in range(lookback, len(high_prices) - lookback):
            current_high = high_prices[i]
            
            # Check if current point is highest in lookback window
            left_window = high_prices[i-lookback:i]
            right_window = high_prices[i+1:i+lookback+1]
            
            if (current_high > np.max(left_window) and 
                current_high > np.max(right_window)):
                
                # Calculate strength based on price deviation and volume
                price_deviation = np.std(high_prices[i-lookback:i+lookback+1])
                volume_strength = volumes[i] / np.mean(volumes[max(0, i-20):i+1])
                
                strength = min(price_deviation * volume_strength, 10.0)
                
                from ict_engine.core_concepts import SwingPoint
                swing_point = SwingPoint(
                    timestamp=timestamps[i],
                    price=current_high,
                    swing_type='high',
                    index=i,
                    strength=strength
                )
                swing_highs.append(swing_point)
        
        return swing_highs
    
    @staticmethod
    def _find_swing_lows_enhanced(stock_data: pd.DataFrame, lookback: int = 5) -> List:
        """Enhanced swing low detection with statistical significance"""
        swing_lows = []
        low_prices = stock_data['low'].values
        volumes = stock_data['volume'].values
        timestamps = stock_data.index
        
        for i in range(lookback, len(low_prices) - lookback):
            current_low = low_prices[i]
            
            # Check if current point is lowest in lookback window
            left_window = low_prices[i-lookback:i]
            right_window = low_prices[i+1:i+lookback+1]
            
            if (current_low < np.min(left_window) and 
                current_low < np.min(right_window)):
                
                # Calculate strength
                price_deviation = np.std(low_prices[i-lookback:i+lookback+1])
                volume_strength = volumes[i] / np.mean(volumes[max(0, i-20):i+1])
                
                strength = min(price_deviation * volume_strength, 10.0)
                
                from ict_engine.core_concepts import SwingPoint
                swing_point = SwingPoint(
                    timestamp=timestamps[i],
                    price=current_low,
                    swing_type='low',
                    index=i,
                    strength=strength
                )
                swing_lows.append(swing_point)
        
        return swing_lows
    
    @staticmethod
    def _analyze_structure_patterns(swing_highs: List, swing_lows: List) -> Dict:
        """Analyze market structure patterns (HH, HL, LH, LL)"""
        patterns = {
            'higher_highs': 0,
            'lower_highs': 0,
            'higher_lows': 0,
            'lower_lows': 0,
            'pattern_sequence': [],
            'trend_consistency': 0.0
        }
        
        if len(swing_highs) < 2:
            return patterns
        
        # Analyze high patterns
        for i in range(1, len(swing_highs)):
            prev_high = swing_highs[i-1].price
            curr_high = swing_highs[i].price
            
            if curr_high > prev_high:
                patterns['higher_highs'] += 1
                patterns['pattern_sequence'].append('HH')
            elif curr_high < prev_high:
                patterns['lower_highs'] += 1
                patterns['pattern_sequence'].append('LH')
        
        # Analyze low patterns
        for i in range(1, len(swing_lows)):
            prev_low = swing_lows[i-1].price
            curr_low = swing_lows[i].price
            
            if curr_low > prev_low:
                patterns['higher_lows'] += 1
                patterns['pattern_sequence'].append('HL')
            elif curr_low < prev_low:
                patterns['lower_lows'] += 1
                patterns['pattern_sequence'].append('LL')
        
        # Calculate trend consistency
        total_patterns = sum([patterns['higher_highs'], patterns['lower_highs'], 
                             patterns['higher_lows'], patterns['lower_lows']])
        
        if total_patterns > 0:
            bullish_patterns = patterns['higher_highs'] + patterns['higher_lows']
            bearish_patterns = patterns['lower_highs'] + patterns['lower_lows']
            patterns['trend_consistency'] = abs(bullish_patterns - bearish_patterns) / total_patterns
        
        return patterns
    
    @staticmethod
    def _classify_current_structure_enhanced(swing_highs: List, swing_lows: List, stock_data: pd.DataFrame) -> Dict:
        """Enhanced structure classification with market context"""
        current_price = stock_data['close'].iloc[-1]
        
        structure = {
            'trend_direction': 'sideways',
            'trend_strength': 0.0,
            'structure_status': 'unclear',
            'key_levels': {
                'resistance': None,
                'support': None
            },
            'next_target': None
        }
        
        if not swing_highs or not swing_lows:
            return structure
        
        # Recent swing points
        recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
        recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
        
        # Determine trend direction
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            high_trend = 1 if recent_highs[-1].price > recent_highs[-2].price else -1
            low_trend = 1 if recent_lows[-1].price > recent_lows[-2].price else -1
            
            if high_trend == 1 and low_trend == 1:
                structure['trend_direction'] = 'bullish'
                structure['trend_strength'] = 0.8
            elif high_trend == -1 and low_trend == -1:
                structure['trend_direction'] = 'bearish'
                structure['trend_strength'] = 0.8
            else:
                structure['trend_direction'] = 'sideways'
                structure['trend_strength'] = 0.3
        
        # Key levels
        structure['key_levels']['resistance'] = max([h.price for h in recent_highs])
        structure['key_levels']['support'] = min([l.price for l in recent_lows])
        
        return structure
    
    @staticmethod
    def _detect_structure_breaks_enhanced(swing_highs: List, swing_lows: List, stock_data: pd.DataFrame) -> List[Dict]:
        """Enhanced structure break detection with confirmation"""
        breaks = []
        current_price = stock_data['close'].iloc[-1]
        current_volume = stock_data['volume'].iloc[-1]
        avg_volume = stock_data['volume'].rolling(20).mean().iloc[-1]
        
        if not swing_highs or not swing_lows:
            return breaks
        
        # Check for resistance breaks
        for high in swing_highs[-5:]:
            if current_price > high.price:
                volume_confirmation = current_volume > avg_volume * 1.2
                price_confirmation = (current_price - high.price) / high.price > 0.002
                
                if volume_confirmation and price_confirmation:
                    breaks.append({
                        'type': 'resistance_break',
                        'level': high.price,
                        'current_price': current_price,
                        'break_strength': min((current_price - high.price) / high.price * 100, 5.0),
                        'volume_confirmation': volume_confirmation,
                        'timestamp': datetime.now()
                    })
        
        # Check for support breaks
        for low in swing_lows[-5:]:
            if current_price < low.price:
                volume_confirmation = current_volume > avg_volume * 1.2
                price_confirmation = (low.price - current_price) / low.price > 0.002
                
                if volume_confirmation and price_confirmation:
                    breaks.append({
                        'type': 'support_break',
                        'level': low.price,
                        'current_price': current_price,
                        'break_strength': min((low.price - current_price) / low.price * 100, 5.0),
                        'volume_confirmation': volume_confirmation,
                        'timestamp': datetime.now()
                    })
        
        return breaks
    
    @staticmethod
    def _analyze_trend_strength(swing_highs: List, swing_lows: List, stock_data: pd.DataFrame) -> Dict:
        """Analyze trend strength using multiple factors"""
        trend_analysis = {
            'momentum_score': 0.0,
            'consistency_score': 0.0,
            'volume_confirmation': 0.0,
            'overall_strength': 0.0,
            'trend_age_days': 0,
            'reliability': 'low'
        }
        
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return trend_analysis
        
        # Momentum analysis
        recent_price_change = (stock_data['close'].iloc[-1] - stock_data['close'].iloc[-20]) / stock_data['close'].iloc[-20]
        trend_analysis['momentum_score'] = min(abs(recent_price_change) * 10, 1.0)
        
        # Volume confirmation
        recent_volume = stock_data['volume'].iloc[-10:].mean()
        historical_volume = stock_data['volume'].iloc[-50:-10].mean()
        volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1.0
        trend_analysis['volume_confirmation'] = min(volume_ratio / 2, 1.0)
        
        # Pattern consistency
        patterns = ICTAnalysisHelpers._analyze_structure_patterns(swing_highs, swing_lows)
        trend_analysis['consistency_score'] = patterns['trend_consistency']
        
        # Overall strength
        trend_analysis['overall_strength'] = np.mean([
            trend_analysis['momentum_score'],
            trend_analysis['consistency_score'],
            trend_analysis['volume_confirmation']
        ])
        
        # Reliability assessment
        if trend_analysis['overall_strength'] > 0.7:
            trend_analysis['reliability'] = 'high'
        elif trend_analysis['overall_strength'] > 0.4:
            trend_analysis['reliability'] = 'medium'
        else:
            trend_analysis['reliability'] = 'low'
        
        return trend_analysis
    
    @staticmethod
    def _swing_point_to_dict(swing_point) -> Dict:
        """Convert SwingPoint to dictionary"""
        return {
            'timestamp': swing_point.timestamp.isoformat() if hasattr(swing_point.timestamp, 'isoformat') else str(swing_point.timestamp),
            'price': float(swing_point.price),
            'swing_type': swing_point.swing_type,
            'index': int(swing_point.index),
            'strength': float(swing_point.strength)
        }
    
    @staticmethod
    def _determine_market_phase(structure_patterns: Dict, trend_analysis: Dict) -> str:
        """Determine current market phase"""
        if trend_analysis['overall_strength'] > 0.6:
            if structure_patterns.get('higher_highs', 0) > structure_patterns.get('lower_highs', 0):
                return 'trending_up'
            elif structure_patterns.get('lower_highs', 0) > structure_patterns.get('higher_highs', 0):
                return 'trending_down'
        elif trend_analysis['consistency_score'] < 0.3:
            return 'ranging'
        else:
            return 'transitional'
        
        return 'unclear'
    
    @staticmethod
    def _calculate_structure_confidence_enhanced(swing_highs: List, swing_lows: List, structure_patterns: Dict) -> float:
        """Calculate confidence in structure analysis"""
        confidence_factors = []
        
        # Number of swing points
        total_swings = len(swing_highs) + len(swing_lows)
        confidence_factors.append(min(total_swings / 10, 1.0))
        
        # Pattern consistency
        confidence_factors.append(structure_patterns.get('trend_consistency', 0.0))
        
        # Recent activity (strength of recent swing points)
        if swing_highs:
            recent_high_strength = np.mean([h.strength for h in swing_highs[-3:]])
            confidence_factors.append(min(recent_high_strength / 5, 1.0))
        
        if swing_lows:
            recent_low_strength = np.mean([l.strength for l in swing_lows[-3:]])
            confidence_factors.append(min(recent_low_strength / 5, 1.0))
        
        return np.mean(confidence_factors) if confidence_factors else 0.0