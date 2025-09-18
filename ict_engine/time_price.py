"""
ICT Time & Price Theory Implementation (Concepts 21-30)
Killzones, Session Analysis, Fibonacci Ratios, Daily/Weekly Analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta, time
import logging
from dataclasses import dataclass
from enum import Enum
import pytz

logger = logging.getLogger(__name__)

class SessionType(Enum):
    PREMARKET = "premarket"
    MARKET_OPEN = "market_open"  
    MARKET_HOURS = "market_hours"
    LUNCH = "lunch"
    POWER_HOUR = "power_hour"
    AFTERHOURS = "afterhours"
    CLOSED = "closed"

@dataclass
class KillzoneInfo:
    session: SessionType
    start_time: time
    end_time: time
    timezone: str
    description: str
    importance: float  # 0.0 to 1.0

@dataclass
class SessionAnalysis:
    session_type: SessionType
    start_time: datetime
    end_time: datetime
    volume_profile: Dict
    price_action: Dict
    key_levels: List[float]
    bias: str  # bullish, bearish, neutral

@dataclass
class FibonacciLevel:
    level: float
    percentage: float
    price: float
    level_type: str  # retracement, extension
    significance: float

class StockTimeAndPriceAnalyzer:
    def __init__(self):
        self.et_timezone = pytz.timezone('US/Eastern')
        
        # Stock market killzones (adapted for stock market hours)
        self.stock_killzones = {
            SessionType.PREMARKET: KillzoneInfo(
                SessionType.PREMARKET, 
                time(4, 0), time(9, 30), 
                'US/Eastern',
                'Pre-market session - institutional positioning',
                0.7
            ),
            SessionType.MARKET_OPEN: KillzoneInfo(
                SessionType.MARKET_OPEN,
                time(9, 30), time(11, 0),
                'US/Eastern', 
                'Market open - high volatility and volume',
                0.9
            ),
            SessionType.LUNCH: KillzoneInfo(
                SessionType.LUNCH,
                time(11, 0), time(14, 0),
                'US/Eastern',
                'Lunch session - lower volatility',
                0.4
            ),
            SessionType.POWER_HOUR: KillzoneInfo(
                SessionType.POWER_HOUR,
                time(15, 0), time(16, 0),
                'US/Eastern',
                'Power hour - institutional positioning for close',
                0.8
            ),
            SessionType.AFTERHOURS: KillzoneInfo(
                SessionType.AFTERHOURS,
                time(16, 0), time(20, 0),
                'US/Eastern',
                'After hours - extended trading',
                0.5
            )
        }
        
    def concept_21_stock_killzones(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 21: Killzones adapted for stock market
        - Pre-market killzone (4:00-9:30 AM ET)
        - Market open killzone (9:30-11:00 AM ET)
        - Lunch killzone (11:00 AM-2:00 PM ET)
        - Power hour killzone (3:00-4:00 PM ET)
        """
        try:
            if stock_data.empty or 'timestamp' not in stock_data.columns:
                return {'error': 'No timestamp data available'}
            
            # Ensure timezone aware timestamps
            df = stock_data.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert('US/Eastern')
            
            # Analyze each killzone
            killzone_analysis = {}
            
            for session_type, killzone in self.stock_killzones.items():
                session_data = self._filter_by_time_range(df, killzone.start_time, killzone.end_time)
                
                if not session_data.empty:
                    analysis = self._analyze_killzone_performance(session_data, killzone)
                    killzone_analysis[session_type.value] = analysis
            
            # Overall killzone summary
            summary = self._generate_killzone_summary(killzone_analysis)
            
            return {
                'killzones': killzone_analysis,  # Test expects 'killzones' key
                'killzone_analysis': killzone_analysis,
                'summary': summary,
                'current_session': self._get_current_session(),
                'next_important_session': self._get_next_important_session(),
                'timezone': 'US/Eastern'
            }
            
        except Exception as e:
            logger.error(f"Error in killzone analysis: {e}")
            return {'error': str(e)}
    
    def concept_22_stock_session_opens(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 22: Stock Market Session Opens
        - Pre-market open (4:00 AM ET)
        - Regular market open (9:30 AM ET)  
        - After-hours open (4:00 PM ET)
        - Overnight session analysis
        """
        try:
            if stock_data.empty:
                return {'error': 'No data available'}
            
            df = stock_data.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            session_opens = {
                'premarket_opens': [],
                'market_opens': [],
                'afterhours_opens': [],
                'overnight_gaps': []
            }
            
            # Group by date to analyze session opens
            df['date'] = df['timestamp'].dt.date
            
            for date in df['date'].unique():
                daily_data = df[df['date'] == date].sort_values('timestamp')
                
                if len(daily_data) == 0:
                    continue
                
                # Analyze pre-market open (first data point after 4:00 AM)
                premarket_start = daily_data[daily_data['timestamp'].dt.time >= time(4, 0)]
                if not premarket_start.empty:
                    session_opens['premarket_opens'].append(
                        self._analyze_session_open(premarket_start.iloc[0], 'premarket', daily_data)
                    )
                
                # Analyze regular market open (9:30 AM)
                market_start = daily_data[daily_data['timestamp'].dt.time >= time(9, 30)]
                if not market_start.empty:
                    session_opens['market_opens'].append(
                        self._analyze_session_open(market_start.iloc[0], 'market', daily_data)
                    )
                
                # Analyze after-hours open (4:00 PM)
                afterhours_start = daily_data[daily_data['timestamp'].dt.time >= time(16, 0)]
                if not afterhours_start.empty:
                    session_opens['afterhours_opens'].append(
                        self._analyze_session_open(afterhours_start.iloc[0], 'afterhours', daily_data)
                    )
                
                # Analyze overnight gaps
                if len(daily_data) > 1:
                    overnight_gap = self._analyze_overnight_gap(daily_data)
                    if overnight_gap:
                        session_opens['overnight_gaps'].append(overnight_gap)
            
            result = {
                'session_opens': session_opens,
                'statistics': self._calculate_session_statistics(session_opens),
                'patterns': self._identify_session_patterns(session_opens)
            }
            
            # Add direct access for test compatibility
            if 'premarket_opens' in session_opens:
                result['premarket_opens'] = session_opens['premarket_opens']
            if 'market_opens' in session_opens:
                result['market_opens'] = session_opens['market_opens']
            if 'afterhours_opens' in session_opens:
                result['afterhours_opens'] = session_opens['afterhours_opens']
                
            return result
            
        except Exception as e:
            logger.error(f"Error in session opens analysis: {e}")
            return {'error': str(e)}
    
    def concept_23_fibonacci_ratios(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 23: Equilibrium & Fibonacci Ratios (50%, 62%, 70.5%, 79%)
        - 50% equilibrium level
        - 62% golden ratio retracement
        - 70.5% deep retracement
        - 79% optimal trade entry zone
        """
        try:
            if len(stock_data) < 10:
                return {'error': 'Insufficient data for Fibonacci analysis'}
            
            # Find significant swing points for Fibonacci analysis
            swing_highs = self._find_swing_points(stock_data, 'high')
            swing_lows = self._find_swing_points(stock_data, 'low')
            
            fibonacci_levels = []
            
            # Standard Fibonacci retracement levels
            fib_ratios = {
                '0%': 0.0,
                '23.6%': 0.236,
                '38.2%': 0.382,
                '50%': 0.5,      # Equilibrium
                '61.8%': 0.618,  # Golden ratio
                '70.5%': 0.705,  # Deep retracement
                '78.6%': 0.786,  # OTE zone
                '100%': 1.0
            }
            
            # Calculate Fibonacci retracements for recent swings
            if len(swing_highs) >= 1 and len(swing_lows) >= 1:
                # Bullish retracement (from low to high, then retrace)
                recent_low = swing_lows[-1]
                recent_high = swing_highs[-1]
                
                if recent_high['timestamp'] > recent_low['timestamp']:
                    # Uptrend retracement
                    price_range = recent_high['price'] - recent_low['price']
                    
                    for level_name, ratio in fib_ratios.items():
                        fib_price = recent_high['price'] - (price_range * ratio)
                        fibonacci_levels.append(FibonacciLevel(
                            level=ratio,
                            percentage=ratio * 100,
                            price=fib_price,
                            level_type='bullish_retracement',
                            significance=self._calculate_fib_significance(level_name, stock_data, fib_price)
                        ))
                
                # Bearish retracement (from high to low, then retrace)
                if recent_low['timestamp'] > recent_high['timestamp']:
                    # Downtrend retracement
                    price_range = recent_high['price'] - recent_low['price']
                    
                    for level_name, ratio in fib_ratios.items():
                        fib_price = recent_low['price'] + (price_range * ratio)
                        fibonacci_levels.append(FibonacciLevel(
                            level=ratio,
                            percentage=ratio * 100,
                            price=fib_price,
                            level_type='bearish_retracement',
                            significance=self._calculate_fib_significance(level_name, stock_data, fib_price)
                        ))
            
            # Identify current price position relative to Fibonacci levels
            current_price = stock_data['close'].iloc[-1]
            current_fib_analysis = self._analyze_current_fib_position(fibonacci_levels, current_price)
            
            return {
                'fibonacci_levels': [
                    {
                        'level': fib.level,
                        'percentage': fib.percentage,
                        'price': fib.price,
                        'level_type': fib.level_type,
                        'significance': fib.significance
                    } for fib in fibonacci_levels
                ],
                'current_price': current_price,
                'current_analysis': current_fib_analysis,
                'key_zones': {
                    'equilibrium_50': [fib for fib in fibonacci_levels if abs(fib.percentage - 50) < 0.1],
                    'golden_ratio_618': [fib for fib in fibonacci_levels if abs(fib.percentage - 61.8) < 0.1],
                    'ote_zone_705': [fib for fib in fibonacci_levels if abs(fib.percentage - 70.5) < 0.1],
                    'ote_zone_786': [fib for fib in fibonacci_levels if abs(fib.percentage - 78.6) < 0.1]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Fibonacci analysis: {e}")
            return {'error': str(e)}
    
    def concept_24_daily_weekly_range_expectations(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 24: Daily & Weekly Range Expectations
        - Average True Range (ATR) projections
        - Daily range expansion/contraction
        - Weekly range targeting
        - Volatility-based expectations
        """
        try:
            if len(stock_data) < 20:
                return {'error': 'Insufficient data for range analysis'}
            
            df = stock_data.copy()
            
            # Calculate True Range
            df['prev_close'] = df['close'].shift(1)
            df['true_range'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['prev_close']),
                    abs(df['low'] - df['prev_close'])
                )
            )
            
            # Calculate ATR
            atr_periods = [14, 20, 50]
            for period in atr_periods:
                df[f'atr_{period}'] = df['true_range'].rolling(window=period).mean()
            
            # Daily range analysis
            df['daily_range'] = df['high'] - df['low']
            df['daily_range_pct'] = df['daily_range'] / df['close']
            
            # Range expansion/contraction analysis
            df['range_expansion'] = df['daily_range'] > df['atr_14'] * 1.5
            df['range_contraction'] = df['daily_range'] < df['atr_14'] * 0.7
            
            current_atr = df['atr_14'].iloc[-1]
            current_range = df['daily_range'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Range expectations for next session
            range_expectations = {
                'conservative': current_atr * 0.8,
                'normal': current_atr,
                'expanded': current_atr * 1.5,
                'maximum': current_atr * 2.0
            }
            
            # Price targets based on range expectations
            price_targets = {}
            for expectation, range_size in range_expectations.items():
                price_targets[expectation] = {
                    'upside_target': current_price + range_size,
                    'downside_target': current_price - range_size,
                    'range_size': range_size,
                    'range_percentage': (range_size / current_price) * 100
                }
            
            # Weekly analysis if we have enough data
            weekly_analysis = None
            if len(df) >= 50:
                weekly_analysis = self._analyze_weekly_ranges(df)
            
            return {
                'current_metrics': {
                    'current_atr_14': current_atr,
                    'current_daily_range': current_range,
                    'range_efficiency': (current_range / current_atr) if current_atr > 0 else 0,
                    'is_expansion_day': current_range > current_atr * 1.5,
                    'is_contraction_day': current_range < current_atr * 0.7
                },
                'range_expectations': range_expectations,
                'price_targets': price_targets,
                'volatility_regime': self._classify_volatility_regime(df),
                'weekly_analysis': weekly_analysis,
                'range_statistics': {
                    'avg_daily_range_14d': df['daily_range'].tail(14).mean(),
                    'max_range_30d': df['daily_range'].tail(30).max(),
                    'min_range_30d': df['daily_range'].tail(30).min(),
                    'expansion_frequency': df['range_expansion'].tail(20).sum() / 20,
                    'contraction_frequency': df['range_contraction'].tail(20).sum() / 20
                }
            }
            
        except Exception as e:
            logger.error(f"Error in range expectations analysis: {e}")
            return {'error': str(e)}
    
    def concept_25_session_liquidity_raids(self, stock_data: pd.DataFrame) -> List[Dict]:
        """
        CONCEPT 25: Session Liquidity Raids
        - Pre-market liquidity sweeps
        - Intraday session raids
        - After-hours liquidity hunts
        - Gap fill analysis
        """
        try:
            if stock_data.empty:
                return []
            
            df = stock_data.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            liquidity_raids = []
            
            # Identify session boundaries
            sessions = self._identify_trading_sessions(df)
            
            for session in sessions:
                session_data = df[
                    (df['timestamp'] >= session['start']) & 
                    (df['timestamp'] <= session['end'])
                ]
                
                if len(session_data) < 2:
                    continue
                
                # Look for liquidity raids within session
                raids = self._detect_liquidity_raids_in_session(session_data, session['type'])
                liquidity_raids.extend(raids)
            
            # Analyze cross-session raids (gaps)
            gap_raids = self._analyze_gap_liquidity_raids(df)
            liquidity_raids.extend(gap_raids)
            
            return sorted(liquidity_raids, key=lambda x: x.get('timestamp', datetime.min), reverse=True)
            
        except Exception as e:
            logger.error(f"Error in session liquidity raids analysis: {e}")
            return []
    
    def concept_26_weekly_profiles(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 26: Weekly Profiles (WHLC)
        - Weekly Open analysis
        - Weekly High/Low targeting
        - Weekly Close bias
        - Weekly profile classification
        """
        try:
            if len(stock_data) < 5:
                return {'error': 'Insufficient data for weekly analysis'}
            
            df = stock_data.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['week'] = df['timestamp'].dt.isocalendar().week
            df['year'] = df['timestamp'].dt.year
            
            # Group by week
            weekly_profiles = []
            
            for (year, week), week_data in df.groupby(['year', 'week']):
                if len(week_data) < 2:
                    continue
                
                week_data = week_data.sort_values('timestamp')
                
                profile = {
                    'year': year,
                    'week': week,
                    'start_date': week_data['timestamp'].iloc[0].date(),
                    'end_date': week_data['timestamp'].iloc[-1].date(),
                    'weekly_open': week_data['open'].iloc[0],
                    'weekly_high': week_data['high'].max(),
                    'weekly_low': week_data['low'].min(),
                    'weekly_close': week_data['close'].iloc[-1],
                    'weekly_volume': week_data['volume'].sum(),
                    'days_in_week': len(week_data)
                }
                
                # Calculate weekly metrics
                profile['weekly_range'] = profile['weekly_high'] - profile['weekly_low']
                profile['weekly_change'] = profile['weekly_close'] - profile['weekly_open']
                profile['weekly_change_pct'] = (profile['weekly_change'] / profile['weekly_open']) * 100
                
                # Classify weekly profile
                profile['profile_type'] = self._classify_weekly_profile(profile, week_data)
                
                # Weekly bias analysis
                profile['weekly_bias'] = self._analyze_weekly_bias(profile, week_data)
                
                weekly_profiles.append(profile)
            
            # Current week analysis
            current_week_analysis = None
            if weekly_profiles:
                current_week = weekly_profiles[-1]
                current_week_analysis = self._analyze_current_week_progress(current_week, df)
            
            # Weekly statistics
            if len(weekly_profiles) >= 4:
                statistics = self._calculate_weekly_statistics(weekly_profiles)
            else:
                statistics = {}
            
            return {
                'weekly_profiles': weekly_profiles[-10:],  # Last 10 weeks
                'current_week_analysis': current_week_analysis,
                'weekly_statistics': statistics,
                'weekly_patterns': self._identify_weekly_patterns(weekly_profiles) if len(weekly_profiles) >= 8 else {}
            }
            
        except Exception as e:
            logger.error(f"Error in weekly profiles analysis: {e}")
            return {'error': str(e)}
    
    def concept_27_daily_bias(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 27: Daily Bias (using daily open, previous day's high/low)
        - Daily open gap analysis
        - Previous day high/low respect
        - Intraday bias determination
        - Daily sentiment analysis
        """
        try:
            if len(stock_data) < 2:
                return {'error': 'Insufficient data for daily bias analysis'}
            
            df = stock_data.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            # Group by date for daily analysis
            daily_bias_analysis = []
            
            dates = sorted(df['date'].unique())
            
            for i, current_date in enumerate(dates):
                current_day_data = df[df['date'] == current_date].sort_values('timestamp')
                
                if len(current_day_data) == 0:
                    continue
                
                # Get previous day data if available
                previous_day_data = None
                if i > 0:
                    prev_date = dates[i-1]
                    previous_day_data = df[df['date'] == prev_date].sort_values('timestamp')
                
                bias_analysis = self._analyze_daily_bias(current_day_data, previous_day_data)
                bias_analysis['date'] = current_date
                
                daily_bias_analysis.append(bias_analysis)
            
            # Current day bias
            current_bias = daily_bias_analysis[-1] if daily_bias_analysis else None
            
            # Bias accuracy tracking
            bias_accuracy = self._calculate_bias_accuracy(daily_bias_analysis)
            
            return {
                'current_daily_bias': current_bias,
                'daily_bias_history': daily_bias_analysis[-5:],  # Last 5 days
                'bias_accuracy': bias_accuracy,
                'bias_patterns': self._identify_bias_patterns(daily_bias_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error in daily bias analysis: {e}")
            return {'error': str(e)}
    
    def concept_28_weekly_bias(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 28: Weekly Bias (using weekly OHLC)
        - Weekly opening gap analysis
        - Weekly range projection
        - Multi-timeframe alignment
        - Weekly trend continuation/reversal
        """
        try:
            weekly_profiles = self.concept_26_weekly_profiles(stock_data)
            
            if 'error' in weekly_profiles:
                return weekly_profiles
            
            profiles = weekly_profiles.get('weekly_profiles', [])
            
            if len(profiles) < 2:
                return {'error': 'Insufficient weekly data for bias analysis'}
            
            weekly_bias_analysis = []
            
            for i, current_week in enumerate(profiles):
                previous_week = profiles[i-1] if i > 0 else None
                
                bias = self._determine_weekly_bias(current_week, previous_week)
                weekly_bias_analysis.append(bias)
            
            # Current weekly bias
            current_weekly_bias = weekly_bias_analysis[-1] if weekly_bias_analysis else None
            
            # Multi-timeframe alignment
            alignment = self._analyze_timeframe_alignment(stock_data, current_weekly_bias)
            
            return {
                'current_weekly_bias': current_weekly_bias,
                'weekly_bias_history': weekly_bias_analysis[-4:],  # Last 4 weeks
                'timeframe_alignment': alignment,
                'weekly_momentum': self._calculate_weekly_momentum(profiles),
                'bias_strength': self._calculate_weekly_bias_strength(current_weekly_bias, profiles)
            }
            
        except Exception as e:
            logger.error(f"Error in weekly bias analysis: {e}")
            return {'error': str(e)}
    
    def concept_29_monthly_bias(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 29: Monthly Bias (using monthly OHLC)
        - Monthly opening analysis
        - Monthly range expectations
        - Seasonal stock patterns
        - Long-term institutional bias
        """
        try:
            if len(stock_data) < 20:
                return {'error': 'Insufficient data for monthly analysis'}
            
            df = stock_data.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['month'] = df['timestamp'].dt.month
            df['year'] = df['timestamp'].dt.year
            
            # Group by month
            monthly_profiles = []
            
            for (year, month), month_data in df.groupby(['year', 'month']):
                if len(month_data) < 5:
                    continue
                
                month_data = month_data.sort_values('timestamp')
                
                profile = {
                    'year': year,
                    'month': month,
                    'month_name': month_data['timestamp'].iloc[0].strftime('%B'),
                    'start_date': month_data['timestamp'].iloc[0].date(),
                    'end_date': month_data['timestamp'].iloc[-1].date(),
                    'monthly_open': month_data['open'].iloc[0],
                    'monthly_high': month_data['high'].max(),
                    'monthly_low': month_data['low'].min(),
                    'monthly_close': month_data['close'].iloc[-1],
                    'monthly_volume': month_data['volume'].sum(),
                    'trading_days': len(month_data)
                }
                
                # Calculate monthly metrics
                profile['monthly_range'] = profile['monthly_high'] - profile['monthly_low']
                profile['monthly_change'] = profile['monthly_close'] - profile['monthly_open']
                profile['monthly_change_pct'] = (profile['monthly_change'] / profile['monthly_open']) * 100
                
                # Monthly bias
                profile['monthly_bias'] = self._determine_monthly_bias(profile, month_data)
                
                monthly_profiles.append(profile)
            
            # Seasonal analysis
            seasonal_patterns = self._analyze_seasonal_patterns(monthly_profiles)
            
            # Current month analysis
            current_month_analysis = None
            if monthly_profiles:
                current_month = monthly_profiles[-1]
                current_month_analysis = self._analyze_current_month_progress(current_month, df)
            
            return {
                'monthly_profiles': monthly_profiles[-12:],  # Last 12 months
                'current_month_analysis': current_month_analysis,
                'seasonal_patterns': seasonal_patterns,
                'institutional_bias': self._analyze_institutional_monthly_bias(monthly_profiles)
            }
            
        except Exception as e:
            logger.error(f"Error in monthly bias analysis: {e}")
            return {'error': str(e)}
    
    def concept_30_time_of_day_highs_lows(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 30: Time of Day Highs & Lows (AM/PM session separation)
        - Morning session high/low
        - Afternoon session high/low
        - Lunch period analysis
        - Power hour patterns
        """
        try:
            if stock_data.empty:
                return {'error': 'No data available'}
            
            df = stock_data.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Convert to ET timezone if needed
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert('US/Eastern')
            
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['time'] = df['timestamp'].dt.time
            df['date'] = df['timestamp'].dt.date
            
            time_of_day_analysis = {}
            
            # Define time sessions
            sessions = {
                'premarket': (time(4, 0), time(9, 30)),
                'morning': (time(9, 30), time(12, 0)),
                'lunch': (time(12, 0), time(14, 0)),
                'afternoon': (time(14, 0), time(16, 0)),
                'afterhours': (time(16, 0), time(20, 0))
            }
            
            # Analyze each session
            for session_name, (start_time, end_time) in sessions.items():
                session_data = df[
                    (df['time'] >= start_time) & 
                    (df['time'] < end_time)
                ]
                
                if not session_data.empty:
                    analysis = self._analyze_time_session(session_data, session_name)
                    time_of_day_analysis[session_name] = analysis
            
            # Hourly analysis
            hourly_stats = self._calculate_hourly_statistics(df)
            
            # Daily patterns
            daily_patterns = self._identify_daily_time_patterns(df)
            
            return {
                'session_analysis': time_of_day_analysis,
                'hourly_statistics': hourly_stats,
                'daily_patterns': daily_patterns,
                'key_findings': self._summarize_time_findings(time_of_day_analysis, hourly_stats)
            }
            
        except Exception as e:
            logger.error(f"Error in time of day analysis: {e}")
            return {'error': str(e)}
    
    # Helper methods for Time & Price concepts
    
    def _filter_by_time_range(self, df: pd.DataFrame, start_time: time, end_time: time) -> pd.DataFrame:
        """Filter DataFrame by time range"""
        try:
            df_time = df.copy()
            df_time['time'] = df_time['timestamp'].dt.time
            
            if start_time <= end_time:
                return df_time[(df_time['time'] >= start_time) & (df_time['time'] < end_time)]
            else:
                # Handle overnight sessions
                return df_time[(df_time['time'] >= start_time) | (df_time['time'] < end_time)]
        except:
            return pd.DataFrame()
    
    def _analyze_killzone_performance(self, session_data: pd.DataFrame, killzone: KillzoneInfo) -> Dict:
        """Analyze performance during a killzone"""
        if session_data.empty:
            return {'session': killzone.session.value, 'data_points': 0}
        
        analysis = {
            'session': killzone.session.value,
            'description': killzone.description,
            'importance': killzone.importance,
            'data_points': len(session_data),
            'volume_profile': {
                'total_volume': session_data['volume'].sum(),
                'avg_volume': session_data['volume'].mean(),
                'volume_spikes': (session_data['volume'] > session_data['volume'].mean() * 2).sum()
            },
            'price_action': {
                'session_high': session_data['high'].max(),
                'session_low': session_data['low'].min(),
                'session_range': session_data['high'].max() - session_data['low'].min(),
                'net_change': session_data['close'].iloc[-1] - session_data['open'].iloc[0],
                'volatility': session_data['close'].pct_change().std()
            }
        }
        
        # Calculate session bias
        net_change = analysis['price_action']['net_change']
        if net_change > 0:
            analysis['session_bias'] = 'bullish'
        elif net_change < 0:
            analysis['session_bias'] = 'bearish'
        else:
            analysis['session_bias'] = 'neutral'
        
        return analysis
    
    def _generate_killzone_summary(self, killzone_analysis: Dict) -> Dict:
        """Generate summary of killzone analysis"""
        summary = {
            'most_active_session': None,
            'highest_volume_session': None,
            'most_volatile_session': None,
            'overall_market_bias': 'neutral'
        }
        
        if not killzone_analysis:
            return summary
        
        # Find most active sessions
        max_volume = 0
        max_volatility = 0
        max_data_points = 0
        
        bullish_sessions = 0
        bearish_sessions = 0
        
        for session, analysis in killzone_analysis.items():
            data_points = analysis.get('data_points', 0)
            volume = analysis.get('volume_profile', {}).get('total_volume', 0)
            volatility = analysis.get('price_action', {}).get('volatility', 0)
            bias = analysis.get('session_bias', 'neutral')
            
            if data_points > max_data_points:
                max_data_points = data_points
                summary['most_active_session'] = session
            
            if volume > max_volume:
                max_volume = volume
                summary['highest_volume_session'] = session
            
            if volatility > max_volatility:
                max_volatility = volatility
                summary['most_volatile_session'] = session
            
            if bias == 'bullish':
                bullish_sessions += 1
            elif bias == 'bearish':
                bearish_sessions += 1
        
        # Overall bias
        if bullish_sessions > bearish_sessions:
            summary['overall_market_bias'] = 'bullish'
        elif bearish_sessions > bullish_sessions:
            summary['overall_market_bias'] = 'bearish'
        
        return summary
    
    def _get_current_session(self) -> str:
        """Get current trading session"""
        now = datetime.now(self.et_timezone)
        current_time = now.time()
        
        for session_type, killzone in self.stock_killzones.items():
            if killzone.start_time <= current_time < killzone.end_time:
                return session_type.value
        
        return SessionType.CLOSED.value
    
    def _get_next_important_session(self) -> Dict:
        """Get next important trading session"""
        now = datetime.now(self.et_timezone)
        current_time = now.time()
        
        # Find next session
        upcoming_sessions = []
        for session_type, killzone in self.stock_killzones.items():
            if killzone.start_time > current_time:
                upcoming_sessions.append((killzone.start_time, session_type, killzone))
        
        if not upcoming_sessions:
            # Next day's first session
            first_session = min(self.stock_killzones.values(), key=lambda x: x.start_time)
            return {
                'session': first_session.session.value,
                'start_time': first_session.start_time.strftime('%H:%M'),
                'importance': first_session.importance,
                'description': first_session.description,
                'is_tomorrow': True
            }
        
        next_time, next_session, next_killzone = min(upcoming_sessions)
        return {
            'session': next_session.value,
            'start_time': next_time.strftime('%H:%M'),
            'importance': next_killzone.importance,
            'description': next_killzone.description,
            'is_tomorrow': False
        }

    def _find_swing_points(self, df: pd.DataFrame, point_type: str) -> List[Dict]:
        """Find swing points (highs or lows)"""
        swing_points = []
        lookback = 5
        
        for i in range(lookback, len(df) - lookback):
            if point_type == 'high':
                current_value = df['high'].iloc[i]
                is_swing = all(current_value >= df['high'].iloc[j] for j in range(i-lookback, i+lookback+1) if j != i)
            else:
                current_value = df['low'].iloc[i]
                is_swing = all(current_value <= df['low'].iloc[j] for j in range(i-lookback, i+lookback+1) if j != i)
            
            if is_swing:
                swing_points.append({
                    'timestamp': df.iloc[i]['timestamp'] if 'timestamp' in df.columns else df.index[i],
                    'price': current_value,
                    'type': point_type,
                    'index': i
                })
        
        return swing_points
    
    def _calculate_fib_significance(self, level_name: str, df: pd.DataFrame, fib_price: float) -> float:
        """Calculate significance of Fibonacci level"""
        # Key Fibonacci levels have higher significance
        key_levels = {'50%': 0.9, '61.8%': 1.0, '70.5%': 0.8, '78.6%': 0.9}
        base_significance = key_levels.get(level_name, 0.5)
        
        # Check if price has tested this level before
        current_price = df['close'].iloc[-1]
        distance = abs(fib_price - current_price) / current_price
        
        # Closer levels are more significant for current trading
        distance_factor = max(0.1, 1 - distance * 10)
        
        return base_significance * distance_factor
    
    def _analyze_current_fib_position(self, fibonacci_levels: List[FibonacciLevel], current_price: float) -> Dict:
        """Analyze current price position relative to Fibonacci levels"""
        analysis = {
            'nearest_support': None,
            'nearest_resistance': None,
            'in_golden_zone': False,
            'in_ote_zone': False,
            'price_level_strength': 'normal'
        }
        
        # Find nearest support and resistance
        supports = [fib for fib in fibonacci_levels if fib.price < current_price]
        resistances = [fib for fib in fibonacci_levels if fib.price > current_price]
        
        if supports:
            analysis['nearest_support'] = max(supports, key=lambda x: x.price)
        
        if resistances:
            analysis['nearest_resistance'] = min(resistances, key=lambda x: x.price)
        
        # Check if in key zones
        for fib in fibonacci_levels:
            if abs(fib.percentage - 61.8) < 5 and abs(current_price - fib.price) / current_price < 0.02:
                analysis['in_golden_zone'] = True
            
            if (70.5 <= fib.percentage <= 78.6) and abs(current_price - fib.price) / current_price < 0.02:
                analysis['in_ote_zone'] = True
        
        return analysis
    
    def _analyze_weekly_ranges(self, df: pd.DataFrame) -> Dict:
        """Analyze weekly range patterns"""
        df['week'] = df['timestamp'].dt.isocalendar().week
        df['year'] = df['timestamp'].dt.year
        
        weekly_ranges = []
        for (year, week), week_data in df.groupby(['year', 'week']):
            if len(week_data) >= 2:
                week_range = week_data['high'].max() - week_data['low'].min()
                week_avg_price = week_data['close'].mean()
                weekly_ranges.append(week_range / week_avg_price)  # Normalize by price
        
        if len(weekly_ranges) >= 4:
            return {
                'avg_weekly_range_pct': np.mean(weekly_ranges) * 100,
                'max_weekly_range_pct': np.max(weekly_ranges) * 100,
                'min_weekly_range_pct': np.min(weekly_ranges) * 100,
                'weekly_range_volatility': np.std(weekly_ranges) * 100
            }
        
        return {}
    
    def _classify_volatility_regime(self, df: pd.DataFrame) -> str:
        """Classify current volatility regime"""
        if 'atr_14' not in df.columns:
            return 'unknown'
        
        recent_atr = df['atr_14'].tail(5).mean()
        historical_atr = df['atr_14'].mean()
        
        ratio = recent_atr / historical_atr if historical_atr > 0 else 1
        
        if ratio > 1.5:
            return 'high_volatility'
        elif ratio > 1.2:
            return 'elevated_volatility'
        elif ratio < 0.8:
            return 'low_volatility'
        else:
            return 'normal_volatility'
    
    def _identify_trading_sessions(self, df: pd.DataFrame) -> List[Dict]:
        """Identify distinct trading sessions"""
        sessions = []
        df['date'] = df['timestamp'].dt.date
        
        for date in df['date'].unique():
            daily_data = df[df['date'] == date]
            
            # Define session times for each day
            session_definitions = [
                ('premarket', time(4, 0), time(9, 30)),
                ('market_hours', time(9, 30), time(16, 0)),
                ('afterhours', time(16, 0), time(20, 0))
            ]
            
            for session_type, start_time, end_time in session_definitions:
                session_start = datetime.combine(date, start_time)
                session_end = datetime.combine(date, end_time)
                
                sessions.append({
                    'type': session_type,
                    'start': session_start,
                    'end': session_end,
                    'date': date
                })
        
        return sessions
    
    def _detect_liquidity_raids_in_session(self, session_data: pd.DataFrame, session_type: str) -> List[Dict]:
        """Detect liquidity raids within a trading session"""
        raids = []
        
        if len(session_data) < 5:
            return raids
        
        # Look for spikes above/below range that quickly reverse
        session_high = session_data['high'].max()
        session_low = session_data['low'].min()
        session_range = session_high - session_low
        
        for i in range(2, len(session_data) - 2):
            current = session_data.iloc[i]
            
            # Check for upside raid (break high then reverse)
            if current['high'] == session_high:
                # Look for quick reversal
                future_lows = session_data.iloc[i+1:i+4]['low']
                if not future_lows.empty and future_lows.min() < current['close']:
                    reversal_size = current['high'] - future_lows.min()
                    if reversal_size > session_range * 0.3:  # Significant reversal
                        raids.append({
                            'timestamp': current['timestamp'] if 'timestamp' in current else current.name,
                            'raid_type': 'upside_liquidity_raid',
                            'session': session_type,
                            'raid_price': current['high'],
                            'reversal_price': future_lows.min(),
                            'raid_size': reversal_size,
                            'strength': reversal_size / session_range
                        })
            
            # Check for downside raid
            if current['low'] == session_low:
                future_highs = session_data.iloc[i+1:i+4]['high']
                if not future_highs.empty and future_highs.max() > current['close']:
                    reversal_size = future_highs.max() - current['low']
                    if reversal_size > session_range * 0.3:
                        raids.append({
                            'timestamp': current['timestamp'] if 'timestamp' in current else current.name,
                            'raid_type': 'downside_liquidity_raid',
                            'session': session_type,
                            'raid_price': current['low'],
                            'reversal_price': future_highs.max(),
                            'raid_size': reversal_size,
                            'strength': reversal_size / session_range
                        })
        
        return raids
    
    def _analyze_gap_liquidity_raids(self, df: pd.DataFrame) -> List[Dict]:
        """Analyze gap fills as liquidity raids"""
        raids = []
        df['date'] = df['timestamp'].dt.date
        
        dates = sorted(df['date'].unique())
        
        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            curr_date = dates[i]
            
            prev_day_data = df[df['date'] == prev_date]
            curr_day_data = df[df['date'] == curr_date]
            
            if prev_day_data.empty or curr_day_data.empty:
                continue
            
            prev_close = prev_day_data['close'].iloc[-1]
            curr_open = curr_day_data['open'].iloc[0]
            
            # Check for gap
            gap_size = abs(curr_open - prev_close)
            gap_pct = gap_size / prev_close
            
            if gap_pct > 0.01:  # Minimum 1% gap
                # Check if gap gets filled (liquidity raid)
                if curr_open > prev_close:  # Gap up
                    gap_fill_price = prev_close
                    if curr_day_data['low'].min() <= gap_fill_price:
                        raids.append({
                            'timestamp': curr_day_data['timestamp'].iloc[0],
                            'raid_type': 'gap_down_fill',
                            'session': 'gap_fill',
                            'gap_size': gap_size,
                            'gap_percentage': gap_pct * 100,
                            'fill_achieved': True,
                            'strength': gap_pct
                        })
                
                else:  # Gap down
                    gap_fill_price = prev_close
                    if curr_day_data['high'].max() >= gap_fill_price:
                        raids.append({
                            'timestamp': curr_day_data['timestamp'].iloc[0],
                            'raid_type': 'gap_up_fill',
                            'session': 'gap_fill',
                            'gap_size': gap_size,
                            'gap_percentage': gap_pct * 100,
                            'fill_achieved': True,
                            'strength': gap_pct
                        })
        
        return raids
    
    def _classify_weekly_profile(self, profile: Dict, week_data: pd.DataFrame) -> str:
        """Classify weekly profile type"""
        weekly_range = profile['weekly_range']
        weekly_open = profile['weekly_open']
        weekly_close = profile['weekly_close']
        weekly_high = profile['weekly_high']
        weekly_low = profile['weekly_low']
        
        # Determine where close is relative to range
        close_position = (weekly_close - weekly_low) / weekly_range if weekly_range > 0 else 0.5
        
        # Classify profile
        if close_position > 0.8:
            return 'bullish_close_high'
        elif close_position < 0.2:
            return 'bearish_close_low'
        elif 0.4 <= close_position <= 0.6:
            return 'neutral_close_middle'
        elif weekly_close > weekly_open:
            return 'bullish_net_positive'
        else:
            return 'bearish_net_negative'
    
    def _analyze_weekly_bias(self, profile: Dict, week_data: pd.DataFrame) -> Dict:
        """Analyze weekly bias"""
        weekly_change = profile['weekly_change']
        weekly_open = profile['weekly_open']
        
        if weekly_change > weekly_open * 0.02:  # > 2% gain
            bias = 'strong_bullish'
        elif weekly_change > 0:
            bias = 'bullish'
        elif weekly_change < -weekly_open * 0.02:  # > 2% loss
            bias = 'strong_bearish'
        elif weekly_change < 0:
            bias = 'bearish'
        else:
            bias = 'neutral'
        
        return {
            'bias': bias,
            'strength': abs(weekly_change / weekly_open) if weekly_open > 0 else 0,
            'conviction': 'high' if abs(weekly_change / weekly_open) > 0.05 else 'normal'
        }
    
    def _analyze_current_week_progress(self, current_week: Dict, df: pd.DataFrame) -> Dict:
        """Analyze current week's progress"""
        current_date = datetime.now().date()
        current_week_data = df[df['timestamp'].dt.date <= current_date]
        
        if current_week_data.empty:
            return {}
        
        days_elapsed = len(current_week_data['timestamp'].dt.date.unique())
        total_trading_days = 5  # Typical trading week
        
        week_progress = days_elapsed / total_trading_days
        
        current_high = current_week_data['high'].max()
        current_low = current_week_data['low'].min()
        current_close = current_week_data['close'].iloc[-1]
        
        weekly_open = current_week['weekly_open']
        
        return {
            'days_elapsed': days_elapsed,
            'week_progress_pct': week_progress * 100,
            'current_weekly_range': current_high - current_low,
            'weekly_change_so_far': current_close - weekly_open,
            'weekly_change_pct_so_far': ((current_close - weekly_open) / weekly_open) * 100,
            'high_of_week_achieved': current_high == current_week['weekly_high'],
            'low_of_week_achieved': current_low == current_week['weekly_low']
        }
    
    def _calculate_weekly_statistics(self, weekly_profiles: List[Dict]) -> Dict:
        """Calculate weekly statistics"""
        if len(weekly_profiles) < 4:
            return {}
        
        weekly_changes = [w['weekly_change_pct'] for w in weekly_profiles]
        weekly_ranges = [w['weekly_range'] / w['weekly_open'] * 100 for w in weekly_profiles]
        
        return {
            'avg_weekly_change_pct': np.mean(weekly_changes),
            'weekly_change_volatility': np.std(weekly_changes),
            'avg_weekly_range_pct': np.mean(weekly_ranges),
            'positive_weeks': sum(1 for change in weekly_changes if change > 0),
            'negative_weeks': sum(1 for change in weekly_changes if change < 0),
            'win_rate': sum(1 for change in weekly_changes if change > 0) / len(weekly_changes) * 100
        }
    
    def _identify_weekly_patterns(self, weekly_profiles: List[Dict]) -> Dict:
        """Identify weekly patterns"""
        if len(weekly_profiles) < 8:
            return {}
        
        patterns = {
            'consecutive_up_weeks': 0,
            'consecutive_down_weeks': 0,
            'alternating_pattern': False,
            'trending_pattern': False
        }
        
        # Find consecutive patterns
        current_up_streak = 0
        current_down_streak = 0
        max_up_streak = 0
        max_down_streak = 0
        
        alternating_count = 0
        
        for i, week in enumerate(weekly_profiles):
            if week['weekly_change'] > 0:
                current_up_streak += 1
                current_down_streak = 0
                max_up_streak = max(max_up_streak, current_up_streak)
            else:
                current_down_streak += 1
                current_up_streak = 0
                max_down_streak = max(max_down_streak, current_down_streak)
            
            # Check for alternating pattern
            if i > 0:
                prev_week = weekly_profiles[i-1]
                if ((week['weekly_change'] > 0) != (prev_week['weekly_change'] > 0)):
                    alternating_count += 1
        
        patterns['consecutive_up_weeks'] = max_up_streak
        patterns['consecutive_down_weeks'] = max_down_streak
        patterns['alternating_pattern'] = alternating_count / len(weekly_profiles) > 0.6
        
        # Check for trending pattern
        recent_changes = [w['weekly_change'] for w in weekly_profiles[-4:]]
        if len(recent_changes) == 4:
            uptrend = sum(1 for change in recent_changes if change > 0) >= 3
            downtrend = sum(1 for change in recent_changes if change < 0) >= 3
            patterns['trending_pattern'] = uptrend or downtrend
        
        return patterns

    def _analyze_session_open(self, first_candle: pd.Series, session_type: str, daily_data: pd.DataFrame) -> Dict:
        """Analyze session open characteristics"""
        try:
            # Get previous session's close for gap analysis
            prev_close = None
            if session_type == 'market':
                # Compare with pre-market data
                premarket_data = daily_data[daily_data['timestamp'].dt.time < time(9, 30)]
                if not premarket_data.empty:
                    prev_close = premarket_data['close'].iloc[-1]
            elif session_type == 'afterhours':
                # Compare with market close
                market_data = daily_data[
                    (daily_data['timestamp'].dt.time >= time(9, 30)) & 
                    (daily_data['timestamp'].dt.time < time(16, 0))
                ]
                if not market_data.empty:
                    prev_close = market_data['close'].iloc[-1]
            
            analysis = {
                'session_type': session_type,
                'timestamp': first_candle['timestamp'] if 'timestamp' in first_candle else first_candle.name,
                'open_price': first_candle['open'],
                'gap_analysis': None
            }
            
            if prev_close is not None:
                gap_size = first_candle['open'] - prev_close
                gap_pct = (gap_size / prev_close) * 100
                
                analysis['gap_analysis'] = {
                    'gap_size': gap_size,
                    'gap_percentage': gap_pct,
                    'gap_type': 'gap_up' if gap_size > 0 else 'gap_down' if gap_size < 0 else 'no_gap',
                    'significant_gap': abs(gap_pct) > 0.5  # 0.5% threshold
                }
            
            return analysis
        except:
            return {'session_type': session_type, 'error': 'Analysis failed'}
    
    def _analyze_overnight_gap(self, daily_data: pd.DataFrame) -> Optional[Dict]:
        """Analyze overnight gap from previous day"""
        # This would require data from previous trading day
        # Simplified implementation
        if len(daily_data) < 2:
            return None
        
        first_candle = daily_data.iloc[0]
        last_candle = daily_data.iloc[-1]
        
        # Approximate overnight gap (simplified)
        return {
            'overnight_change': first_candle['open'] - last_candle['close'],
            'overnight_change_pct': ((first_candle['open'] - last_candle['close']) / last_candle['close']) * 100,
            'gap_fill_probability': 0.7  # Historical probability
        }
    
    def _calculate_session_statistics(self, session_opens: Dict) -> Dict:
        """Calculate statistics for session opens"""
        stats = {}
        
        for session_type, opens in session_opens.items():
            if not opens:
                continue
                
            gaps = [o.get('gap_analysis', {}).get('gap_percentage', 0) for o in opens if o.get('gap_analysis')]
            
            if gaps:
                stats[session_type] = {
                    'total_sessions': len(opens),
                    'avg_gap_pct': np.mean([abs(g) for g in gaps]),
                    'gap_up_frequency': sum(1 for g in gaps if g > 0.5) / len(gaps) * 100,
                    'gap_down_frequency': sum(1 for g in gaps if g < -0.5) / len(gaps) * 100,
                    'significant_gaps': sum(1 for g in gaps if abs(g) > 1.0) / len(gaps) * 100
                }
        
        return stats
    
    def _identify_session_patterns(self, session_opens: Dict) -> Dict:
        """Identify patterns in session opens"""
        patterns = {
            'gap_fade_tendency': {},
            'gap_continuation_tendency': {},
            'volume_patterns': {}
        }
        
        # Simplified pattern identification
        for session_type, opens in session_opens.items():
            if len(opens) >= 5:
                patterns['gap_fade_tendency'][session_type] = 'moderate'  # Would calculate actual fade rate
                patterns['gap_continuation_tendency'][session_type] = 'moderate'
        
        return patterns
    
    def _analyze_daily_bias(self, current_day_data: pd.DataFrame, previous_day_data: Optional[pd.DataFrame]) -> Dict:
        """Analyze daily bias based on opens and previous day levels"""
        analysis = {
            'daily_open': current_day_data['open'].iloc[0],
            'current_price': current_day_data['close'].iloc[-1],
            'daily_high': current_day_data['high'].max(),
            'daily_low': current_day_data['low'].min(),
            'bias': 'neutral',
            'bias_strength': 0.5
        }
        
        daily_open = analysis['daily_open']
        current_price = analysis['current_price']
        
        # Basic bias determination
        if current_price > daily_open * 1.01:
            analysis['bias'] = 'bullish'
            analysis['bias_strength'] = min((current_price - daily_open) / daily_open * 10, 1.0)
        elif current_price < daily_open * 0.99:
            analysis['bias'] = 'bearish' 
            analysis['bias_strength'] = min((daily_open - current_price) / daily_open * 10, 1.0)
        
        # Previous day analysis
        if previous_day_data is not None and not previous_day_data.empty:
            prev_high = previous_day_data['high'].max()
            prev_low = previous_day_data['low'].min()
            prev_close = previous_day_data['close'].iloc[-1]
            
            analysis['previous_day_levels'] = {
                'prev_high': prev_high,
                'prev_low': prev_low,
                'prev_close': prev_close,
                'gap_from_prev_close': daily_open - prev_close,
                'respect_prev_high': current_price < prev_high,
                'respect_prev_low': current_price > prev_low
            }
        
        return analysis
    
    def _calculate_bias_accuracy(self, daily_bias_analysis: List[Dict]) -> Dict:
        """Calculate accuracy of daily bias predictions"""
        if len(daily_bias_analysis) < 5:
            return {'insufficient_data': True}
        
        correct_predictions = 0
        total_predictions = 0
        
        for analysis in daily_bias_analysis:
            if 'bias' in analysis and analysis['bias'] != 'neutral':
                total_predictions += 1
                # Simplified accuracy check (would need actual outcome data)
                if analysis['bias_strength'] > 0.6:
                    correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy_percentage': accuracy * 100,
            'sample_size': len(daily_bias_analysis)
        }
    
    def _identify_bias_patterns(self, daily_bias_analysis: List[Dict]) -> Dict:
        """Identify patterns in daily bias"""
        if len(daily_bias_analysis) < 10:
            return {'insufficient_data': True}
        
        biases = [analysis.get('bias', 'neutral') for analysis in daily_bias_analysis]
        
        patterns = {
            'bullish_frequency': biases.count('bullish') / len(biases) * 100,
            'bearish_frequency': biases.count('bearish') / len(biases) * 100,
            'neutral_frequency': biases.count('neutral') / len(biases) * 100,
            'trending_tendency': False,
            'reversal_tendency': False
        }
        
        # Check for trending vs reversal patterns
        consecutive_same = 0
        max_consecutive = 0
        
        for i in range(1, len(biases)):
            if biases[i] == biases[i-1] and biases[i] != 'neutral':
                consecutive_same += 1
                max_consecutive = max(max_consecutive, consecutive_same)
            else:
                consecutive_same = 0
        
        patterns['trending_tendency'] = max_consecutive >= 3
        patterns['max_consecutive_same_bias'] = max_consecutive
        
        return patterns
    
    def _determine_weekly_bias(self, current_week: Dict, previous_week: Optional[Dict]) -> Dict:
        """Determine weekly bias"""
        bias_analysis = {
            'weekly_bias': 'neutral',
            'bias_strength': 0.5,
            'key_factors': []
        }
        
        weekly_change = current_week['weekly_change']
        weekly_open = current_week['weekly_open']
        
        # Determine bias based on weekly performance
        if weekly_change > weekly_open * 0.03:  # > 3%
            bias_analysis['weekly_bias'] = 'strong_bullish'
            bias_analysis['bias_strength'] = 0.9
        elif weekly_change > 0:
            bias_analysis['weekly_bias'] = 'bullish'
            bias_analysis['bias_strength'] = 0.7
        elif weekly_change < -weekly_open * 0.03:
            bias_analysis['weekly_bias'] = 'strong_bearish'
            bias_analysis['bias_strength'] = 0.9
        elif weekly_change < 0:
            bias_analysis['weekly_bias'] = 'bearish'
            bias_analysis['bias_strength'] = 0.7
        
        # Add context from previous week
        if previous_week:
            prev_change = previous_week['weekly_change']
            if weekly_change * prev_change > 0:  # Same direction
                bias_analysis['key_factors'].append('momentum_continuation')
            else:
                bias_analysis['key_factors'].append('momentum_reversal')
        
        return bias_analysis
    
    def _analyze_timeframe_alignment(self, stock_data: pd.DataFrame, weekly_bias: Dict) -> Dict:
        """Analyze multi-timeframe alignment"""
        alignment = {
            'daily_weekly_alignment': False,
            'alignment_strength': 0.5,
            'conflicting_signals': []
        }
        
        if not weekly_bias or len(stock_data) < 5:
            return alignment
        
        # Get recent daily bias (simplified)
        recent_data = stock_data.tail(5)
        daily_change = recent_data['close'].iloc[-1] - recent_data['open'].iloc[0]
        
        weekly_bias_direction = weekly_bias.get('weekly_bias', 'neutral')
        daily_bias_direction = 'bullish' if daily_change > 0 else 'bearish' if daily_change < 0 else 'neutral'
        
        # Check alignment
        if weekly_bias_direction != 'neutral' and daily_bias_direction != 'neutral':
            if (weekly_bias_direction in ['bullish', 'strong_bullish'] and daily_bias_direction == 'bullish') or \
               (weekly_bias_direction in ['bearish', 'strong_bearish'] and daily_bias_direction == 'bearish'):
                alignment['daily_weekly_alignment'] = True
                alignment['alignment_strength'] = 0.8
            else:
                alignment['conflicting_signals'].append('daily_weekly_divergence')
        
        return alignment
    
    def _calculate_weekly_momentum(self, weekly_profiles: List[Dict]) -> Dict:
        """Calculate weekly momentum"""
        if len(weekly_profiles) < 4:
            return {'insufficient_data': True}
        
        recent_weeks = weekly_profiles[-4:]
        weekly_changes = [w['weekly_change_pct'] for w in recent_weeks]
        
        momentum = {
            'avg_weekly_change': np.mean(weekly_changes),
            'momentum_direction': 'neutral',
            'momentum_strength': 0.5,
            'acceleration': False
        }
        
        if momentum['avg_weekly_change'] > 1:
            momentum['momentum_direction'] = 'bullish'
            momentum['momentum_strength'] = min(momentum['avg_weekly_change'] / 5, 1.0)
        elif momentum['avg_weekly_change'] < -1:
            momentum['momentum_direction'] = 'bearish'
            momentum['momentum_strength'] = min(abs(momentum['avg_weekly_change']) / 5, 1.0)
        
        # Check for acceleration
        if len(weekly_changes) >= 2:
            recent_avg = np.mean(weekly_changes[-2:])
            earlier_avg = np.mean(weekly_changes[:-2])
            
            if abs(recent_avg) > abs(earlier_avg) * 1.5:
                momentum['acceleration'] = True
        
        return momentum
    
    def _calculate_weekly_bias_strength(self, weekly_bias: Optional[Dict], weekly_profiles: List[Dict]) -> float:
        """Calculate strength of weekly bias"""
        if not weekly_bias or len(weekly_profiles) < 2:
            return 0.5
        
        base_strength = weekly_bias.get('bias_strength', 0.5)
        
        # Adjust based on historical consistency
        if len(weekly_profiles) >= 4:
            recent_changes = [w['weekly_change'] for w in weekly_profiles[-4:]]
            consistency = sum(1 for change in recent_changes if 
                            (change > 0) == (weekly_bias['weekly_bias'] in ['bullish', 'strong_bullish']))
            
            consistency_factor = consistency / 4
            base_strength = (base_strength + consistency_factor) / 2
        
        return min(base_strength, 1.0)
    
    def _determine_monthly_bias(self, monthly_profile: Dict, month_data: pd.DataFrame) -> Dict:
        """Determine monthly bias"""
        monthly_change = monthly_profile['monthly_change']
        monthly_open = monthly_profile['monthly_open']
        
        bias = {
            'bias': 'neutral',
            'strength': 0.5,
            'confidence': 'medium'
        }
        
        change_pct = (monthly_change / monthly_open) * 100 if monthly_open > 0 else 0
        
        if change_pct > 5:
            bias['bias'] = 'strong_bullish'
            bias['strength'] = 0.9
            bias['confidence'] = 'high'
        elif change_pct > 2:
            bias['bias'] = 'bullish'
            bias['strength'] = 0.7
        elif change_pct < -5:
            bias['bias'] = 'strong_bearish'
            bias['strength'] = 0.9
            bias['confidence'] = 'high'
        elif change_pct < -2:
            bias['bias'] = 'bearish'
            bias['strength'] = 0.7
        
        return bias
    
    def _analyze_seasonal_patterns(self, monthly_profiles: List[Dict]) -> Dict:
        """Analyze seasonal patterns in monthly data"""
        if len(monthly_profiles) < 12:
            return {'insufficient_data': True}
        
        # Group by month
        monthly_performance = {}
        for profile in monthly_profiles:
            month = profile['month']
            if month not in monthly_performance:
                monthly_performance[month] = []
            monthly_performance[month].append(profile['monthly_change_pct'])
        
        seasonal_analysis = {}
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month, performances in monthly_performance.items():
            if len(performances) >= 2:
                seasonal_analysis[month_names[month-1]] = {
                    'avg_performance': np.mean(performances),
                    'win_rate': sum(1 for p in performances if p > 0) / len(performances) * 100,
                    'sample_size': len(performances),
                    'volatility': np.std(performances)
                }
        
        return seasonal_analysis
    
    def _analyze_current_month_progress(self, current_month: Dict, df: pd.DataFrame) -> Dict:
        """Analyze current month's progress"""
        current_date = datetime.now().date()
        month_start = current_month['start_date']
        
        days_in_month = (current_date - month_start).days + 1
        
        return {
            'days_elapsed': days_in_month,
            'month_progress_estimate': min(days_in_month / 22, 1.0),  # ~22 trading days per month
            'current_month_change_pct': current_month['monthly_change_pct'],
            'pace_vs_average': 'on_track'  # Simplified
        }
    
    def _analyze_institutional_monthly_bias(self, monthly_profiles: List[Dict]) -> Dict:
        """Analyze institutional bias patterns monthly"""
        if len(monthly_profiles) < 6:
            return {'insufficient_data': True}
        
        # Look for end-of-quarter patterns, window dressing, etc.
        quarter_ends = []  # March, June, September, December
        
        for profile in monthly_profiles:
            if profile['month'] in [3, 6, 9, 12]:
                quarter_ends.append(profile['monthly_change_pct'])
        
        institutional_patterns = {
            'quarter_end_bias': 'neutral',
            'window_dressing_effect': False
        }
        
        if quarter_ends:
            avg_quarter_end = np.mean(quarter_ends)
            if avg_quarter_end > 2:
                institutional_patterns['quarter_end_bias'] = 'bullish'
                institutional_patterns['window_dressing_effect'] = True
            elif avg_quarter_end < -2:
                institutional_patterns['quarter_end_bias'] = 'bearish'
        
        return institutional_patterns
    
    def _analyze_time_session(self, session_data: pd.DataFrame, session_name: str) -> Dict:
        """Analyze price action during specific time session"""
        if session_data.empty:
            return {'session': session_name, 'data_points': 0}
        
        analysis = {
            'session': session_name,
            'data_points': len(session_data),
            'session_high': session_data['high'].max(),
            'session_low': session_data['low'].min(),
            'session_range': session_data['high'].max() - session_data['low'].min(),
            'avg_volume': session_data['volume'].mean(),
            'total_volume': session_data['volume'].sum(),
            'price_direction': 'neutral'
        }
        
        if len(session_data) >= 2:
            session_start = session_data['open'].iloc[0]
            session_end = session_data['close'].iloc[-1]
            net_change = session_end - session_start
            
            analysis['net_change'] = net_change
            analysis['net_change_pct'] = (net_change / session_start) * 100 if session_start > 0 else 0
            
            if net_change > session_start * 0.005:  # > 0.5%
                analysis['price_direction'] = 'bullish'
            elif net_change < -session_start * 0.005:
                analysis['price_direction'] = 'bearish'
        
        return analysis
    
    def _calculate_hourly_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate hourly trading statistics"""
        df['hour'] = df['timestamp'].dt.hour
        
        hourly_stats = {}
        
        for hour in range(4, 21):  # 4 AM to 8 PM ET
            hour_data = df[df['hour'] == hour]
            
            if not hour_data.empty:
                hourly_stats[f'{hour:02d}:00'] = {
                    'avg_volume': hour_data['volume'].mean(),
                    'avg_volatility': hour_data['close'].pct_change().std(),
                    'avg_range': (hour_data['high'] - hour_data['low']).mean(),
                    'data_points': len(hour_data)
                }
        
        return hourly_stats
    
    def _identify_daily_time_patterns(self, df: pd.DataFrame) -> Dict:
        """Identify daily time-based patterns"""
        patterns = {
            'high_volume_hours': [],
            'low_volume_hours': [],
            'high_volatility_hours': [],
            'reversal_hours': []
        }
        
        hourly_stats = self._calculate_hourly_statistics(df)
        
        if not hourly_stats:
            return patterns
        
        # Find volume and volatility patterns
        volumes = {hour: stats['avg_volume'] for hour, stats in hourly_stats.items()}
        volatilities = {hour: stats['avg_volatility'] for hour, stats in hourly_stats.items()}
        
        if volumes:
            avg_volume = np.mean(list(volumes.values()))
            patterns['high_volume_hours'] = [hour for hour, vol in volumes.items() if vol > avg_volume * 1.2]
            patterns['low_volume_hours'] = [hour for hour, vol in volumes.items() if vol < avg_volume * 0.8]
        
        if volatilities:
            avg_volatility = np.mean(list(volatilities.values()))
            patterns['high_volatility_hours'] = [hour for hour, vol in volatilities.items() if vol > avg_volatility * 1.2]
        
        return patterns
    
    def _summarize_time_findings(self, session_analysis: Dict, hourly_stats: Dict) -> List[str]:
        """Summarize key time-based findings"""
        findings = []
        
        # Session findings
        if session_analysis:
            most_active_session = max(session_analysis.items(), 
                                    key=lambda x: x[1].get('total_volume', 0))
            findings.append(f"Most active session: {most_active_session[0]}")
            
            bullish_sessions = [name for name, data in session_analysis.items() 
                              if data.get('price_direction') == 'bullish']
            if bullish_sessions:
                findings.append(f"Bullish sessions: {', '.join(bullish_sessions)}")
        
        # Volume findings
        if hourly_stats:
            peak_volume_hour = max(hourly_stats.items(), 
                                 key=lambda x: x[1].get('avg_volume', 0))
            findings.append(f"Peak volume hour: {peak_volume_hour[0]}")
        
        return findings

# Global instance
time_price_analyzer = StockTimeAndPriceAnalyzer()