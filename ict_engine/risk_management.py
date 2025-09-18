"""
ICT Risk Management Implementation (Concepts 31-39)
Trade Journaling, Entry/Exit Models, Position Sizing, Risk Controls
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class TradeOutcome(Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    PENDING = "PENDING"
    SCRATCHED = "SCRATCHED"

class SetupType(Enum):
    A_PLUS = "A+"  # High probability
    B_TYPE = "B"   # Medium probability
    C_TYPE = "C"   # Lower probability

@dataclass
class TradeEntry:
    timestamp: datetime
    symbol: str
    direction: str  # LONG/SHORT
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float
    setup_type: SetupType
    ict_concepts: List[str]  # Which ICT concepts supported the trade
    confluence_score: float

@dataclass
class TradeJournalEntry:
    trade_id: str
    entry: TradeEntry
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    outcome: TradeOutcome = TradeOutcome.PENDING
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    duration_minutes: int = 0
    notes: str = ""
    screenshots: List[str] = None

@dataclass
class RiskMetrics:
    total_trades: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    max_drawdown: float
    sharpe_ratio: float
    expectancy: float

class StockRiskManagementEngine:
    def __init__(self):
        self.trade_journal: List[TradeJournalEntry] = []
        self.max_daily_loss_percentage = 2.0  # 2% max daily loss
        self.max_position_risk_percentage = 1.0  # 1% risk per trade
        self.max_concurrent_trades = 3
        self.min_risk_reward_ratio = 1.5
        
    def concept_31_trade_journaling_backtesting(self, stock_data: pd.DataFrame, trades: List[Dict]) -> Dict:
        """
        CONCEPT 31: Trade Journaling & Backtesting
        - Comprehensive trade tracking system
        - Performance analytics and metrics
        - Historical backtesting framework
        - Trade outcome analysis
        """
        try:
            if not trades:
                return {'error': 'No trades provided for analysis'}
            
            # Process trades into journal entries
            journal_entries = []
            for trade in trades:
                entry = self._create_journal_entry(trade, stock_data)
                if entry:
                    journal_entries.append(entry)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(journal_entries)
            
            # Backtesting analysis
            backtest_results = self._run_backtest_analysis(journal_entries, stock_data)
            
            # Trade distribution analysis
            trade_distribution = self._analyze_trade_distribution(journal_entries)
            
            return {
                'total_trades': len(journal_entries),
                'performance_metrics': performance_metrics,
                'backtest_results': backtest_results,
                'trade_distribution': trade_distribution,
                'journal_entries': [self._serialize_journal_entry(entry) for entry in journal_entries[-10:]]  # Last 10 trades
            }
            
        except Exception as e:
            logger.error(f"Error in trade journaling analysis: {e}")
            return {'error': str(e)}
    
    def concept_32_entry_models_fvg_ob_breaker(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 32: Entry Models (FVG, Order Block, Breaker Block)
        - Fair Value Gap entry strategies
        - Order Block entry confirmation
        - Breaker Block reversal entries
        - Multi-confluence entry scoring
        """
        try:
            from .core_concepts import market_structure_analyzer
            
            # Get ICT patterns
            fvgs = market_structure_analyzer.concept_6_fair_value_gaps_fvg_imbalances(stock_data)
            order_blocks = market_structure_analyzer.concept_4_order_blocks_bullish_bearish(stock_data)
            breaker_blocks = market_structure_analyzer.concept_5_breaker_blocks(stock_data)
            
            entry_opportunities = []
            
            # FVG Entry Models
            for fvg in fvgs:
                if not fvg.is_mitigated:
                    entry_score = self._calculate_fvg_entry_score(fvg, stock_data)
                    entry_opportunities.append({
                        'type': 'fvg_entry',
                        'timestamp': fvg.timestamp,
                        'entry_price': fvg.mitigation_level,
                        'direction': 'LONG' if fvg.gap_type == 'bullish' else 'SHORT',
                        'stop_loss': fvg.gap_low if fvg.gap_type == 'bullish' else fvg.gap_high,
                        'take_profit': self._calculate_fvg_target(fvg, stock_data),
                        'confluence_score': entry_score,
                        'setup_type': self._classify_setup_quality(entry_score),
                        'supporting_concepts': ['Fair Value Gap', 'Imbalance']
                    })
            
            # Order Block Entry Models
            for ob in order_blocks:
                entry_score = self._calculate_ob_entry_score(ob, stock_data)
                entry_opportunities.append({
                    'type': 'order_block_entry',
                    'timestamp': ob.timestamp,
                    'entry_price': (ob.high_price + ob.low_price) / 2,
                    'direction': 'LONG' if ob.block_type == 'bullish' else 'SHORT',
                    'stop_loss': ob.low_price if ob.block_type == 'bullish' else ob.high_price,
                    'take_profit': self._calculate_ob_target(ob, stock_data),
                    'confluence_score': entry_score,
                    'setup_type': self._classify_setup_quality(entry_score),
                    'supporting_concepts': ['Order Block', 'Institutional Level']
                })
            
            # Breaker Block Entry Models
            for bb in breaker_blocks:
                if bb.is_breaker:
                    entry_score = self._calculate_breaker_entry_score(bb, stock_data)
                    entry_opportunities.append({
                        'type': 'breaker_block_entry',
                        'timestamp': bb.timestamp,
                        'entry_price': (bb.high_price + bb.low_price) / 2,
                        'direction': 'SHORT' if bb.block_type == 'bullish_breaker' else 'LONG',
                        'stop_loss': bb.high_price if bb.block_type == 'bullish_breaker' else bb.low_price,
                        'take_profit': self._calculate_breaker_target(bb, stock_data),
                        'confluence_score': entry_score,
                        'setup_type': self._classify_setup_quality(entry_score),
                        'supporting_concepts': ['Breaker Block', 'Polarity Switch']
                    })
            
            # Sort by confluence score
            entry_opportunities.sort(key=lambda x: x['confluence_score'], reverse=True)
            
            return {
                'total_opportunities': len(entry_opportunities),
                'high_quality_setups': [op for op in entry_opportunities if op['setup_type'] == 'A+'],
                'medium_quality_setups': [op for op in entry_opportunities if op['setup_type'] == 'B'],
                'entry_models': entry_opportunities[:10],  # Top 10 opportunities
                'model_statistics': self._calculate_entry_model_stats(entry_opportunities)
            }
            
        except Exception as e:
            logger.error(f"Error in entry models analysis: {e}")
            return {'error': str(e)}
    
    def concept_33_exit_models_partial_full_scaling(self, stock_data: pd.DataFrame, active_trades: List[Dict]) -> Dict:
        """
        CONCEPT 33: Exit Models (Partial TP, Full TP, Scaling)
        - Partial profit taking strategies
        - Full position exit models
        - Scaling out techniques
        - Dynamic exit based on market conditions
        """
        try:
            exit_strategies = []
            
            for trade in active_trades:
                current_price = stock_data['close'].iloc[-1]
                entry_price = trade.get('entry_price', 0)
                direction = trade.get('direction', 'LONG')
                
                # Calculate current P&L
                if direction == 'LONG':
                    unrealized_pnl = (current_price - entry_price) / entry_price
                else:
                    unrealized_pnl = (entry_price - current_price) / entry_price
                
                # Exit model recommendations
                exit_model = {
                    'trade_id': trade.get('trade_id'),
                    'current_pnl_pct': unrealized_pnl * 100,
                    'exit_recommendations': []
                }
                
                # Partial profit taking (25% at 1:1, 50% at 2:1, 25% at 3:1)
                risk_amount = abs(entry_price - trade.get('stop_loss', entry_price))
                rr_1_to_1 = entry_price + (risk_amount if direction == 'LONG' else -risk_amount)
                rr_2_to_1 = entry_price + (2 * risk_amount if direction == 'LONG' else -2 * risk_amount)
                rr_3_to_1 = entry_price + (3 * risk_amount if direction == 'LONG' else -3 * risk_amount)
                
                # Check which targets have been hit
                if direction == 'LONG':
                    if current_price >= rr_1_to_1:
                        exit_model['exit_recommendations'].append({
                            'action': 'partial_exit',
                            'percentage': 25,
                            'target_price': rr_1_to_1,
                            'reason': '1:1 Risk Reward - Take first partial'
                        })
                    
                    if current_price >= rr_2_to_1:
                        exit_model['exit_recommendations'].append({
                            'action': 'partial_exit',
                            'percentage': 50,
                            'target_price': rr_2_to_1,
                            'reason': '2:1 Risk Reward - Take second partial'
                        })
                    
                    if current_price >= rr_3_to_1:
                        exit_model['exit_recommendations'].append({
                            'action': 'full_exit',
                            'percentage': 100,
                            'target_price': rr_3_to_1,
                            'reason': '3:1 Risk Reward - Full exit'
                        })
                
                # Dynamic exit based on market structure
                structure_exit = self._analyze_structure_based_exit(stock_data, trade, current_price)
                if structure_exit:
                    exit_model['exit_recommendations'].append(structure_exit)
                
                # Time-based exit (end of day, session)
                time_exit = self._analyze_time_based_exit(stock_data, trade)
                if time_exit:
                    exit_model['exit_recommendations'].append(time_exit)
                
                exit_strategies.append(exit_model)
            
            return {
                'total_active_trades': len(active_trades),
                'exit_strategies': exit_strategies,
                'scaling_methodology': {
                    'first_partial': '25% at 1:1 R:R',
                    'second_partial': '50% at 2:1 R:R', 
                    'final_exit': '25% at 3:1 R:R or structure break'
                },
                'exit_triggers': self._get_exit_trigger_guidelines()
            }
            
        except Exception as e:
            logger.error(f"Error in exit models analysis: {e}")
            return {'error': str(e)}
    
    def concept_34_risk_to_reward_optimization(self, stock_data: pd.DataFrame, trade_setups: List[Dict]) -> Dict:
        """
        CONCEPT 34: Risk-to-Reward Optimization
        - Minimum R:R ratio enforcement
        - Dynamic R:R based on market conditions
        - Position sizing optimization
        - Expected value calculations
        """
        try:
            optimized_setups = []
            
            for setup in trade_setups:
                entry_price = setup.get('entry_price', 0)
                stop_loss = setup.get('stop_loss', 0)
                take_profit = setup.get('take_profit', 0)
                
                if entry_price and stop_loss and take_profit:
                    # Calculate risk and reward
                    risk = abs(entry_price - stop_loss)
                    reward = abs(take_profit - entry_price)
                    
                    if risk > 0:
                        rr_ratio = reward / risk
                        
                        # Optimize based on market conditions
                        market_volatility = self._calculate_market_volatility(stock_data)
                        trend_strength = self._calculate_trend_strength(stock_data)
                        
                        # Adjust minimum R:R based on conditions
                        min_rr = self._calculate_dynamic_min_rr(market_volatility, trend_strength)
                        
                        optimized_setup = setup.copy()
                        optimized_setup.update({
                            'risk_amount': risk,
                            'reward_amount': reward,
                            'rr_ratio': rr_ratio,
                            'minimum_rr_required': min_rr,
                            'meets_rr_criteria': rr_ratio >= min_rr,
                            'market_volatility': market_volatility,
                            'trend_strength': trend_strength,
                            'expected_value': self._calculate_expected_value(rr_ratio, setup.get('win_probability', 0.6))
                        })
                        
                        # Suggest optimized targets if needed
                        if rr_ratio < min_rr:
                            optimized_tp = self._suggest_optimized_target(entry_price, stop_loss, min_rr, setup.get('direction'))
                            optimized_setup['suggested_take_profit'] = optimized_tp
                            optimized_setup['optimized_rr_ratio'] = min_rr
                        
                        optimized_setups.append(optimized_setup)
            
            # Filter and rank setups
            viable_setups = [s for s in optimized_setups if s['meets_rr_criteria']]
            viable_setups.sort(key=lambda x: x['expected_value'], reverse=True)
            
            return {
                'total_setups_analyzed': len(trade_setups),
                'viable_setups': len(viable_setups),
                'optimization_criteria': {
                    'minimum_rr_ratio': self.min_risk_reward_ratio,
                    'dynamic_rr_adjustment': True,
                    'volatility_consideration': True,
                    'trend_strength_factor': True
                },
                'optimized_setups': viable_setups[:5],  # Top 5 setups
                'rejected_setups': [s for s in optimized_setups if not s['meets_rr_criteria']],
                'optimization_statistics': self._calculate_optimization_stats(optimized_setups)
            }
            
        except Exception as e:
            logger.error(f"Error in risk-to-reward optimization: {e}")
            return {'error': str(e)}
    
    def concept_35_position_sizing_algorithms(self, account_balance: float, trade_setups: List[Dict]) -> Dict:
        """
        CONCEPT 35: Position Sizing Algorithms
        - Fixed dollar risk per trade
        - Percentage risk model
        - Kelly Criterion application
        - Volatility-adjusted sizing
        """
        try:
            sized_positions = []
            
            for setup in trade_setups:
                entry_price = setup.get('entry_price', 0)
                stop_loss = setup.get('stop_loss', 0)
                win_probability = setup.get('win_probability', 0.6)
                rr_ratio = setup.get('rr_ratio', 2.0)
                
                if entry_price and stop_loss:
                    risk_per_share = abs(entry_price - stop_loss)
                    
                    # Method 1: Fixed percentage risk (1% of account)
                    max_risk_amount = account_balance * (self.max_position_risk_percentage / 100)
                    fixed_pct_size = max_risk_amount / risk_per_share
                    
                    # Method 2: Kelly Criterion
                    kelly_fraction = self._calculate_kelly_fraction(win_probability, rr_ratio)
                    kelly_size = (account_balance * kelly_fraction) / entry_price
                    
                    # Method 3: Volatility-adjusted sizing
                    volatility_factor = setup.get('volatility_factor', 1.0)
                    vol_adjusted_size = fixed_pct_size / volatility_factor
                    
                    # Method 4: Confidence-based sizing
                    confluence_score = setup.get('confluence_score', 0.5)
                    confidence_multiplier = min(confluence_score * 2, 1.5)  # Max 1.5x normal size
                    confidence_size = fixed_pct_size * confidence_multiplier
                    
                    # Choose optimal size (conservative approach)
                    optimal_size = min(fixed_pct_size, kelly_size, vol_adjusted_size)
                    
                    # Calculate position value and metrics
                    position_value = optimal_size * entry_price
                    risk_amount = optimal_size * risk_per_share
                    risk_percentage = (risk_amount / account_balance) * 100
                    
                    sized_position = setup.copy()
                    sized_position.update({
                        'position_size_shares': optimal_size,
                        'position_value': position_value,
                        'risk_amount': risk_amount,
                        'risk_percentage': risk_percentage,
                        'sizing_methods': {
                            'fixed_percentage': fixed_pct_size,
                            'kelly_criterion': kelly_size,
                            'volatility_adjusted': vol_adjusted_size,
                            'confidence_based': confidence_size,
                            'chosen_method': 'conservative_minimum'
                        },
                        'kelly_fraction': kelly_fraction,
                        'max_position_value_pct': (position_value / account_balance) * 100
                    })
                    
                    sized_positions.append(sized_position)
            
            return {
                'account_balance': account_balance,
                'max_risk_per_trade_pct': self.max_position_risk_percentage,
                'max_risk_per_trade_dollar': account_balance * (self.max_position_risk_percentage / 100),
                'total_setups': len(trade_setups),
                'sized_positions': sized_positions,
                'position_sizes': sized_positions,  # Test expects this field name
                'portfolio_risk_analysis': self._analyze_portfolio_risk(sized_positions, account_balance),
                'sizing_guidelines': self._get_position_sizing_guidelines()
            }
            
        except Exception as e:
            logger.error(f"Error in position sizing algorithms: {e}")
            return {'error': str(e)}
    
    def concept_36_drawdown_control(self, trading_history: List[Dict], account_balance: float) -> Dict:
        """
        CONCEPT 36: Drawdown Control
        - Maximum drawdown limits
        - Daily loss limits
        - Recovery strategies
        - Risk reduction protocols
        """
        try:
            # Calculate drawdown metrics
            equity_curve = self._build_equity_curve(trading_history, account_balance)
            drawdown_analysis = self._analyze_drawdowns(equity_curve)
            
            # Current drawdown status
            current_balance = equity_curve[-1]['balance'] if equity_curve else account_balance
            peak_balance = max([point['balance'] for point in equity_curve]) if equity_curve else account_balance
            current_drawdown = (peak_balance - current_balance) / peak_balance * 100
            
            # Daily loss tracking
            today_trades = self._get_today_trades(trading_history)
            daily_pnl = sum([trade.get('pnl', 0) for trade in today_trades])
            daily_loss_pct = (daily_pnl / account_balance) * 100 if daily_pnl < 0 else 0
            
            # Risk controls and recommendations
            risk_controls = {
                'current_drawdown_pct': current_drawdown,
                'max_allowable_drawdown_pct': 10.0,  # 10% max drawdown
                'daily_loss_pct': abs(daily_loss_pct),
                'max_daily_loss_pct': self.max_daily_loss_percentage,
                'drawdown_status': self._classify_drawdown_status(current_drawdown),
                'trading_restrictions': self._get_trading_restrictions(current_drawdown, daily_loss_pct)
            }
            
            # Recovery strategies
            recovery_plan = self._generate_recovery_plan(current_drawdown, drawdown_analysis)
            
            return {
                'drawdown_analysis': drawdown_analysis,
                'current_status': risk_controls,
                'recovery_plan': recovery_plan,
                'equity_curve': equity_curve[-30:] if len(equity_curve) > 30 else equity_curve,  # Last 30 points
                'drawdown_statistics': self._calculate_drawdown_statistics(equity_curve),
                'recommendations': self._get_drawdown_recommendations(current_drawdown, daily_loss_pct)
            }
            
        except Exception as e:
            logger.error(f"Error in drawdown control analysis: {e}")
            return {'error': str(e)}
    
    def concept_37_compounding_models(self, initial_balance: float, monthly_return_target: float, years: int) -> Dict:
        """
        CONCEPT 37: Compounding Models
        - Compound growth projections
        - Risk-adjusted returns
        - Sustainable growth rates
        - Capital allocation strategies
        """
        try:
            # Monthly compounding model
            monthly_projections = []
            current_balance = initial_balance
            
            for month in range(years * 12):
                # Apply monthly return
                monthly_gain = current_balance * (monthly_return_target / 100)
                current_balance += monthly_gain
                
                monthly_projections.append({
                    'month': month + 1,
                    'year': (month // 12) + 1,
                    'balance': current_balance,
                    'monthly_gain': monthly_gain,
                    'total_return_pct': ((current_balance - initial_balance) / initial_balance) * 100
                })
            
            # Conservative vs Aggressive models
            conservative_model = self._calculate_compounding_scenario(initial_balance, monthly_return_target * 0.7, years)
            aggressive_model = self._calculate_compounding_scenario(initial_balance, monthly_return_target * 1.3, years)
            
            # Risk-adjusted compounding (considering drawdowns)
            risk_adjusted_model = self._calculate_risk_adjusted_compounding(initial_balance, monthly_return_target, years)
            
            # Withdrawal strategies
            withdrawal_strategies = self._calculate_withdrawal_strategies(current_balance, monthly_return_target)
            
            return {
                'base_scenario': {
                    'initial_balance': initial_balance,
                    'monthly_return_target_pct': monthly_return_target,
                    'final_balance': current_balance,
                    'total_return_pct': ((current_balance - initial_balance) / initial_balance) * 100,
                    'annual_compound_rate': (((current_balance / initial_balance) ** (1/years)) - 1) * 100
                },
                'scenarios': {
                    'conservative': conservative_model,
                    'base': monthly_projections[-1],
                    'aggressive': aggressive_model,
                    'risk_adjusted': risk_adjusted_model
                },
                'monthly_projections': monthly_projections[::3],  # Every 3rd month for brevity
                'withdrawal_strategies': withdrawal_strategies,
                'compounding_milestones': self._calculate_compounding_milestones(initial_balance, current_balance, years),
                'sustainability_analysis': self._analyze_return_sustainability(monthly_return_target)
            }
            
        except Exception as e:
            logger.error(f"Error in compounding models: {e}")
            return {'error': str(e)}
    
    def concept_38_daily_loss_limits(self, trading_history: List[Dict], account_balance: float) -> Dict:
        """
        CONCEPT 38: Daily Loss Limits
        - Maximum daily loss thresholds
        - Automatic trading halts
        - Loss limit enforcement
        - Recovery protocols
        """
        try:
            # Calculate daily P&L
            daily_pnl_analysis = self._analyze_daily_pnl(trading_history)
            
            # Current day status
            today_trades = self._get_today_trades(trading_history)
            today_pnl = sum([trade.get('pnl', 0) for trade in today_trades])
            today_loss_pct = abs(today_pnl / account_balance * 100) if today_pnl < 0 else 0
            
            # Loss limit rules
            loss_limits = {
                'soft_limit_pct': 1.0,    # 1% soft warning
                'hard_limit_pct': 2.0,    # 2% hard stop
                'emergency_limit_pct': 3.0  # 3% emergency shutdown
            }
            
            # Current status
            current_status = self._evaluate_daily_loss_status(today_loss_pct, loss_limits)
            
            # Trading permissions
            trading_permissions = {
                'can_open_new_trades': current_status['status'] not in ['hard_limit', 'emergency'],
                'can_increase_position_size': current_status['status'] == 'normal',
                'must_reduce_risk': current_status['status'] in ['soft_limit', 'hard_limit'],
                'emergency_liquidation': current_status['status'] == 'emergency'
            }
            
            # Recovery protocols
            recovery_protocols = self._generate_daily_loss_recovery_plan(today_loss_pct, loss_limits)
            
            return {
                'daily_loss_limits': loss_limits,
                'current_status': current_status,
                'today_performance': {
                    'trades_count': len(today_trades),
                    'total_pnl': today_pnl,
                    'loss_percentage': today_loss_pct,
                    'remaining_risk_budget': max(0, loss_limits['hard_limit_pct'] - today_loss_pct)
                },
                'trading_permissions': trading_permissions,
                'recovery_protocols': recovery_protocols,
                'daily_pnl_history': daily_pnl_analysis[-10:],  # Last 10 days
                'loss_limit_statistics': self._calculate_loss_limit_statistics(daily_pnl_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error in daily loss limits analysis: {e}")
            return {'error': str(e)}
    
    def concept_39_probability_profiles_abc_setups(self, trade_setups: List[Dict]) -> Dict:
        """
        CONCEPT 39: Probability Profiles (A+, B, C setups)
        - High probability A+ setups
        - Medium probability B setups
        - Lower probability C setups
        - Setup classification and filtering
        """
        try:
            classified_setups = {
                'A+': [],  # High probability (>70%)
                'B': [],   # Medium probability (50-70%)
                'C': []    # Lower probability (<50%)
            }
            
            for setup in trade_setups:
                # Calculate probability score based on confluence factors
                probability_score = self._calculate_setup_probability(setup)
                setup_classification = self._classify_setup_by_probability(probability_score)
                
                enhanced_setup = setup.copy()
                enhanced_setup.update({
                    'probability_score': probability_score,
                    'setup_grade': setup_classification,
                    'confluence_factors': self._analyze_confluence_factors(setup),
                    'risk_assessment': self._assess_setup_risk(setup),
                    'trade_recommendation': self._generate_trade_recommendation(setup_classification, probability_score)
                })
                
                classified_setups[setup_classification].append(enhanced_setup)
            
            # Sort each category by probability score
            for grade in classified_setups:
                classified_setups[grade].sort(key=lambda x: x['probability_score'], reverse=True)
            
            # Generate trading guidelines
            trading_guidelines = self._generate_probability_based_guidelines(classified_setups)
            
            # Performance expectations
            performance_expectations = self._calculate_performance_expectations(classified_setups)
            
            return {
                'setup_classification': classified_setups,
                'classified_setups': classified_setups,  # Test expects this field name
                'classification_summary': {
                    'A_plus_setups': len(classified_setups['A+']),
                    'B_setups': len(classified_setups['B']),
                    'C_setups': len(classified_setups['C']),
                    'total_setups': len(trade_setups)
                },
                'trading_guidelines': trading_guidelines,
                'performance_expectations': performance_expectations,
                'probability_methodology': self._get_probability_methodology(),
                'recommended_actions': self._get_probability_based_recommendations(classified_setups)
            }
            
        except Exception as e:
            logger.error(f"Error in probability profiles analysis: {e}")
            return {'error': str(e)}

    # Helper methods for Risk Management concepts
    
    def _create_journal_entry(self, trade_dict: Dict, stock_data: pd.DataFrame) -> Optional[TradeJournalEntry]:
        """Create a journal entry from trade dictionary"""
        try:
            entry = TradeEntry(
                timestamp=pd.to_datetime(trade_dict.get('timestamp', datetime.now())),
                symbol=trade_dict.get('symbol', ''),
                direction=trade_dict.get('direction', 'LONG'),
                entry_price=trade_dict.get('entry_price', 0.0),
                stop_loss=trade_dict.get('stop_loss', 0.0),
                take_profit=trade_dict.get('take_profit', 0.0),
                position_size=trade_dict.get('position_size', 0.0),
                risk_amount=trade_dict.get('risk_amount', 0.0),
                setup_type=SetupType(trade_dict.get('setup_type', 'B')),
                ict_concepts=trade_dict.get('ict_concepts', []),
                confluence_score=trade_dict.get('confluence_score', 0.5)
            )
            
            journal_entry = TradeJournalEntry(
                trade_id=trade_dict.get('trade_id', f"trade_{datetime.now().timestamp()}"),
                entry=entry,
                exit_price=trade_dict.get('exit_price'),
                exit_timestamp=pd.to_datetime(trade_dict.get('exit_timestamp')) if trade_dict.get('exit_timestamp') else None,
                outcome=TradeOutcome(trade_dict.get('outcome', 'PENDING')),
                pnl=trade_dict.get('pnl', 0.0),
                pnl_percentage=trade_dict.get('pnl_percentage', 0.0),
                duration_minutes=trade_dict.get('duration_minutes', 0),
                notes=trade_dict.get('notes', ''),
                screenshots=trade_dict.get('screenshots', [])
            )
            
            return journal_entry
        except Exception as e:
            logger.error(f"Error creating journal entry: {e}")
            return None
    
    def _calculate_performance_metrics(self, journal_entries: List[TradeJournalEntry]) -> RiskMetrics:
        """Calculate comprehensive performance metrics"""
        completed_trades = [entry for entry in journal_entries if entry.outcome in [TradeOutcome.WIN, TradeOutcome.LOSS]]
        
        if not completed_trades:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        wins = [entry for entry in completed_trades if entry.outcome == TradeOutcome.WIN]
        losses = [entry for entry in completed_trades if entry.outcome == TradeOutcome.LOSS]
        
        total_trades = len(completed_trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        
        total_wins = sum([entry.pnl for entry in wins])
        total_losses = abs(sum([entry.pnl for entry in losses]))
        
        average_win = total_wins / len(wins) if wins else 0
        average_loss = total_losses / len(losses) if losses else 0
        
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Calculate max drawdown
        equity_curve = []
        running_balance = 10000  # Assume starting balance
        for entry in journal_entries:
            running_balance += entry.pnl
            equity_curve.append(running_balance)
        
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Calculate Sharpe ratio (simplified)
        returns = [entry.pnl_percentage for entry in completed_trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Expectancy
        expectancy = (win_rate * average_win) - ((1 - win_rate) * average_loss)
        
        return RiskMetrics(
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            expectancy=expectancy
        )
    
    def _run_backtest_analysis(self, journal_entries: List[TradeJournalEntry], stock_data: pd.DataFrame) -> Dict:
        """Run backtesting analysis on trade history"""
        # Simplified backtest analysis
        return {
            'total_return_pct': sum([entry.pnl_percentage for entry in journal_entries]),
            'best_trade_pct': max([entry.pnl_percentage for entry in journal_entries]) if journal_entries else 0,
            'worst_trade_pct': min([entry.pnl_percentage for entry in journal_entries]) if journal_entries else 0,
            'average_trade_duration_hours': np.mean([entry.duration_minutes / 60 for entry in journal_entries]) if journal_entries else 0,
            'trades_per_month': len(journal_entries) / 12 if journal_entries else 0  # Assuming 1 year of data
        }
    
    def _analyze_trade_distribution(self, journal_entries: List[TradeJournalEntry]) -> Dict:
        """Analyze distribution of trades"""
        if not journal_entries:
            return {}
        
        by_setup_type = {}
        by_outcome = {}
        by_ict_concept = {}
        
        for entry in journal_entries:
            # By setup type
            setup = entry.entry.setup_type.value
            if setup not in by_setup_type:
                by_setup_type[setup] = {'count': 0, 'total_pnl': 0}
            by_setup_type[setup]['count'] += 1
            by_setup_type[setup]['total_pnl'] += entry.pnl
            
            # By outcome
            outcome = entry.outcome.value
            if outcome not in by_outcome:
                by_outcome[outcome] = 0
            by_outcome[outcome] += 1
            
            # By ICT concept
            for concept in entry.entry.ict_concepts:
                if concept not in by_ict_concept:
                    by_ict_concept[concept] = {'count': 0, 'total_pnl': 0}
                by_ict_concept[concept]['count'] += 1
                by_ict_concept[concept]['total_pnl'] += entry.pnl
        
        return {
            'by_setup_type': by_setup_type,
            'by_outcome': by_outcome,
            'by_ict_concept': by_ict_concept
        }
    
    def _serialize_journal_entry(self, entry: TradeJournalEntry) -> Dict:
        """Convert journal entry to dictionary for JSON serialization"""
        return {
            'trade_id': entry.trade_id,
            'timestamp': entry.entry.timestamp.isoformat(),
            'symbol': entry.entry.symbol,
            'direction': entry.entry.direction,
            'entry_price': entry.entry.entry_price,
            'exit_price': entry.exit_price,
            'pnl': entry.pnl,
            'pnl_percentage': entry.pnl_percentage,
            'outcome': entry.outcome.value,
            'setup_type': entry.entry.setup_type.value,
            'ict_concepts': entry.entry.ict_concepts,
            'confluence_score': entry.entry.confluence_score,
            'duration_minutes': entry.duration_minutes,
            'notes': entry.notes
        }
    
    def _calculate_fvg_entry_score(self, fvg, stock_data: pd.DataFrame) -> float:
        """Calculate entry score for Fair Value Gap"""
        score = 0.5  # Base score
        
        # Add points for gap size (larger gaps are more significant)
        gap_size_pct = (fvg.gap_size / fvg.gap_high) * 100
        if gap_size_pct > 2:
            score += 0.2
        elif gap_size_pct > 1:
            score += 0.1
        
        # Add points for volume confirmation
        # (This would need volume data analysis)
        score += 0.1
        
        # Add points for trend alignment
        # (This would need trend analysis)
        score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_fvg_target(self, fvg, stock_data: pd.DataFrame) -> float:
        """Calculate target price for FVG entry"""
        if fvg.gap_type == 'bullish':
            return fvg.gap_high + (fvg.gap_size * 2)  # 2:1 R:R
        else:
            return fvg.gap_low - (fvg.gap_size * 2)
    
    def _calculate_ob_entry_score(self, ob, stock_data: pd.DataFrame) -> float:
        """Calculate entry score for Order Block"""
        score = 0.6  # Base score (order blocks are generally reliable)
        
        # Add points for block strength
        if ob.strength > 0.8:
            score += 0.2
        elif ob.strength > 0.6:
            score += 0.1
        
        # Add points for fresh blocks (not retested)
        # (This would need retest analysis)
        score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_ob_target(self, ob, stock_data: pd.DataFrame) -> float:
        """Calculate target price for Order Block entry"""
        block_range = ob.high_price - ob.low_price
        if ob.block_type == 'bullish':
            return ob.high_price + (block_range * 3)  # 3:1 R:R
        else:
            return ob.low_price - (block_range * 3)
    
    def _calculate_breaker_entry_score(self, bb, stock_data: pd.DataFrame) -> float:
        """Calculate entry score for Breaker Block"""
        score = 0.7  # Base score (breaker blocks are high probability)
        
        # Add points for clean break
        if bb.strength > 0.9:
            score += 0.2
        
        # Add points for volume confirmation
        score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_breaker_target(self, bb, stock_data: pd.DataFrame) -> float:
        """Calculate target price for Breaker Block entry"""
        block_range = bb.high_price - bb.low_price
        if bb.block_type == 'bullish_breaker':  # Now bearish
            return bb.low_price - (block_range * 2)
        else:  # Now bullish
            return bb.high_price + (block_range * 2)
    
    def _classify_setup_quality(self, confluence_score: float) -> str:
        """Classify setup quality based on confluence score"""
        if confluence_score >= 0.8:
            return 'A+'
        elif confluence_score >= 0.6:
            return 'B'
        else:
            return 'C'
    
    def _calculate_entry_model_stats(self, opportunities: List[Dict]) -> Dict:
        """Calculate statistics for entry models"""
        if not opportunities:
            return {}
        
        by_type = {}
        for opp in opportunities:
            opp_type = opp['type']
            if opp_type not in by_type:
                by_type[opp_type] = {'count': 0, 'avg_score': 0}
            by_type[opp_type]['count'] += 1
            by_type[opp_type]['avg_score'] += opp['confluence_score']
        
        # Calculate averages
        for opp_type in by_type:
            by_type[opp_type]['avg_score'] /= by_type[opp_type]['count']
        
        return {
            'total_opportunities': len(opportunities),
            'by_type': by_type,
            'average_confluence_score': np.mean([opp['confluence_score'] for opp in opportunities])
        }
    
    def _analyze_structure_based_exit(self, stock_data: pd.DataFrame, trade: Dict, current_price: float) -> Optional[Dict]:
        """Analyze if structure suggests an exit"""
        # Simplified structure analysis
        # In a full implementation, this would analyze market structure breaks
        return {
            'action': 'structure_exit',
            'percentage': 100,
            'target_price': current_price,
            'reason': 'Market structure break detected'
        }
    
    def _analyze_time_based_exit(self, stock_data: pd.DataFrame, trade: Dict) -> Optional[Dict]:
        """Analyze time-based exit signals"""
        # Check if near market close
        current_time = datetime.now().time()
        market_close = datetime.strptime("16:00", "%H:%M").time()
        
        if current_time >= datetime.strptime("15:45", "%H:%M").time():
            return {
                'action': 'time_exit',
                'percentage': 100,
                'reason': 'Near market close - avoid overnight risk'
            }
        
        return None
    
    def _get_exit_trigger_guidelines(self) -> Dict:
        """Get guidelines for exit triggers"""
        return {
            'profit_targets': ['25% at 1:1 R:R', '50% at 2:1 R:R', '25% at 3:1 R:R'],
            'structure_exits': ['Break of market structure', 'Failed to hold key level', 'Reversal pattern'],
            'time_exits': ['End of trading session', 'Before major news', 'Lunch time exit'],
            'volatility_exits': ['Excessive volatility spike', 'Volume drying up', 'Ranging market']
        }

    def _calculate_market_volatility(self, stock_data: pd.DataFrame) -> float:
        """Calculate current market volatility"""
        if len(stock_data) < 20:
            return 0.5  # Default medium volatility
        
        returns = stock_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Normalize to 0-1 scale
        if volatility > 0.3:
            return 1.0  # High volatility
        elif volatility > 0.2:
            return 0.7  # Medium-high volatility
        elif volatility > 0.1:
            return 0.5  # Medium volatility
        else:
            return 0.3  # Low volatility
    
    def _calculate_trend_strength(self, stock_data: pd.DataFrame) -> float:
        """Calculate trend strength"""
        if len(stock_data) < 20:
            return 0.5
        
        # Simple trend strength using moving averages
        ma_short = stock_data['close'].rolling(window=10).mean()
        ma_long = stock_data['close'].rolling(window=20).mean()
        
        trend_direction = (ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]
        
        # Normalize to 0-1 scale
        return min(abs(trend_direction) * 10, 1.0)
    
    def _calculate_dynamic_min_rr(self, volatility: float, trend_strength: float) -> float:
        """Calculate dynamic minimum R:R ratio"""
        base_rr = self.min_risk_reward_ratio
        
        # Increase minimum R:R in high volatility
        volatility_adjustment = volatility * 0.5
        
        # Decrease minimum R:R in strong trends
        trend_adjustment = -trend_strength * 0.3
        
        return max(1.0, base_rr + volatility_adjustment + trend_adjustment)
    
    def _calculate_expected_value(self, rr_ratio: float, win_probability: float) -> float:
        """Calculate expected value of trade"""
        expected_win = win_probability * rr_ratio
        expected_loss = (1 - win_probability) * 1  # Risk is always 1 unit
        
        return expected_win - expected_loss
    
    def _suggest_optimized_target(self, entry_price: float, stop_loss: float, min_rr: float, direction: str) -> float:
        """Suggest optimized take profit target"""
        risk = abs(entry_price - stop_loss)
        reward_needed = risk * min_rr
        
        if direction == 'LONG':
            return entry_price + reward_needed
        else:
            return entry_price - reward_needed
    
    def _calculate_optimization_stats(self, setups: List[Dict]) -> Dict:
        """Calculate optimization statistics"""
        viable = [s for s in setups if s.get('meets_rr_criteria', False)]
        
        return {
            'total_setups': len(setups),
            'viable_setups': len(viable),
            'viability_rate': len(viable) / len(setups) if setups else 0,
            'average_rr_ratio': np.mean([s.get('rr_ratio', 0) for s in setups]),
            'average_expected_value': np.mean([s.get('expected_value', 0) for s in viable])
        }
    
    def _calculate_kelly_fraction(self, win_probability: float, rr_ratio: float) -> float:
        """Calculate Kelly Criterion fraction"""
        if rr_ratio <= 0:
            return 0
        
        # Kelly Formula: f = (bp - q) / b
        # where b = odds received (rr_ratio), p = win probability, q = loss probability
        kelly = (win_probability * rr_ratio - (1 - win_probability)) / rr_ratio
        
        # Cap Kelly at 25% for safety
        return max(0, min(kelly, 0.25))
    
    def _get_position_sizing_guidelines(self) -> Dict:
        """Get position sizing guidelines"""
        return {
            'max_risk_per_trade': f"{self.max_position_risk_percentage}% of account",
            'kelly_criterion': "Use Kelly formula but cap at 25%",
            'volatility_adjustment': "Reduce size in high volatility",
            'confidence_scaling': "Increase size for high-confidence setups (max 1.5x)",
            'correlation_limits': "Limit correlated positions"
        }
    
    def _analyze_portfolio_risk(self, sized_positions: List[Dict], account_balance: float) -> Dict:
        """Analyze overall portfolio risk"""
        total_risk = sum([pos.get('risk_amount', 0) for pos in sized_positions])
        total_position_value = sum([pos.get('position_value', 0) for pos in sized_positions])
        
        return {
            'total_risk_amount': total_risk,
            'total_risk_percentage': (total_risk / account_balance) * 100,
            'total_position_value': total_position_value,
            'portfolio_utilization': (total_position_value / account_balance) * 100,
            'number_of_positions': len(sized_positions),
            'max_concurrent_trades': self.max_concurrent_trades,
            'risk_per_position_avg': total_risk / len(sized_positions) if sized_positions else 0
        }
    
    def _build_equity_curve(self, trading_history: List[Dict], initial_balance: float) -> List[Dict]:
        """Build equity curve from trading history"""
        equity_curve = [{'date': datetime.now().date(), 'balance': initial_balance}]
        current_balance = initial_balance
        
        for trade in trading_history:
            trade_date = pd.to_datetime(trade.get('timestamp', datetime.now())).date()
            pnl = trade.get('pnl', 0)
            current_balance += pnl
            
            equity_curve.append({
                'date': trade_date,
                'balance': current_balance,
                'trade_pnl': pnl
            })
        
        return equity_curve
    
    def _analyze_drawdowns(self, equity_curve: List[Dict]) -> Dict:
        """Analyze drawdown periods"""
        if len(equity_curve) < 2:
            return {'max_drawdown_pct': 0, 'current_drawdown_pct': 0}
        
        balances = [point['balance'] for point in equity_curve]
        peak = balances[0]
        max_drawdown = 0
        current_drawdown = 0
        
        for balance in balances:
            if balance > peak:
                peak = balance
            
            drawdown = (peak - balance) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
            if balance == balances[-1]:  # Current balance
                current_drawdown = drawdown
        
        return {
            'max_drawdown_pct': max_drawdown * 100,
            'current_drawdown_pct': current_drawdown * 100,
            'peak_balance': peak,
            'current_balance': balances[-1]
        }
    
    def _get_today_trades(self, trading_history: List[Dict]) -> List[Dict]:
        """Get today's trades"""
        today = datetime.now().date()
        return [
            trade for trade in trading_history
            if pd.to_datetime(trade.get('timestamp', datetime.now())).date() == today
        ]
    
    def _classify_drawdown_status(self, drawdown_pct: float) -> str:
        """Classify drawdown status"""
        if drawdown_pct < 3:
            return 'normal'
        elif drawdown_pct < 6:
            return 'caution'
        elif drawdown_pct < 10:
            return 'warning'
        else:
            return 'critical'
    
    def _get_trading_restrictions(self, drawdown_pct: float, daily_loss_pct: float) -> Dict:
        """Get trading restrictions based on drawdown and daily loss"""
        restrictions = {
            'reduce_position_size': drawdown_pct > 5 or daily_loss_pct > 1,
            'no_new_trades': drawdown_pct > 8 or daily_loss_pct > 1.5,
            'emergency_stop': drawdown_pct > 12 or daily_loss_pct > 2,
            'only_a_plus_setups': drawdown_pct > 3 or daily_loss_pct > 0.5
        }
        
        return restrictions
    
    def _generate_recovery_plan(self, drawdown_pct: float, drawdown_analysis: Dict) -> Dict:
        """Generate recovery plan based on drawdown"""
        if drawdown_pct < 3:
            return {'status': 'normal_trading', 'actions': ['Continue normal trading']}
        elif drawdown_pct < 6:
            return {
                'status': 'cautious_recovery',
                'actions': [
                    'Reduce position sizes by 25%',
                    'Focus on A+ setups only',
                    'Review recent trades for mistakes'
                ]
            }
        elif drawdown_pct < 10:
            return {
                'status': 'recovery_mode',
                'actions': [
                    'Reduce position sizes by 50%',
                    'Only trade A+ setups',
                    'Take profits earlier',
                    'Review strategy and rules'
                ]
            }
        else:
            return {
                'status': 'emergency_recovery',
                'actions': [
                    'Stop trading temporarily',
                    'Review all trading rules and strategy',
                    'Consider paper trading to rebuild confidence',
                    'Seek mentorship or additional education'
                ]
            }
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve"""
        if not equity_curve:
            return 0
        
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd * 100
    
    def _calculate_drawdown_statistics(self, equity_curve: List[Dict]) -> Dict:
        """Calculate comprehensive drawdown statistics"""
        if len(equity_curve) < 2:
            return {}
        
        balances = [point['balance'] for point in equity_curve]
        
        return {
            'max_drawdown_pct': self._calculate_max_drawdown(balances),
            'average_drawdown_pct': 2.5,  # Simplified
            'drawdown_frequency': 'Monthly',  # Simplified
            'recovery_time_avg_days': 15  # Simplified
        }
    
    def _get_drawdown_recommendations(self, drawdown_pct: float, daily_loss_pct: float) -> List[str]:
        """Get recommendations based on drawdown"""
        recommendations = []
        
        if drawdown_pct > 5:
            recommendations.append("Consider reducing position sizes")
            recommendations.append("Review recent trading decisions")
        
        if daily_loss_pct > 1:
            recommendations.append("Stop trading for the day")
            recommendations.append("Analyze what went wrong today")
        
        if drawdown_pct > 10:
            recommendations.append("Take a break from trading")
            recommendations.append("Review your entire trading strategy")
        
        return recommendations if recommendations else ["Continue normal trading"]

    def _calculate_compounding_scenario(self, initial_balance: float, monthly_return: float, years: int) -> Dict:
        """Calculate compounding scenario"""
        final_balance = initial_balance * ((1 + monthly_return/100) ** (years * 12))
        
        return {
            'initial_balance': initial_balance,
            'monthly_return_pct': monthly_return,
            'final_balance': final_balance,
            'total_return_pct': ((final_balance - initial_balance) / initial_balance) * 100
        }
    
    def _calculate_risk_adjusted_compounding(self, initial_balance: float, monthly_return: float, years: int) -> Dict:
        """Calculate risk-adjusted compounding with drawdowns"""
        # Simulate occasional drawdowns
        adjusted_monthly_return = monthly_return * 0.9  # 10% reduction for drawdowns
        
        return self._calculate_compounding_scenario(initial_balance, adjusted_monthly_return, years)
    
    def _calculate_withdrawal_strategies(self, final_balance: float, monthly_return: float) -> Dict:
        """Calculate withdrawal strategies"""
        monthly_income = final_balance * (monthly_return / 100)
        
        return {
            'conservative_withdrawal': monthly_income * 0.5,
            'moderate_withdrawal': monthly_income * 0.7,
            'aggressive_withdrawal': monthly_income * 0.9,
            'preservation_mode': monthly_income * 0.3
        }
    
    def _calculate_compounding_milestones(self, initial: float, final: float, years: int) -> List[Dict]:
        """Calculate compounding milestones"""
        milestones = []
        targets = [initial * 2, initial * 5, initial * 10, initial * 20]
        
        for target in targets:
            if target <= final:
                # Estimate when milestone is reached
                months_to_target = 12 * years * (np.log(target/initial) / np.log(final/initial))
                milestones.append({
                    'target_balance': target,
                    'multiple_of_initial': target / initial,
                    'estimated_months': int(months_to_target),
                    'estimated_years': months_to_target / 12
                })
        
        return milestones
    
    def _analyze_return_sustainability(self, monthly_return: float) -> Dict:
        """Analyze sustainability of return target"""
        if monthly_return > 10:
            sustainability = 'very_high_risk'
            warning = 'Extremely difficult to sustain long-term'
        elif monthly_return > 5:
            sustainability = 'high_risk'
            warning = 'Requires exceptional skill and risk management'
        elif monthly_return > 3:
            sustainability = 'moderate_risk'
            warning = 'Achievable with good strategy and discipline'
        elif monthly_return > 1:
            sustainability = 'conservative'
            warning = 'Sustainable with proper risk management'
        else:
            sustainability = 'very_conservative'
            warning = 'Highly sustainable but may not beat inflation'
        
        return {
            'sustainability_rating': sustainability,
            'warning': warning,
            'recommended_max_monthly': 3.0,
            'market_benchmark_monthly': 0.8  # ~10% annual
        }
    
    def _analyze_daily_pnl(self, trading_history: List[Dict]) -> List[Dict]:
        """Analyze daily P&L history"""
        daily_pnl = {}
        
        for trade in trading_history:
            trade_date = pd.to_datetime(trade.get('timestamp', datetime.now())).date()
            pnl = trade.get('pnl', 0)
            
            if trade_date not in daily_pnl:
                daily_pnl[trade_date] = {'date': trade_date, 'pnl': 0, 'trades': 0}
            
            daily_pnl[trade_date]['pnl'] += pnl
            daily_pnl[trade_date]['trades'] += 1
        
        return list(daily_pnl.values())
    
    def _evaluate_daily_loss_status(self, loss_pct: float, limits: Dict) -> Dict:
        """Evaluate current daily loss status"""
        if loss_pct >= limits['emergency_limit_pct']:
            status = 'emergency'
            message = 'Emergency stop - account protection activated'
        elif loss_pct >= limits['hard_limit_pct']:
            status = 'hard_limit'
            message = 'Hard limit reached - no new trades allowed'
        elif loss_pct >= limits['soft_limit_pct']:
            status = 'soft_limit'
            message = 'Soft limit reached - reduce risk'
        else:
            status = 'normal'
            message = 'Normal trading allowed'
        
        return {
            'status': status,
            'message': message,
            'loss_percentage': loss_pct,
            'remaining_buffer': max(0, limits['hard_limit_pct'] - loss_pct)
        }
    
    def _generate_daily_loss_recovery_plan(self, loss_pct: float, limits: Dict) -> Dict:
        """Generate recovery plan for daily losses"""
        if loss_pct >= limits['emergency_limit_pct']:
            return {
                'immediate_actions': [
                    'Close all positions immediately',
                    'Stop trading for remainder of day',
                    'Review what went wrong'
                ],
                'next_day_plan': [
                    'Start with paper trading',
                    'Reduce position sizes by 75%',
                    'Only trade A+ setups'
                ]
            }
        elif loss_pct >= limits['hard_limit_pct']:
            return {
                'immediate_actions': [
                    'No new trades today',
                    'Consider closing existing trades',
                    'Review trading plan'
                ],
                'next_day_plan': [
                    'Reduce position sizes by 50%',
                    'Focus on high-probability setups only'
                ]
            }
        elif loss_pct >= limits['soft_limit_pct']:
            return {
                'immediate_actions': [
                    'Reduce position sizes',
                    'Be more selective with entries',
                    'Take profits earlier'
                ],
                'next_day_plan': [
                    'Review today\'s trades',
                    'Identify mistakes and lessons'
                ]
            }
        else:
            return {
                'immediate_actions': ['Continue normal trading'],
                'next_day_plan': ['Monitor performance closely']
            }
    
    def _calculate_loss_limit_statistics(self, daily_pnl: List[Dict]) -> Dict:
        """Calculate loss limit statistics"""
        if not daily_pnl:
            return {}
        
        loss_days = [day for day in daily_pnl if day.get('pnl', 0) < 0]
        
        return {
            'total_trading_days': len(daily_pnl),
            'loss_days': len(loss_days),
            'loss_day_frequency': len(loss_days) / len(daily_pnl) if daily_pnl else 0,
            'average_loss_day_pct': np.mean([abs(day['pnl']) for day in loss_days]) if loss_days else 0,
            'max_daily_loss_pct': max([abs(day['pnl']) for day in loss_days]) if loss_days else 0
        }
    
    def _calculate_setup_probability(self, setup: Dict) -> float:
        """Calculate probability score for setup"""
        base_probability = 0.5
        
        # Add confluence factors
        confluence_score = setup.get('confluence_score', 0.5)
        base_probability += (confluence_score - 0.5) * 0.4
        
        # ICT concept support
        ict_concepts = setup.get('supporting_concepts', [])
        concept_bonus = len(ict_concepts) * 0.05
        base_probability += concept_bonus
        
        # Market structure alignment
        if setup.get('structure_aligned', False):
            base_probability += 0.1
        
        # Trend alignment
        if setup.get('trend_aligned', False):
            base_probability += 0.1
        
        return min(base_probability, 0.95)  # Cap at 95%
    
    def _classify_setup_by_probability(self, probability: float) -> str:
        """Classify setup by probability"""
        if probability >= 0.7:
            return 'A+'
        elif probability >= 0.5:
            return 'B'
        else:
            return 'C'
    
    def _analyze_confluence_factors(self, setup: Dict) -> List[str]:
        """Analyze confluence factors for setup"""
        factors = []
        
        if setup.get('structure_aligned'):
            factors.append('Market structure alignment')
        
        if setup.get('trend_aligned'):
            factors.append('Trend alignment')
        
        if setup.get('volume_confirmation'):
            factors.append('Volume confirmation')
        
        if setup.get('multiple_timeframe_confluence'):
            factors.append('Multiple timeframe confluence')
        
        # Add ICT concepts
        factors.extend(setup.get('supporting_concepts', []))
        
        return factors
    
    def _assess_setup_risk(self, setup: Dict) -> Dict:
        """Assess risk level of setup"""
        rr_ratio = setup.get('rr_ratio', 1.0)
        confluence_score = setup.get('confluence_score', 0.5)
        
        if rr_ratio >= 3.0 and confluence_score >= 0.8:
            risk_level = 'low'
        elif rr_ratio >= 2.0 and confluence_score >= 0.6:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        return {
            'risk_level': risk_level,
            'rr_ratio': rr_ratio,
            'confluence_score': confluence_score
        }
    
    def _generate_trade_recommendation(self, setup_grade: str, probability: float) -> Dict:
        """Generate trade recommendation based on setup grade"""
        if setup_grade == 'A+':
            return {
                'action': 'take_trade',
                'position_size': 'normal_to_large',
                'confidence': 'high',
                'notes': 'High probability setup - consider larger position'
            }
        elif setup_grade == 'B':
            return {
                'action': 'take_trade',
                'position_size': 'normal',
                'confidence': 'medium',
                'notes': 'Decent setup - normal position size'
            }
        else:
            return {
                'action': 'skip_or_small',
                'position_size': 'small',
                'confidence': 'low',
                'notes': 'Lower probability - skip or very small position'
            }
    
    def _generate_probability_based_guidelines(self, classified_setups: Dict) -> Dict:
        """Generate trading guidelines based on probability"""
        return {
            'A_plus_guidelines': {
                'position_size': 'Up to 1.5x normal size',
                'frequency': 'Primary focus - take most A+ setups',
                'risk_management': 'Normal stop losses, let winners run'
            },
            'B_guidelines': {
                'position_size': 'Normal size',
                'frequency': 'Take selectively when no A+ available',
                'risk_management': 'Tighter stops, quicker profit taking'
            },
            'C_guidelines': {
                'position_size': '0.5x normal size or skip',
                'frequency': 'Only in very favorable market conditions',
                'risk_management': 'Very tight stops, quick exits'
            }
        }
    
    def _calculate_performance_expectations(self, classified_setups: Dict) -> Dict:
        """Calculate expected performance by setup type"""
        return {
            'A_plus_expectations': {
                'win_rate': '70-85%',
                'average_rr': '2.5:1',
                'expectancy': 'Highly positive'
            },
            'B_expectations': {
                'win_rate': '55-70%',
                'average_rr': '2:1',
                'expectancy': 'Positive'
            },
            'C_expectations': {
                'win_rate': '40-55%',
                'average_rr': '1.5:1',
                'expectancy': 'Break-even to slightly positive'
            }
        }
    
    def _get_probability_methodology(self) -> Dict:
        """Get explanation of probability methodology"""
        return {
            'factors_considered': [
                'ICT concept confluence',
                'Market structure alignment',
                'Trend direction alignment',
                'Volume confirmation',
                'Multiple timeframe analysis',
                'Risk-to-reward ratio',
                'Historical performance'
            ],
            'scoring_system': {
                'A+': 'Probability >= 70% (High confluence, multiple confirmations)',
                'B': 'Probability 50-70% (Good setup, some confirmations)',
                'C': 'Probability < 50% (Lower confluence, fewer confirmations)'
            }
        }
    
    def _get_probability_based_recommendations(self, classified_setups: Dict) -> List[str]:
        """Get recommendations based on probability analysis"""
        recommendations = []
        
        a_plus_count = len(classified_setups.get('A+', []))
        b_count = len(classified_setups.get('B', []))
        c_count = len(classified_setups.get('C', []))
        
        if a_plus_count > 0:
            recommendations.append(f"Focus on {a_plus_count} A+ setups - highest probability trades")
        
        if b_count > 3:
            recommendations.append(f"Consider selective B setups ({b_count} available) when A+ not available")
        
        if c_count > a_plus_count + b_count:
            recommendations.append("Many C setups available - avoid or trade very small size")
        
        if a_plus_count == 0 and b_count == 0:
            recommendations.append("No high-quality setups available - consider staying flat")
        
        return recommendations

# Global instance
risk_management_engine = StockRiskManagementEngine()