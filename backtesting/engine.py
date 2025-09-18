"""
Comprehensive Backtesting Framework for ICT Trading System
Walk-forward analysis, Monte Carlo simulation, and realistic cost modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import itertools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import time
import pickle
from pathlib import Path

from config.settings import settings
from monitoring.logging import logger, performance_monitor
from ict_engine.ict_complete import CompleteICTConceptsEngine
from trading_engine.engine import ICTSignalGenerator, TradeSignal
from data_pipeline.data_source import data_pipeline

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: str
    end_date: str
    initial_capital: float = 100000
    symbols: List[str] = None
    timeframes: List[str] = None
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005   # 0.05%
    market_impact_threshold: float = 0.01  # 1% of average volume
    
    # Walk-forward settings
    training_period_days: int = 252  # 1 year
    testing_period_days: int = 63    # 3 months
    rebalance_frequency: int = 21    # 3 weeks
    
    # Monte Carlo settings
    monte_carlo_runs: int = 1000
    confidence_levels: List[float] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        if self.timeframes is None:
            self.timeframes = ['5m', '15m', '1h', '1d']
        if self.confidence_levels is None:
            self.confidence_levels = [0.95, 0.99]

@dataclass
class Trade:
    """Individual trade record"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    commission: float
    slippage: float
    duration_minutes: int
    signal_confidence: float
    supporting_patterns: List[str]
    exit_reason: str
    
@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    config: BacktestConfig
    trades: List[Trade]
    equity_curve: pd.DataFrame
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    pattern_performance: Dict[str, Dict]
    monte_carlo_results: Optional[Dict] = None
    walk_forward_results: Optional[Dict] = None

class TradingCostModel:
    """Realistic trading cost modeling"""
    
    def __init__(self, config: BacktestConfig):
        self.commission_rate = config.commission_rate
        self.slippage_rate = config.slippage_rate
        self.market_impact_threshold = config.market_impact_threshold
    
    def calculate_costs(self, trade_data: Dict, market_data: pd.DataFrame, avg_volume: float) -> Dict[str, float]:
        """Calculate realistic trading costs"""
        
        # Base commission
        commission = trade_data['price'] * trade_data['quantity'] * self.commission_rate
        
        # Market impact based on trade size vs average volume
        volume_ratio = trade_data['quantity'] / avg_volume
        if volume_ratio > self.market_impact_threshold:
            market_impact = self.slippage_rate * (volume_ratio / self.market_impact_threshold)
        else:
            market_impact = 0.0
        
        # Bid-ask spread simulation
        volatility = market_data['high'].rolling(20).std().iloc[-1] / market_data['close'].iloc[-1]
        spread_cost = volatility * 0.5  # Half spread cost
        
        # Total slippage
        total_slippage = (self.slippage_rate + market_impact + spread_cost) * trade_data['price'] * trade_data['quantity']
        
        return {
            'commission': commission,
            'slippage': total_slippage,
            'market_impact': market_impact * trade_data['price'] * trade_data['quantity'],
            'total_cost': commission + total_slippage
        }

class PerformanceAnalyzer:
    """Advanced performance analysis"""
    
    @staticmethod
    def calculate_metrics(trades: List[Trade], equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        if not trades or equity_curve.empty:
            return {}
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = total_trades - winning_trades
        
        total_pnl = sum(t.pnl for t in trades)
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = sum(t.pnl for t in trades if t.pnl < 0)
        
        # Returns calculation
        returns = equity_curve['equity'].pct_change().dropna()
        cumulative_return = (equity_curve['equity'].iloc[-1] / equity_curve['equity'].iloc[0]) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        peak = equity_curve['equity'].expanding().max()
        drawdown = (equity_curve['equity'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = (cumulative_return * 100) / abs(max_drawdown * 100) if max_drawdown != 0 else 0
        
        # Advanced metrics
        sortino_ratio = PerformanceAnalyzer._calculate_sortino_ratio(returns)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': gross_profit / abs(gross_loss) if gross_loss != 0 else 0,
            'avg_win': gross_profit / winning_trades if winning_trades > 0 else 0,
            'avg_loss': gross_loss / losing_trades if losing_trades > 0 else 0,
            'cumulative_return': cumulative_return,
            'annual_return': cumulative_return,  # Simplified for now
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'total_commission': sum(t.commission for t in trades),
            'total_slippage': sum(t.slippage for t in trades),
            'avg_trade_duration_hours': np.mean([t.duration_minutes / 60 for t in trades]),
            'best_trade': max(trades, key=lambda x: x.pnl).pnl if trades else 0,
            'worst_trade': min(trades, key=lambda x: x.pnl).pnl if trades else 0,
        }
    
    @staticmethod
    def _calculate_sortino_ratio(returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0
        
        downside_std = downside_returns.std() * np.sqrt(252)
        return (returns.mean() * 252) / downside_std if downside_std > 0 else 0
    
    @staticmethod
    def analyze_pattern_performance(trades: List[Trade]) -> Dict[str, Dict]:
        """Analyze performance by pattern type"""
        pattern_stats = {}
        
        for trade in trades:
            for pattern in trade.supporting_patterns:
                if pattern not in pattern_stats:
                    pattern_stats[pattern] = {
                        'trades': [],
                        'total_pnl': 0,
                        'win_count': 0,
                        'total_count': 0
                    }
                
                pattern_stats[pattern]['trades'].append(trade)
                pattern_stats[pattern]['total_pnl'] += trade.pnl
                pattern_stats[pattern]['total_count'] += 1
                if trade.pnl > 0:
                    pattern_stats[pattern]['win_count'] += 1
        
        # Calculate metrics for each pattern
        for pattern, stats in pattern_stats.items():
            stats['win_rate'] = stats['win_count'] / stats['total_count']
            stats['avg_pnl'] = stats['total_pnl'] / stats['total_count']
            stats['profit_factor'] = sum(t.pnl for t in stats['trades'] if t.pnl > 0) / abs(sum(t.pnl for t in stats['trades'] if t.pnl < 0)) if any(t.pnl < 0 for t in stats['trades']) else float('inf')
        
        return pattern_stats

class MonteCarloSimulator:
    """Monte Carlo simulation for risk assessment"""
    
    def __init__(self, trades: List[Trade], config: BacktestConfig):
        self.trades = trades
        self.config = config
        
    def run_simulation(self) -> Dict[str, Any]:
        """Run Monte Carlo simulation"""
        
        if not self.trades:
            return {}
        
        # Extract trade returns
        trade_returns = [t.pnl_percent for t in self.trades]
        
        # Run simulations
        simulation_results = []
        
        for _ in range(self.config.monte_carlo_runs):
            # Randomly sample trades with replacement
            sampled_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            
            # Calculate portfolio equity curve
            equity = self.config.initial_capital
            equity_curve = [equity]
            
            for ret in sampled_returns:
                equity *= (1 + ret)
                equity_curve.append(equity)
            
            # Calculate final return and max drawdown
            final_return = (equity - self.config.initial_capital) / self.config.initial_capital
            
            # Calculate drawdown
            peak = equity_curve[0]
            max_dd = 0
            for value in equity_curve:
                if value > peak:
                    peak = value
                dd = (value - peak) / peak
                if dd < max_dd:
                    max_dd = dd
            
            simulation_results.append({
                'final_return': final_return,
                'max_drawdown': max_dd,
                'final_equity': equity
            })
        
        # Analyze results
        final_returns = [r['final_return'] for r in simulation_results]
        max_drawdowns = [r['max_drawdown'] for r in simulation_results]
        final_equities = [r['final_equity'] for r in simulation_results]
        
        results = {
            'simulations_run': self.config.monte_carlo_runs,
            'return_percentiles': {
                '5%': np.percentile(final_returns, 5),
                '25%': np.percentile(final_returns, 25),
                '50%': np.percentile(final_returns, 50),
                '75%': np.percentile(final_returns, 75),
                '95%': np.percentile(final_returns, 95)
            },
            'drawdown_percentiles': {
                '5%': np.percentile(max_drawdowns, 5),
                '25%': np.percentile(max_drawdowns, 25),
                '50%': np.percentile(max_drawdowns, 50),
                '75%': np.percentile(max_drawdowns, 75),
                '95%': np.percentile(max_drawdowns, 95)
            },
            'probability_positive_return': len([r for r in final_returns if r > 0]) / len(final_returns),
            'expected_return': np.mean(final_returns),
            'return_volatility': np.std(final_returns),
            'expected_final_equity': np.mean(final_equities)
        }
        
        return results

class WalkForwardAnalyzer:
    """Walk-forward analysis implementation"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
    async def run_walk_forward(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run walk-forward analysis"""
        
        logger.info("Starting walk-forward analysis")
        
        # Define periods
        start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')
        
        training_period = timedelta(days=self.config.training_period_days)
        testing_period = timedelta(days=self.config.testing_period_days)
        
        results = []
        current_date = start_date
        
        while current_date + training_period + testing_period <= end_date:
            
            train_start = current_date
            train_end = current_date + training_period
            test_start = train_end
            test_end = test_start + testing_period
            
            logger.info(f"Walk-forward period: {train_start.date()} to {test_end.date()}")
            
            # Run backtest for this period
            period_result = await self._run_period_backtest(
                data, train_start, train_end, test_start, test_end
            )
            
            if period_result:
                results.append(period_result)
            
            # Move to next period
            current_date += timedelta(days=self.config.rebalance_frequency)
        
        # Aggregate results
        return self._aggregate_walk_forward_results(results)
    
    async def _run_period_backtest(self, data: Dict[str, pd.DataFrame], 
                                 train_start: datetime, train_end: datetime,
                                 test_start: datetime, test_end: datetime) -> Optional[Dict]:
        """Run backtest for a specific period"""
        
        try:
            # Extract testing period data
            test_data = {}
            for symbol, df in data.items():
                period_data = df[(df.index >= test_start) & (df.index <= test_end)]
                if len(period_data) > 0:
                    test_data[symbol] = period_data
            
            if not test_data:
                return None
            
            # Run backtest on test period
            backtest_engine = BacktestEngine(self.config)
            result = await backtest_engine._run_single_backtest(test_data, is_walk_forward=True)
            
            return {
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'performance_metrics': result.performance_metrics,
                'trades_count': len(result.trades),
                'total_return': result.performance_metrics.get('cumulative_return', 0)
            }
            
        except Exception as e:
            logger.error(f"Error in period backtest: {e}")
            return None
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate walk-forward results"""
        
        if not results:
            return {}
        
        # Aggregate metrics
        total_returns = [r['total_return'] for r in results]
        
        return {
            'periods_analyzed': len(results),
            'average_return': np.mean(total_returns),
            'return_std': np.std(total_returns),
            'best_period_return': max(total_returns),
            'worst_period_return': min(total_returns),
            'positive_periods': len([r for r in total_returns if r > 0]),
            'consistency_ratio': len([r for r in total_returns if r > 0]) / len(total_returns),
            'period_results': results
        }

class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.signal_generator = ICTSignalGenerator()
        self.cost_model = TradingCostModel(config)
        
    async def run_comprehensive_backtest(self) -> BacktestResult:
        """Run comprehensive backtest with all features"""
        
        logger.info("Starting comprehensive backtest")
        start_time = time.time()
        
        # Load data
        data = await self._load_historical_data()
        if not data:
            raise ValueError("No data available for backtesting")
        
        # Run main backtest
        result = await self._run_single_backtest(data)
        
        # Run Monte Carlo simulation
        if self.config.monte_carlo_runs > 0:
            logger.info("Running Monte Carlo simulation")
            mc_simulator = MonteCarloSimulator(result.trades, self.config)
            result.monte_carlo_results = mc_simulator.run_simulation()
        
        # Run walk-forward analysis
        logger.info("Running walk-forward analysis")
        wf_analyzer = WalkForwardAnalyzer(self.config)
        result.walk_forward_results = await wf_analyzer.run_walk_forward(data)
        
        execution_time = time.time() - start_time
        logger.info(f"Backtest completed in {execution_time:.2f} seconds")
        performance_monitor.record_execution_time("comprehensive_backtest", execution_time * 1000)
        
        return result
    
    async def _load_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Load historical data for all symbols"""
        
        data = {}
        
        for symbol in self.config.symbols:
            try:
                logger.info(f"Loading data for {symbol}")
                
                historical_data = await data_pipeline.get_historical_data(
                    symbol, self.config.start_date, self.config.end_date, '1d'
                )
                
                if historical_data:
                    df_data = []
                    for data_point in historical_data:
                        df_data.append({
                            'timestamp': data_point.timestamp,
                            'open': data_point.open,
                            'high': data_point.high,
                            'low': data_point.low,
                            'close': data_point.close,
                            'volume': data_point.volume
                        })
                    
                    df = pd.DataFrame(df_data)
                    df.set_index('timestamp', inplace=True)
                    df = df.sort_index()
                    
                    if len(df) > 50:  # Minimum data requirement
                        data[symbol] = df
                        logger.info(f"Loaded {len(df)} data points for {symbol}")
                
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
        
        return data
    
    async def _run_single_backtest(self, data: Dict[str, pd.DataFrame], is_walk_forward: bool = False) -> BacktestResult:
        """Run single backtest"""
        
        all_trades = []
        equity_values = []
        current_equity = self.config.initial_capital
        equity_values.append({'timestamp': datetime.strptime(self.config.start_date, '%Y-%m-%d'), 'equity': current_equity})
        
        # Track positions
        positions = {}
        
        # Generate signals and simulate trading
        for symbol, df in data.items():
            try:
                logger.info(f"Processing {symbol} with {len(df)} data points")
                
                # Set up ICT engine
                self.signal_generator.ict_engine.symbol = symbol
                
                # Analyze data in chunks to simulate real-time
                chunk_size = 50  # Analyze 50 bars at a time
                
                for i in range(chunk_size, len(df), 5):  # Move 5 bars at a time
                    chunk_data = df.iloc[:i]
                    
                    if len(chunk_data) < chunk_size:
                        continue
                    
                    # Run ICT analysis
                    analysis = self.signal_generator.ict_engine.analyze_market_structure(chunk_data)
                    
                    if 'error' in analysis:
                        continue
                    
                    # Generate signals
                    signals = await self.signal_generator._analyze_for_signals(
                        symbol, chunk_data, analysis, '1d'
                    )
                    
                    # Process signals
                    for signal in signals:
                        if signal.confidence > 0.6:  # Only high-confidence signals
                            
                            # Check if we can enter position
                            if symbol not in positions:
                                
                                # Calculate trade details
                                entry_time = chunk_data.index[i]
                                entry_price = signal.entry_price
                                
                                # Calculate position size based on available capital
                                max_position_value = current_equity * 0.1  # 10% max per position
                                position_size = min(signal.position_size, max_position_value / entry_price)
                                
                                if position_size > 0:
                                    # Calculate costs
                                    avg_volume = chunk_data['volume'].tail(20).mean()
                                    costs = self.cost_model.calculate_costs(
                                        {'price': entry_price, 'quantity': position_size},
                                        chunk_data,
                                        avg_volume
                                    )
                                    
                                    # Enter position
                                    positions[symbol] = {
                                        'signal': signal,
                                        'entry_time': entry_time,
                                        'entry_price': entry_price,
                                        'quantity': position_size,
                                        'costs': costs
                                    }
                                    
                                    current_equity -= costs['total_cost']
                    
                    # Check for exits
                    current_price = chunk_data['close'].iloc[-1]
                    current_time = chunk_data.index[-1]
                    
                    symbols_to_exit = []
                    for pos_symbol, position in positions.items():
                        if pos_symbol == symbol:
                            # Check exit conditions
                            pnl_percent = (current_price - position['entry_price']) / position['entry_price']
                            
                            should_exit = False
                            exit_reason = ""
                            
                            if position['signal'].signal_type == 'buy':
                                if current_price >= position['signal'].take_profit:
                                    should_exit = True
                                    exit_reason = "take_profit"
                                elif current_price <= position['signal'].stop_loss:
                                    should_exit = True
                                    exit_reason = "stop_loss"
                            else:  # sell signal
                                if current_price <= position['signal'].take_profit:
                                    should_exit = True
                                    exit_reason = "take_profit"
                                elif current_price >= position['signal'].stop_loss:
                                    should_exit = True
                                    exit_reason = "stop_loss"
                            
                            # Time-based exit (max 30 days)
                            if (current_time - position['entry_time']).days > 30:
                                should_exit = True
                                exit_reason = "time_exit"
                            
                            if should_exit:
                                # Calculate exit costs
                                exit_costs = self.cost_model.calculate_costs(
                                    {'price': current_price, 'quantity': position['quantity']},
                                    chunk_data,
                                    chunk_data['volume'].tail(20).mean()
                                )
                                
                                # Calculate PnL
                                if position['signal'].signal_type == 'buy':
                                    pnl = (current_price - position['entry_price']) * position['quantity']
                                else:
                                    pnl = (position['entry_price'] - current_price) * position['quantity']
                                
                                pnl -= (position['costs']['total_cost'] + exit_costs['total_cost'])
                                
                                # Create trade record
                                trade = Trade(
                                    symbol=symbol,
                                    entry_time=position['entry_time'],
                                    exit_time=current_time,
                                    side='long' if position['signal'].signal_type == 'buy' else 'short',
                                    entry_price=position['entry_price'],
                                    exit_price=current_price,
                                    quantity=position['quantity'],
                                    pnl=pnl,
                                    pnl_percent=pnl / (position['entry_price'] * position['quantity']),
                                    commission=position['costs']['commission'] + exit_costs['commission'],
                                    slippage=position['costs']['slippage'] + exit_costs['slippage'],
                                    duration_minutes=int((current_time - position['entry_time']).total_seconds() / 60),
                                    signal_confidence=position['signal'].confidence,
                                    supporting_patterns=position['signal'].supporting_patterns,
                                    exit_reason=exit_reason
                                )
                                
                                all_trades.append(trade)
                                current_equity += pnl + (position['entry_price'] * position['quantity'])
                                symbols_to_exit.append(pos_symbol)
                                
                                logger.debug(f"Closed position in {symbol}, PnL: ${pnl:.2f}")
                    
                    # Remove exited positions
                    for symbol_exit in symbols_to_exit:
                        del positions[symbol_exit]
                    
                    # Record equity
                    if not is_walk_forward:  # Don't record every point in walk-forward
                        equity_values.append({
                            'timestamp': current_time,
                            'equity': current_equity + sum(pos['entry_price'] * pos['quantity'] for pos in positions.values())
                        })
            
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(equity_values)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate performance metrics
        performance_metrics = PerformanceAnalyzer.calculate_metrics(all_trades, equity_df)
        
        # Analyze pattern performance
        pattern_performance = PerformanceAnalyzer.analyze_pattern_performance(all_trades)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(all_trades, equity_df)
        
        logger.info(f"Backtest completed: {len(all_trades)} trades, "
                   f"Total PnL: ${performance_metrics.get('total_pnl', 0):.2f}, "
                   f"Win Rate: {performance_metrics.get('win_rate', 0):.2%}")
        
        return BacktestResult(
            config=self.config,
            trades=all_trades,
            equity_curve=equity_df,
            performance_metrics=performance_metrics,
            risk_metrics=risk_metrics,
            pattern_performance=pattern_performance
        )
    
    def _calculate_risk_metrics(self, trades: List[Trade], equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate additional risk metrics"""
        
        if not trades:
            return {}
        
        # Consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        for trade in trades:
            if trade.pnl < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        # Largest loss streak value
        loss_streak_value = 0
        current_streak_value = 0
        
        for trade in trades:
            if trade.pnl < 0:
                current_streak_value += trade.pnl
                loss_streak_value = min(loss_streak_value, current_streak_value)
            else:
                current_streak_value = 0
        
        return {
            'max_consecutive_losses': max_consecutive_losses,
            'largest_loss_streak_value': loss_streak_value,
            'average_trade_size': np.mean([t.quantity * t.entry_price for t in trades]),
            'risk_per_trade_avg': np.mean([abs(t.pnl) for t in trades if t.pnl < 0]) if any(t.pnl < 0 for t in trades) else 0,
            'reward_per_trade_avg': np.mean([t.pnl for t in trades if t.pnl > 0]) if any(t.pnl > 0 for t in trades) else 0,
        }
    
    def save_results(self, result: BacktestResult, filepath: str):
        """Save backtest results to file"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(result, f)
            
            logger.info(f"Backtest results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

# Factory function for easy backtesting
async def run_backtest(symbols: List[str], start_date: str, end_date: str, 
                      initial_capital: float = 100000, **kwargs) -> BacktestResult:
    """Convenience function to run a backtest"""
    
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        symbols=symbols,
        **kwargs
    )
    
    engine = BacktestEngine(config)
    return await engine.run_comprehensive_backtest()

__all__ = [
    'BacktestEngine', 'BacktestConfig', 'BacktestResult', 'Trade', 
    'MonteCarloSimulator', 'WalkForwardAnalyzer', 'PerformanceAnalyzer',
    'run_backtest'
]