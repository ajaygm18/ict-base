"""
Real-Time Trading Engine for ICT System
Handles order execution, position management, and real-time data processing
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np

from config.settings import settings
from monitoring.logging import logger, trading_logger, performance_monitor
from data_pipeline.data_source import data_pipeline
from ict_engine.ict_complete import CompleteICTConceptsEngine

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    """Trading order representation"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    timestamp: datetime = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Position:
    """Current position representation"""
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    entry_orders: List[str]
    
@dataclass
class TradeSignal:
    """Trading signal from ICT analysis"""
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    supporting_patterns: List[str]
    risk_reward_ratio: float
    position_size: float
    timestamp: datetime
    timeframe: str
    
class PaperTradingBroker:
    """Paper trading implementation for testing"""
    
    def __init__(self, initial_balance: float = 100000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trade_history: List[Dict] = []
        self.order_counter = 0
        
    async def place_order(self, order: Order) -> str:
        """Place an order in paper trading"""
        try:
            order.id = f"ORDER_{self.order_counter:06d}"
            self.order_counter += 1
            
            # Simulate order processing
            await asyncio.sleep(0.1)  # Simulate network latency
            
            # Get current market price
            market_data = await data_pipeline.get_real_time_data([order.symbol])
            if not market_data.get(order.symbol):
                order.status = OrderStatus.REJECTED
                self.orders[order.id] = order
                return order.id
            
            current_price = market_data[order.symbol][-1].close
            
            # Execute order based on type
            if order.order_type == OrderType.MARKET:
                fill_price = current_price
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.avg_fill_price = fill_price
                
                # Update position
                await self._update_position(order, fill_price)
                
            elif order.order_type == OrderType.LIMIT:
                # Check if limit order can be filled
                if ((order.side == OrderSide.BUY and current_price <= order.price) or
                    (order.side == OrderSide.SELL and current_price >= order.price)):
                    
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.avg_fill_price = order.price
                    
                    await self._update_position(order, order.price)
                else:
                    # Order remains pending
                    order.status = OrderStatus.PENDING
            
            self.orders[order.id] = order
            trading_logger.log_trade_entry(asdict(order))
            
            return order.id
            
        except Exception as e:
            logger.error(f"Error placing order: {e}", order_data=asdict(order))
            order.status = OrderStatus.REJECTED
            return order.id
    
    async def _update_position(self, order: Order, fill_price: float):
        """Update position based on filled order"""
        try:
            if order.symbol not in self.positions:
                # New position
                if order.side == OrderSide.BUY:
                    quantity = order.filled_quantity
                else:
                    quantity = -order.filled_quantity
                
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=quantity,
                    avg_price=fill_price,
                    market_value=quantity * fill_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    timestamp=datetime.now(),
                    entry_orders=[order.id]
                )
            else:
                # Update existing position
                position = self.positions[order.symbol]
                
                if order.side == OrderSide.BUY:
                    new_quantity = position.quantity + order.filled_quantity
                else:
                    new_quantity = position.quantity - order.filled_quantity
                
                if new_quantity == 0:
                    # Position closed
                    realized_pnl = (fill_price - position.avg_price) * abs(position.quantity)
                    if position.quantity < 0:  # Short position
                        realized_pnl = -realized_pnl
                    
                    position.realized_pnl += realized_pnl
                    self.balance += realized_pnl
                    
                    # Remove position
                    del self.positions[order.symbol]
                    
                else:
                    # Update position
                    if (position.quantity > 0 and order.side == OrderSide.BUY) or \
                       (position.quantity < 0 and order.side == OrderSide.SELL):
                        # Adding to position
                        total_value = (position.quantity * position.avg_price + 
                                     order.filled_quantity * fill_price)
                        position.quantity = new_quantity
                        position.avg_price = total_value / abs(new_quantity)
                    else:
                        # Reducing position
                        position.quantity = new_quantity
                        if new_quantity != 0:
                            # Partial close
                            realized_pnl = (fill_price - position.avg_price) * order.filled_quantity
                            if position.quantity < 0:
                                realized_pnl = -realized_pnl
                            position.realized_pnl += realized_pnl
                            self.balance += realized_pnl
                
                position.entry_orders.append(order.id)
            
            # Update balance for commissions
            commission = fill_price * order.filled_quantity * 0.001  # 0.1% commission
            self.balance -= commission
            order.commission = commission
            
        except Exception as e:
            logger.error(f"Error updating position: {e}", symbol=order.symbol)
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        # Update unrealized PnL
        if self.positions:
            symbols = list(self.positions.keys())
            market_data = await data_pipeline.get_real_time_data(symbols)
            
            for symbol, position in self.positions.items():
                if symbol in market_data and market_data[symbol]:
                    current_price = market_data[symbol][-1].close
                    position.market_value = position.quantity * current_price
                    position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
        
        return self.positions
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        positions = await self.get_positions()
        total_market_value = sum(pos.market_value for pos in positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in positions.values())
        
        return {
            'balance': self.balance,
            'total_market_value': total_market_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'equity': self.balance + total_unrealized_pnl,
            'buying_power': self.balance * 2,  # 2:1 leverage for stocks
            'positions_count': len(positions),
            'orders_count': len([o for o in self.orders.values() if o.status == OrderStatus.PENDING])
        }

class ICTSignalGenerator:
    """Generate trading signals from ICT analysis"""
    
    def __init__(self):
        self.ict_engine = CompleteICTConceptsEngine()
        # Avoid circular import by importing here
        try:
            from risk_engine.position_manager import PositionManager
            self.position_manager = PositionManager(100000)  # Default initial capital
        except ImportError:
            self.position_manager = None
        except Exception:
            self.position_manager = None
        
    async def generate_signals(self, symbols: List[str], timeframe: str = "5m") -> List[TradeSignal]:
        """Generate trading signals for symbols"""
        signals = []
        
        try:
            for symbol in symbols:
                # Get recent data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                historical_data = await data_pipeline.get_historical_data(
                    symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), timeframe
                )
                
                if len(historical_data) < 50:
                    continue
                
                # Convert to DataFrame
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
                
                # Run ICT analysis
                self.ict_engine.symbol = symbol
                analysis = self.ict_engine.analyze_market_structure(df)
                
                if 'error' in analysis:
                    continue
                
                # Generate signals from analysis
                symbol_signals = await self._analyze_for_signals(symbol, df, analysis, timeframe)
                signals.extend(symbol_signals)
                
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return signals
    
    async def _analyze_for_signals(self, symbol: str, df: pd.DataFrame, analysis: Dict, timeframe: str) -> List[TradeSignal]:
        """Analyze ICT data for trading signals"""
        signals = []
        current_price = df['close'].iloc[-1]
        
        try:
            # Check for Order Block signals
            order_blocks = analysis['concepts']['concept_4']
            if 'active_bullish' in order_blocks:
                for ob in order_blocks['active_bullish']:
                    if self._is_price_near_level(current_price, ob['mitigation_level'], 0.01):
                        # Bullish signal
                        stop_loss = ob['low'] * 0.995  # 0.5% below OB low
                        take_profit = current_price + (current_price - stop_loss) * 2  # 2:1 RR
                        
                        signal = TradeSignal(
                            symbol=symbol,
                            signal_type='buy',
                            confidence=min(ob['strength'] / 5, 1.0),
                            entry_price=current_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            supporting_patterns=['bullish_order_block'],
                            risk_reward_ratio=2.0,
                            position_size=self._calculate_position_size(current_price, stop_loss),
                            timestamp=datetime.now(),
                            timeframe=timeframe
                        )
                        signals.append(signal)
            
            # Check for Fair Value Gap signals
            fvgs = analysis['concepts']['concept_6']
            if 'active_bullish_fvgs' in fvgs:
                for fvg in fvgs['active_bullish_fvgs']:
                    if self._is_price_near_level(current_price, fvg['mitigation_level'], 0.015):
                        stop_loss = fvg['gap_low'] * 0.99
                        take_profit = current_price + (current_price - stop_loss) * 1.5
                        
                        signal = TradeSignal(
                            symbol=symbol,
                            signal_type='buy',
                            confidence=min(fvg['strength'] / 5, 1.0),
                            entry_price=current_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            supporting_patterns=['bullish_fvg'],
                            risk_reward_ratio=1.5,
                            position_size=self._calculate_position_size(current_price, stop_loss),
                            timestamp=datetime.now(),
                            timeframe=timeframe
                        )
                        signals.append(signal)
            
            # Check Premium/Discount levels
            ote = analysis['concepts']['concept_10']
            if ote.get('optimal_entries'):
                for entry in ote['optimal_entries']:
                    if abs(current_price - entry['entry_level']) / current_price < 0.02:
                        signal = TradeSignal(
                            symbol=symbol,
                            signal_type=entry['type'].split('_')[0],  # 'bullish' or 'bearish'
                            confidence=0.8,
                            entry_price=entry['entry_level'],
                            stop_loss=entry['stop_loss'],
                            take_profit=entry['target'],
                            supporting_patterns=['optimal_trade_entry'],
                            risk_reward_ratio=entry['risk_reward'],
                            position_size=self._calculate_position_size(entry['entry_level'], entry['stop_loss']),
                            timestamp=datetime.now(),
                            timeframe=timeframe
                        )
                        signals.append(signal)
            
        except Exception as e:
            logger.error(f"Error analyzing signals for {symbol}: {e}")
        
        return signals
    
    def _is_price_near_level(self, current_price: float, level: float, threshold: float) -> bool:
        """Check if current price is near a significant level"""
        return abs(current_price - level) / current_price <= threshold
    
    def _calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        risk_per_trade = settings.risk_management.max_position_risk
        risk_amount = 10000 * risk_per_trade  # Assuming $10k account
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk > 0:
            position_size = risk_amount / price_risk
            return min(position_size, 1000)  # Max 1000 shares
        
        return 100  # Default position size

class TradingEngine:
    """Main trading engine orchestrator"""
    
    def __init__(self):
        self.broker = PaperTradingBroker()
        self.signal_generator = ICTSignalGenerator()
        self.is_running = False
        self.active_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        self.performance_stats = {
            'signals_generated': 0,
            'orders_placed': 0,
            'profitable_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0
        }
    
    async def start(self):
        """Start the trading engine"""
        logger.info("Starting ICT Trading Engine")
        self.is_running = True
        
        # Start main trading loop
        await asyncio.gather(
            self._signal_generation_loop(),
            self._order_monitoring_loop(),
            self._performance_monitoring_loop()
        )
    
    async def stop(self):
        """Stop the trading engine"""
        logger.info("Stopping ICT Trading Engine")
        self.is_running = False
    
    async def _signal_generation_loop(self):
        """Main signal generation loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Generate signals
                signals = await self.signal_generator.generate_signals(self.active_symbols)
                
                # Process signals
                for signal in signals:
                    if signal.confidence > 0.6:  # Only high-confidence signals
                        await self._process_signal(signal)
                
                self.performance_stats['signals_generated'] += len(signals)
                
                execution_time = (time.time() - start_time) * 1000
                performance_monitor.record_execution_time("signal_generation", execution_time)
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in signal generation loop: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def _process_signal(self, signal: TradeSignal):
        """Process a trading signal"""
        try:
            # Check if we already have a position
            positions = await self.broker.get_positions()
            if signal.symbol in positions:
                logger.info(f"Already have position in {signal.symbol}, skipping signal")
                return
            
            # Create order
            order = Order(
                id="",  # Will be set by broker
                symbol=signal.symbol,
                side=OrderSide.BUY if signal.signal_type == 'buy' else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=signal.position_size,
                timestamp=datetime.now()
            )
            
            # Place order
            order_id = await self.broker.place_order(order)
            
            logger.info(f"Placed order for {signal.symbol}", 
                       signal_type=signal.signal_type,
                       confidence=signal.confidence,
                       order_id=order_id)
            
            self.performance_stats['orders_placed'] += 1
            
            trading_logger.log_signal_generation(asdict(signal))
            
        except Exception as e:
            logger.error(f"Error processing signal for {signal.symbol}: {e}")
    
    async def _order_monitoring_loop(self):
        """Monitor and manage active orders"""
        while self.is_running:
            try:
                # Check positions for exit conditions
                positions = await self.broker.get_positions()
                
                for symbol, position in positions.items():
                    await self._check_exit_conditions(symbol, position)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in order monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _check_exit_conditions(self, symbol: str, position: Position):
        """Check if position should be exited"""
        try:
            # Get current price
            market_data = await data_pipeline.get_real_time_data([symbol])
            if not market_data.get(symbol):
                return
            
            current_price = market_data[symbol][-1].close
            
            # Simple exit logic (this could be enhanced with ICT concepts)
            pnl_percent = (current_price - position.avg_price) / position.avg_price
            
            # Exit conditions
            should_exit = False
            exit_reason = ""
            
            if pnl_percent > 0.05:  # 5% profit
                should_exit = True
                exit_reason = "profit_target"
            elif pnl_percent < -0.02:  # 2% loss
                should_exit = True
                exit_reason = "stop_loss"
            
            if should_exit:
                # Create exit order
                exit_order = Order(
                    id="",
                    symbol=symbol,
                    side=OrderSide.SELL if position.quantity > 0 else OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=abs(position.quantity)
                )
                
                order_id = await self.broker.place_order(exit_order)
                
                logger.info(f"Exited position in {symbol}", 
                           reason=exit_reason,
                           pnl_percent=pnl_percent,
                           order_id=order_id)
                
                if pnl_percent > 0:
                    self.performance_stats['profitable_trades'] += 1
                else:
                    self.performance_stats['losing_trades'] += 1
                
                self.performance_stats['total_pnl'] += position.unrealized_pnl
                
        except Exception as e:
            logger.error(f"Error checking exit conditions for {symbol}: {e}")
    
    async def _performance_monitoring_loop(self):
        """Monitor and log performance metrics"""
        while self.is_running:
            try:
                account_info = await self.broker.get_account_info()
                
                performance_monitor.record_execution_time("account_equity", account_info['equity'])
                
                logger.info("Performance Update",
                           equity=account_info['equity'],
                           total_pnl=account_info['total_unrealized_pnl'] + account_info['total_realized_pnl'],
                           positions=account_info['positions_count'],
                           signals_generated=self.performance_stats['signals_generated'],
                           orders_placed=self.performance_stats['orders_placed'])
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        account_info = await self.broker.get_account_info()
        
        return {
            'is_running': self.is_running,
            'account_info': account_info,
            'performance_stats': self.performance_stats,
            'active_symbols': self.active_symbols,
            'timestamp': datetime.now().isoformat()
        }

# Global trading engine instance
trading_engine = TradingEngine()

__all__ = ['TradingEngine', 'TradeSignal', 'Order', 'Position', 'ICTSignalGenerator', 'trading_engine']