"""
Professional Position Management System for Real Trading
Implements proper position sizing, risk controls, and compliance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from decimal import Decimal, ROUND_HALF_UP

logger = logging.getLogger(__name__)

class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class PositionStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"

@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk per trade
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_position_size: float = 0.1  # 10% max position size
    max_sector_exposure: float = 0.3  # 30% max sector exposure
    max_correlation_exposure: float = 0.5  # 50% max correlated positions
    min_liquidity_adv: float = 0.1  # Min 10% of ADV (Average Daily Volume)
    max_drawdown: float = 0.15  # 15% max drawdown
    risk_free_rate: float = 0.05  # 5% risk-free rate

@dataclass
class Position:
    """Individual position tracking"""
    symbol: str
    side: PositionSide
    quantity: int
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    entry_time: datetime
    risk_amount: float
    position_value: float
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    
    def update_unrealized_pnl(self, current_price: float) -> float:
        """Update and return unrealized P&L"""
        if self.status != PositionStatus.OPEN:
            return self.pnl
        
        if self.side == PositionSide.LONG:
            unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            unrealized_pnl = (self.entry_price - current_price) * self.quantity
        
        # Update max excursions
        if unrealized_pnl > self.max_favorable_excursion:
            self.max_favorable_excursion = unrealized_pnl
        if unrealized_pnl < -self.max_adverse_excursion:
            self.max_adverse_excursion = -unrealized_pnl
        
        return unrealized_pnl

@dataclass
class Portfolio:
    """Portfolio state tracking"""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    closed_positions: List[Position] = field(default_factory=list)
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    
    def get_equity(self, market_prices: Dict[str, float]) -> float:
        """Calculate total portfolio equity"""
        unrealized_pnl = sum(
            pos.update_unrealized_pnl(market_prices.get(pos.symbol, pos.entry_price))
            for pos in self.positions.values()
        )
        return self.cash + unrealized_pnl
    
    def get_exposure(self, market_prices: Dict[str, float]) -> float:
        """Calculate total market exposure"""
        return sum(
            abs(pos.quantity * market_prices.get(pos.symbol, pos.entry_price))
            for pos in self.positions.values()
        )

class PositionSizer:
    """Advanced position sizing algorithms"""
    
    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
    
    def calculate_position_size(
        self,
        account_value: float,
        entry_price: float,
        stop_loss: float,
        confidence: float = 1.0,
        volatility: float = None,
        adv: float = None  # Average Daily Volume
    ) -> Dict:
        """
        Calculate optimal position size using multiple methodologies
        """
        try:
            # Method 1: Fixed fractional (Kelly-based)
            kelly_size = self._kelly_position_size(account_value, entry_price, stop_loss, confidence)
            
            # Method 2: Volatility-based sizing
            volatility_size = self._volatility_position_size(
                account_value, entry_price, volatility or 0.02
            ) if volatility else kelly_size
            
            # Method 3: Risk parity
            risk_parity_size = self._risk_parity_size(account_value, entry_price, stop_loss)
            
            # Take conservative approach - minimum of all methods
            base_size = min(kelly_size, volatility_size, risk_parity_size)
            
            # Apply liquidity constraints
            if adv:
                liquidity_constraint = adv * self.risk_limits.min_liquidity_adv
                max_shares = int(liquidity_constraint / entry_price)
                base_size = min(base_size, max_shares)
            
            # Apply position size limits
            max_position_value = account_value * self.risk_limits.max_position_size
            max_shares_by_limit = int(max_position_value / entry_price)
            final_size = min(base_size, max_shares_by_limit)
            
            # Calculate risk metrics
            risk_amount = abs(final_size * (entry_price - stop_loss))
            risk_percentage = risk_amount / account_value
            position_value = final_size * entry_price
            
            return {
                'shares': max(final_size, 0),
                'position_value': position_value,
                'risk_amount': risk_amount,
                'risk_percentage': risk_percentage,
                'kelly_size': kelly_size,
                'volatility_size': volatility_size,
                'risk_parity_size': risk_parity_size,
                'liquidity_constrained': adv is not None,
                'max_liquidity_shares': int(liquidity_constraint / entry_price) if adv else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {'shares': 0, 'position_value': 0, 'risk_amount': 0, 'risk_percentage': 0}
    
    def _kelly_position_size(self, account_value: float, entry_price: float, 
                           stop_loss: float, confidence: float) -> int:
        """Kelly criterion position sizing"""
        try:
            risk_per_share = abs(entry_price - stop_loss)
            max_risk = account_value * self.risk_limits.max_portfolio_risk
            
            # Kelly fraction adjusted by confidence
            # Assuming 60% win rate and 2:1 reward:risk ratio (conservative)
            win_rate = 0.6 * confidence
            avg_win = risk_per_share * 2
            avg_loss = risk_per_share
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            kelly_risk = account_value * kelly_fraction
            shares = int(min(kelly_risk, max_risk) / risk_per_share)
            
            return max(shares, 0)
            
        except Exception:
            return 0
    
    def _volatility_position_size(self, account_value: float, entry_price: float, 
                                volatility: float) -> int:
        """Volatility-based position sizing"""
        try:
            # Target volatility approach
            target_volatility = 0.15  # 15% annual target volatility
            
            # Calculate position size to achieve target volatility
            position_volatility = volatility * np.sqrt(252)  # Annualize
            volatility_scalar = target_volatility / position_volatility if position_volatility > 0 else 0
            
            max_position_value = account_value * min(volatility_scalar, self.risk_limits.max_position_size)
            shares = int(max_position_value / entry_price)
            
            return max(shares, 0)
            
        except Exception:
            return 0
    
    def _risk_parity_size(self, account_value: float, entry_price: float, stop_loss: float) -> int:
        """Risk parity position sizing"""
        try:
            risk_per_share = abs(entry_price - stop_loss)
            target_risk = account_value * self.risk_limits.max_portfolio_risk
            shares = int(target_risk / risk_per_share) if risk_per_share > 0 else 0
            
            return max(shares, 0)
            
        except Exception:
            return 0

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
        self.position_sizer = PositionSizer(risk_limits)
        
    def validate_new_position(
        self,
        portfolio: Portfolio,
        symbol: str,
        side: PositionSide,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        market_prices: Dict[str, float],
        sector_info: Dict[str, str] = None,
        correlation_matrix: pd.DataFrame = None
    ) -> Dict:
        """
        Comprehensive position validation before entry
        """
        validation = {
            'approved': True,
            'reasons': [],
            'warnings': [],
            'risk_metrics': {}
        }
        
        try:
            current_equity = portfolio.get_equity(market_prices)
            position_value = quantity * entry_price
            risk_amount = abs(quantity * (entry_price - stop_loss))
            
            # 1. Portfolio risk check
            portfolio_risk = risk_amount / current_equity
            if portfolio_risk > self.risk_limits.max_portfolio_risk:
                validation['approved'] = False
                validation['reasons'].append(
                    f'Portfolio risk {portfolio_risk:.2%} exceeds limit {self.risk_limits.max_portfolio_risk:.2%}'
                )
            
            # 2. Position size check
            position_percentage = position_value / current_equity
            if position_percentage > self.risk_limits.max_position_size:
                validation['approved'] = False
                validation['reasons'].append(
                    f'Position size {position_percentage:.2%} exceeds limit {self.risk_limits.max_position_size:.2%}'
                )
            
            # 3. Daily loss check
            if portfolio.daily_pnl < -current_equity * self.risk_limits.max_daily_loss:
                validation['approved'] = False
                validation['reasons'].append(
                    f'Daily loss limit reached: {portfolio.daily_pnl/current_equity:.2%}'
                )
            
            # 4. Drawdown check
            if portfolio.max_drawdown > self.risk_limits.max_drawdown:
                validation['approved'] = False
                validation['reasons'].append(
                    f'Max drawdown exceeded: {portfolio.max_drawdown:.2%}'
                )
            
            # 5. Existing position check (prevent doubling down without approval)
            if symbol in portfolio.positions:
                existing_pos = portfolio.positions[symbol]
                if existing_pos.side == side:
                    validation['warnings'].append(
                        f'Adding to existing {side.value} position in {symbol}'
                    )
            
            # 6. Sector exposure check
            if sector_info and symbol in sector_info:
                sector = sector_info[symbol]
                sector_exposure = self._calculate_sector_exposure(
                    portfolio, sector, sector_info, market_prices
                )
                sector_exposure += position_value  # Add new position
                
                if sector_exposure / current_equity > self.risk_limits.max_sector_exposure:
                    validation['approved'] = False
                    validation['reasons'].append(
                        f'Sector exposure {sector_exposure/current_equity:.2%} exceeds limit'
                    )
            
            # 7. Correlation check
            if correlation_matrix is not None and symbol in correlation_matrix.index:
                corr_exposure = self._calculate_correlation_exposure(
                    portfolio, symbol, correlation_matrix, market_prices
                )
                if corr_exposure > self.risk_limits.max_correlation_exposure:
                    validation['warnings'].append(
                        f'High correlation exposure: {corr_exposure:.2%}'
                    )
            
            # Store risk metrics
            validation['risk_metrics'] = {
                'portfolio_risk': portfolio_risk,
                'position_percentage': position_percentage,
                'risk_amount': risk_amount,
                'position_value': position_value,
                'current_equity': current_equity,
                'daily_pnl': portfolio.daily_pnl,
                'max_drawdown': portfolio.max_drawdown
            }
            
        except Exception as e:
            logger.error(f"Error validating position: {e}")
            validation['approved'] = False
            validation['reasons'].append(f'Validation error: {str(e)}')
        
        return validation
    
    def calculate_dynamic_stop_loss(
        self,
        entry_price: float,
        side: PositionSide,
        atr: float,
        support_resistance: List[float] = None,
        volatility: float = None
    ) -> float:
        """Calculate dynamic stop loss based on market conditions"""
        try:
            if side == PositionSide.LONG:
                # For long positions, stop below entry
                base_stop = entry_price - (atr * 1.5)  # 1.5 ATR stop
                
                # Adjust for support levels
                if support_resistance:
                    nearby_support = max([level for level in support_resistance if level < entry_price], default=base_stop)
                    base_stop = min(base_stop, nearby_support - (atr * 0.5))
                
            else:  # SHORT
                # For short positions, stop above entry
                base_stop = entry_price + (atr * 1.5)  # 1.5 ATR stop
                
                # Adjust for resistance levels
                if support_resistance:
                    nearby_resistance = min([level for level in support_resistance if level > entry_price], default=base_stop)
                    base_stop = max(base_stop, nearby_resistance + (atr * 0.5))
            
            # Volatility adjustment
            if volatility:
                volatility_multiplier = min(max(volatility / 0.02, 0.5), 2.0)  # Scale between 0.5x and 2x
                if side == PositionSide.LONG:
                    base_stop = entry_price - abs(entry_price - base_stop) * volatility_multiplier
                else:
                    base_stop = entry_price + abs(base_stop - entry_price) * volatility_multiplier
            
            return round(base_stop, 2)
            
        except Exception as e:
            logger.error(f"Error calculating dynamic stop: {e}")
            # Fallback to 2% stop
            return entry_price * (0.98 if side == PositionSide.LONG else 1.02)
    
    def _calculate_sector_exposure(
        self,
        portfolio: Portfolio,
        sector: str,
        sector_info: Dict[str, str],
        market_prices: Dict[str, float]
    ) -> float:
        """Calculate current sector exposure"""
        sector_exposure = 0.0
        
        for symbol, position in portfolio.positions.items():
            if sector_info.get(symbol) == sector:
                position_value = abs(position.quantity * market_prices.get(symbol, position.entry_price))
                sector_exposure += position_value
        
        return sector_exposure
    
    def _calculate_correlation_exposure(
        self,
        portfolio: Portfolio,
        symbol: str,
        correlation_matrix: pd.DataFrame,
        market_prices: Dict[str, float]
    ) -> float:
        """Calculate correlation-adjusted exposure"""
        try:
            total_corr_exposure = 0.0
            
            for pos_symbol, position in portfolio.positions.items():
                if pos_symbol in correlation_matrix.index:
                    correlation = correlation_matrix.loc[symbol, pos_symbol]
                    position_value = abs(position.quantity * market_prices.get(pos_symbol, position.entry_price))
                    corr_exposure = position_value * abs(correlation)
                    total_corr_exposure += corr_exposure
            
            return total_corr_exposure
            
        except Exception:
            return 0.0

class PositionManager:
    """Main position management interface"""
    
    def __init__(self, initial_capital: float, risk_limits: RiskLimits = None):
        self.risk_limits = risk_limits or RiskLimits()
        self.risk_manager = RiskManager(self.risk_limits)
        self.portfolio = Portfolio(cash=initial_capital)
        self.portfolio.peak_equity = initial_capital
        
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        confidence: float = 1.0,
        market_data: Dict = None
    ) -> Dict:
        """Calculate optimal position size for new position"""
        current_equity = self.portfolio.cash  # Simplified
        
        volatility = market_data.get('volatility') if market_data else None
        adv = market_data.get('average_daily_volume') if market_data else None
        
        return self.risk_manager.position_sizer.calculate_position_size(
            current_equity, entry_price, stop_loss, confidence, volatility, adv
        )
    
    def validate_trade(
        self,
        symbol: str,
        side: PositionSide,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        market_prices: Dict[str, float],
        market_context: Dict = None
    ) -> Dict:
        """Validate trade before execution"""
        return self.risk_manager.validate_new_position(
            self.portfolio,
            symbol,
            side,
            quantity,
            entry_price,
            stop_loss,
            market_prices,
            market_context.get('sector_info') if market_context else None,
            market_context.get('correlation_matrix') if market_context else None
        )
    
    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float = None
    ) -> bool:
        """Open new position after validation"""
        try:
            # Validate first
            market_prices = {symbol: entry_price}
            validation = self.validate_trade(symbol, side, quantity, entry_price, stop_loss, market_prices)
            
            if not validation['approved']:
                logger.warning(f"Position rejected: {validation['reasons']}")
                return False
            
            # Create position
            position_value = quantity * entry_price
            risk_amount = abs(quantity * (entry_price - stop_loss))
            
            position = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_time=datetime.now(),
                risk_amount=risk_amount,
                position_value=position_value
            )
            
            # Update portfolio
            self.portfolio.positions[symbol] = position
            self.portfolio.cash -= position_value  # Simplified - assumes full cash purchase
            
            logger.info(f"Opened {side.value} position: {quantity} shares of {symbol} at ${entry_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return False
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "Manual close"
    ) -> bool:
        """Close existing position"""
        try:
            if symbol not in self.portfolio.positions:
                logger.warning(f"No open position found for {symbol}")
                return False
            
            position = self.portfolio.positions[symbol]
            
            # Calculate P&L
            if position.side == PositionSide.LONG:
                pnl = (exit_price - position.entry_price) * position.quantity
            else:
                pnl = (position.entry_price - exit_price) * position.quantity
            
            # Update position
            position.exit_price = exit_price
            position.exit_time = datetime.now()
            position.pnl = pnl
            position.status = PositionStatus.CLOSED
            
            # Update portfolio
            self.portfolio.cash += position.quantity * exit_price  # Simplified
            self.portfolio.total_pnl += pnl
            self.portfolio.daily_pnl += pnl
            
            # Move to closed positions
            self.portfolio.closed_positions.append(position)
            del self.portfolio.positions[symbol]
            
            logger.info(f"Closed position: {symbol} at ${exit_price:.2f}, P&L: ${pnl:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def update_portfolio(self, market_prices: Dict[str, float]) -> Dict:
        """Update portfolio with current market prices"""
        try:
            current_equity = self.portfolio.get_equity(market_prices)
            
            # Update peak equity and drawdown
            if current_equity > self.portfolio.peak_equity:
                self.portfolio.peak_equity = current_equity
            
            current_drawdown = (self.portfolio.peak_equity - current_equity) / self.portfolio.peak_equity
            if current_drawdown > self.portfolio.max_drawdown:
                self.portfolio.max_drawdown = current_drawdown
            
            # Calculate metrics
            total_exposure = self.portfolio.get_exposure(market_prices)
            
            return {
                'equity': current_equity,
                'cash': self.portfolio.cash,
                'exposure': total_exposure,
                'daily_pnl': self.portfolio.daily_pnl,
                'total_pnl': self.portfolio.total_pnl,
                'max_drawdown': self.portfolio.max_drawdown,
                'open_positions': len(self.portfolio.positions),
                'leverage': total_exposure / current_equity if current_equity > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
            return {}
    
    def get_risk_report(self, market_prices: Dict[str, float]) -> Dict:
        """Generate comprehensive risk report"""
        try:
            portfolio_metrics = self.update_portfolio(market_prices)
            
            # Position-level risk
            position_risks = []
            for symbol, position in self.portfolio.positions.items():
                current_price = market_prices.get(symbol, position.entry_price)
                unrealized_pnl = position.update_unrealized_pnl(current_price)
                
                position_risks.append({
                    'symbol': symbol,
                    'side': position.side.value,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': current_price,
                    'unrealized_pnl': unrealized_pnl,
                    'risk_amount': position.risk_amount,
                    'stop_loss': position.stop_loss,
                    'position_value': position.quantity * current_price,
                    'days_held': (datetime.now() - position.entry_time).days
                })
            
            return {
                'timestamp': datetime.now(),
                'portfolio_metrics': portfolio_metrics,
                'position_risks': position_risks,
                'risk_limits': {
                    'max_portfolio_risk': self.risk_limits.max_portfolio_risk,
                    'max_daily_loss': self.risk_limits.max_daily_loss,
                    'max_position_size': self.risk_limits.max_position_size,
                    'max_drawdown': self.risk_limits.max_drawdown
                },
                'compliance_status': {
                    'within_daily_loss_limit': portfolio_metrics['daily_pnl'] > -portfolio_metrics['equity'] * self.risk_limits.max_daily_loss,
                    'within_drawdown_limit': portfolio_metrics['max_drawdown'] < self.risk_limits.max_drawdown,
                    'leverage_acceptable': portfolio_metrics['leverage'] <= 1.0
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return {}

# Global instance for the application
position_manager = PositionManager(100000)  # $100k initial capital