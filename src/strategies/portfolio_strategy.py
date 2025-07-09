"""
Portfolio Strategy
Implements 70% dividend aristocrats / 30% high-growth allocation with RSI-based trading rules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

from src.data.fundamental_analyzer import FundamentalAnalyzer, CompanyType
from src.data.dividend_aristocrat_analyzer import DividendAristocratAnalyzer, DividendStatus
from src.strategies.rsi_strategy import RSIStrategy, StockCategory, SignalType

logger = logging.getLogger(__name__)

class PortfolioAllocation(Enum):
    """Portfolio allocation types."""
    DIVIDEND_ARISTOCRATS = "dividend_aristocrats"  # 70% allocation
    HIGH_GROWTH_UNPROFITABLE = "high_growth_unprofitable"  # 30% allocation

class TradingAction(Enum):
    """Trading actions based on RSI and stock type."""
    BUY_DIVIDEND_ARISTOCRAT = "buy_dividend_aristocrat"
    SELL_COVERED_CALL_DIVIDEND = "sell_covered_call_dividend"  # RSI > 70
    BUY_BULL_PUT_SPREAD_GROWTH = "buy_bull_put_spread_growth"  # RSI < 30
    SELL_BEAR_CALL_SPREAD_GROWTH = "sell_bear_call_spread_growth"  # RSI > 70
    HOLD = "hold"

@dataclass
class PortfolioSignal:
    """Portfolio trading signal with allocation context."""
    action: TradingAction
    symbol: str
    category: PortfolioAllocation
    rsi_value: float
    current_price: float
    allocation_weight: float
    confidence: float
    timestamp: datetime
    fundamental_metrics: Optional[Dict] = None

class PortfolioStrategy:
    """
    Portfolio strategy with 70% dividend aristocrats / 30% high-growth allocation.
    
    Trading Rules:
    - Dividend Aristocrats (70%): Buy and hold, sell covered calls when RSI > 70
    - High-Growth Unprofitable (30%): Use aggressive spreads based on RSI
    """
    
    def __init__(self, 
                 dividend_alloc: float = 0.70,  # 70% dividend aristocrats
                 growth_alloc: float = 0.30,    # 30% high-growth unprofitable
                 dividend_rsi_threshold: float = 70.0,  # RSI threshold for dividend covered calls
                 growth_oversold: float = 30.0,  # RSI oversold for growth stocks
                 growth_overbought: float = 70.0):  # RSI overbought for growth stocks
        
        # Portfolio allocation
        self.dividend_allocation = dividend_alloc
        self.growth_allocation = growth_alloc
        
        # RSI thresholds
        self.dividend_rsi_threshold = dividend_rsi_threshold
        self.growth_oversold = growth_oversold
        self.growth_overbought = growth_overbought
        
        # Initialize analyzers
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.dividend_analyzer = DividendAristocratAnalyzer()
        self.rsi_strategy = RSIStrategy(fundamental_analyzer=self.fundamental_analyzer)
        
        # Portfolio tracking
        self.dividend_aristocrats = []
        self.high_growth_unprofitable = []
        self.portfolio_signals = []
        
    def scan_for_eligible_companies(self, symbols: List[str]) -> Dict:
        """Scan for eligible dividend aristocrats and high-growth companies."""
        logger.info("🔍 Scanning for eligible companies...")
        
        # Analyze dividend aristocrats
        dividend_report = self.dividend_analyzer.generate_dividend_report(symbols)
        eligible_dividends = dividend_report['eligible_aristocrats']
        
        # Analyze high-growth unprofitable
        fundamental_report = self.fundamental_analyzer.generate_analysis_report(symbols)
        eligible_growth = []
        
        for symbol, metrics in fundamental_report['detailed_analysis'].items():
            if (metrics.is_eligible and 
                metrics.company_type == CompanyType.HIGH_GROWTH_UNPROFITABLE):
                eligible_growth.append(symbol)
        
        # Update portfolio lists
        self.dividend_aristocrats = eligible_dividends
        self.high_growth_unprofitable = eligible_growth
        
        results = {
            'dividend_aristocrats': {
                'count': len(eligible_dividends),
                'symbols': eligible_dividends,
                'allocation': self.dividend_allocation,
                'target_weight': self.dividend_allocation / len(eligible_dividends) if eligible_dividends else 0
            },
            'high_growth_unprofitable': {
                'count': len(eligible_growth),
                'symbols': eligible_growth,
                'allocation': self.growth_allocation,
                'target_weight': self.growth_allocation / len(eligible_growth) if eligible_growth else 0
            },
            'total_eligible': len(eligible_dividends) + len(eligible_growth)
        }
        
        logger.info(f"✅ Found {len(eligible_dividends)} dividend aristocrats")
        logger.info(f"✅ Found {len(eligible_growth)} high-growth unprofitable companies")
        
        return results
    
    def generate_portfolio_signal(self, symbol: str, data: pd.DataFrame, 
                                portfolio_value: float = 100000) -> PortfolioSignal:
        """
        Generate portfolio-specific trading signal based on stock type and RSI.
        
        Args:
            symbol: Stock symbol
            data: OHLCV data
            portfolio_value: Current portfolio value
            
        Returns:
            PortfolioSignal object
        """
        try:
            # Determine stock category
            category = self._get_stock_category(symbol)
            
            if category == PortfolioAllocation.DIVIDEND_ARISTOCRATS:
                return self._generate_dividend_signal(symbol, data, portfolio_value)
            elif category == PortfolioAllocation.HIGH_GROWTH_UNPROFITABLE:
                return self._generate_growth_signal(symbol, data, portfolio_value)
            else:
                return self._generate_hold_signal(symbol, data)
                
        except Exception as e:
            logger.error(f"Error generating portfolio signal for {symbol}: {e}")
            return self._generate_hold_signal(symbol, data)
    
    def _get_stock_category(self, symbol: str) -> PortfolioAllocation:
        """Determine if stock is dividend aristocrat or high-growth unprofitable."""
        if symbol in self.dividend_aristocrats:
            return PortfolioAllocation.DIVIDEND_ARISTOCRATS
        elif symbol in self.high_growth_unprofitable:
            return PortfolioAllocation.HIGH_GROWTH_UNPROFITABLE
        else:
            # Analyze on-the-fly if not in lists
            try:
                # Check dividend status first
                dividend_metrics = self.dividend_analyzer.analyze_dividend_history(symbol)
                if dividend_metrics.is_eligible:
                    return PortfolioAllocation.DIVIDEND_ARISTOCRATS
                
                # Check fundamental status
                fundamental_metrics = self.fundamental_analyzer.analyze_company(symbol)
                if (fundamental_metrics.is_eligible and 
                    fundamental_metrics.company_type == CompanyType.HIGH_GROWTH_UNPROFITABLE):
                    return PortfolioAllocation.HIGH_GROWTH_UNPROFITABLE
                
            except Exception as e:
                logger.debug(f"Error categorizing {symbol}: {e}")
            
            return PortfolioAllocation.DIVIDEND_ARISTOCRATS  # Default to dividend
    
    def _generate_dividend_signal(self, symbol: str, data: pd.DataFrame, 
                                 portfolio_value: float) -> PortfolioSignal:
        """Generate signal for dividend aristocrats."""
        # Calculate RSI
        rsi = self.rsi_strategy.calculate_rsi(data['Close'])
        current_rsi = rsi.iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        # Dividend aristocrat trading rules
        if current_rsi > self.dividend_rsi_threshold:
            # RSI > 70: Sell covered call
            action = TradingAction.SELL_COVERED_CALL_DIVIDEND
            confidence = min(0.9, (current_rsi - self.dividend_rsi_threshold) / (100 - self.dividend_rsi_threshold))
        else:
            # RSI <= 70: Buy and hold
            action = TradingAction.BUY_DIVIDEND_ARISTOCRAT
            confidence = 0.8
        
        # Calculate allocation weight
        allocation_weight = self.dividend_allocation / len(self.dividend_aristocrats) if self.dividend_aristocrats else 0
        
        return PortfolioSignal(
            action=action,
            symbol=symbol,
            category=PortfolioAllocation.DIVIDEND_ARISTOCRATS,
            rsi_value=current_rsi,
            current_price=current_price,
            allocation_weight=allocation_weight,
            confidence=confidence,
            timestamp=data.index[-1],
            fundamental_metrics={
                'rsi_threshold': self.dividend_rsi_threshold,
                'strategy': 'buy_and_hold_with_covered_calls'
            }
        )
    
    def _generate_growth_signal(self, symbol: str, data: pd.DataFrame, 
                               portfolio_value: float) -> PortfolioSignal:
        """Generate signal for high-growth unprofitable companies."""
        # Calculate RSI
        rsi = self.rsi_strategy.calculate_rsi(data['Close'])
        current_rsi = rsi.iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        # High-growth unprofitable trading rules
        if current_rsi < self.growth_oversold:
            # RSI < 30: Buy bull put spread
            action = TradingAction.BUY_BULL_PUT_SPREAD_GROWTH
            confidence = min(0.9, (self.growth_oversold - current_rsi) / self.growth_oversold)
        elif current_rsi > self.growth_overbought:
            # RSI > 70: Sell bear call spread
            action = TradingAction.SELL_BEAR_CALL_SPREAD_GROWTH
            confidence = min(0.9, (current_rsi - self.growth_overbought) / (100 - self.growth_overbought))
        else:
            # 30 <= RSI <= 70: Hold
            action = TradingAction.HOLD
            confidence = 0.5
        
        # Calculate allocation weight
        allocation_weight = self.growth_allocation / len(self.high_growth_unprofitable) if self.high_growth_unprofitable else 0
        
        return PortfolioSignal(
            action=action,
            symbol=symbol,
            category=PortfolioAllocation.HIGH_GROWTH_UNPROFITABLE,
            rsi_value=current_rsi,
            current_price=current_price,
            allocation_weight=allocation_weight,
            confidence=confidence,
            timestamp=data.index[-1],
            fundamental_metrics={
                'oversold_threshold': self.growth_oversold,
                'overbought_threshold': self.growth_overbought,
                'strategy': 'aggressive_spreads'
            }
        )
    
    def _generate_hold_signal(self, symbol: str, data: pd.DataFrame) -> PortfolioSignal:
        """Generate hold signal for unqualified stocks."""
        return PortfolioSignal(
            action=TradingAction.HOLD,
            symbol=symbol,
            category=PortfolioAllocation.DIVIDEND_ARISTOCRATS,  # Default
            rsi_value=50.0,
            current_price=data['Close'].iloc[-1] if len(data) > 0 else 0,
            allocation_weight=0.0,
            confidence=0.0,
            timestamp=data.index[-1] if len(data) > 0 else datetime.now()
        )
    
    def get_portfolio_allocation(self) -> Dict:
        """Get current portfolio allocation."""
        return {
            'dividend_aristocrats': {
                'allocation': self.dividend_allocation,
                'symbols': self.dividend_aristocrats,
                'count': len(self.dividend_aristocrats),
                'target_weight_per_stock': self.dividend_allocation / len(self.dividend_aristocrats) if self.dividend_aristocrats else 0
            },
            'high_growth_unprofitable': {
                'allocation': self.growth_allocation,
                'symbols': self.high_growth_unprofitable,
                'count': len(self.high_growth_unprofitable),
                'target_weight_per_stock': self.growth_allocation / len(self.high_growth_unprofitable) if self.high_growth_unprofitable else 0
            }
        }
    
    def get_trading_rules_summary(self) -> Dict:
        """Get summary of trading rules."""
        return {
            'portfolio_allocation': {
                'dividend_aristocrats': f"{self.dividend_allocation*100:.0f}%",
                'high_growth_unprofitable': f"{self.growth_allocation*100:.0f}%"
            },
            'trading_rules': {
                'dividend_aristocrats': {
                    'strategy': 'Buy and hold, sell covered calls when RSI > 70',
                    'rsi_threshold': self.dividend_rsi_threshold,
                    'actions': ['BUY_DIVIDEND_ARISTOCRAT', 'SELL_COVERED_CALL_DIVIDEND']
                },
                'high_growth_unprofitable': {
                    'strategy': 'Aggressive spreads based on RSI',
                    'oversold_threshold': self.growth_oversold,
                    'overbought_threshold': self.growth_overbought,
                    'actions': ['BUY_BULL_PUT_SPREAD_GROWTH', 'SELL_BEAR_CALL_SPREAD_GROWTH']
                }
            }
        }
    
    def validate_portfolio_balance(self) -> Dict:
        """Validate portfolio balance and provide recommendations."""
        dividend_count = len(self.dividend_aristocrats)
        growth_count = len(self.high_growth_unprofitable)
        
        validation = {
            'current_balance': {
                'dividend_aristocrats': dividend_count,
                'high_growth_unprofitable': growth_count,
                'total_eligible': dividend_count + growth_count
            },
            'recommendations': []
        }
        
        # Check minimum requirements
        if dividend_count < 5:
            validation['recommendations'].append(
                "Add more dividend aristocrats for portfolio stability (minimum 5 recommended)"
            )
        
        if growth_count < 3:
            validation['recommendations'].append(
                "Add more high-growth unprofitable companies for growth exposure (minimum 3 recommended)"
            )
        
        # Check allocation balance
        if dividend_count > 0 and growth_count > 0:
            actual_dividend_ratio = dividend_count / (dividend_count + growth_count)
            target_ratio = self.dividend_allocation
            
            if abs(actual_dividend_ratio - target_ratio) > 0.1:  # 10% tolerance
                validation['recommendations'].append(
                    f"Portfolio allocation may need adjustment (current: {actual_dividend_ratio:.1%}, target: {target_ratio:.1%})"
                )
        
        return validation 

    def calculate_exponential_weights(self, companies: List[Dict]) -> Dict[str, float]:
        """
        Calculate exponential weights based on operating income growth.
        
        Args:
            companies: List of company dicts with operating_income_growth field
            
        Returns:
            Dict mapping symbol to weight
        """
        if not companies:
            return {}
        
        # Sort by operating income growth (highest first)
        sorted_companies = sorted(companies, key=lambda x: x.get('operating_income_growth', 0), reverse=True)
        
        # Get growth rates
        growth_rates = [company.get('operating_income_growth', 0) for company in sorted_companies]
        
        if not growth_rates or all(rate <= 0 for rate in growth_rates):
            # If no positive growth, use equal weights
            weight_per_company = 1.0 / len(sorted_companies)
            return {company['symbol']: weight_per_company for company in sorted_companies}
        
        # Calculate exponential weights
        # Use exp(growth_rate / 100) to normalize and emphasize differences
        exp_weights = [np.exp(rate / 100) for rate in growth_rates]
        total_weight = sum(exp_weights)
        
        # Normalize to sum to 1
        normalized_weights = [weight / total_weight for weight in exp_weights]
        
        # Create result dict
        weights = {}
        for i, company in enumerate(sorted_companies):
            weights[company['symbol']] = normalized_weights[i]
        
        return weights
    
    def get_latest_portfolio_allocation(self, dividend_aristocrats: List[Dict], 
                                      high_growth_stocks: List[Dict]) -> Dict:
        """
        Get the latest portfolio allocation based on current data.
        
        Args:
            dividend_aristocrats: List of dividend aristocrat companies with financial data
            high_growth_stocks: List of high-growth unprofitable companies
            
        Returns:
            Portfolio allocation dict
        """
        # Calculate weights for dividend aristocrats (70% of portfolio)
        dividend_weights = self.calculate_exponential_weights(dividend_aristocrats)
        
        # Equal weights for high-growth stocks (30% of portfolio)
        growth_weight_per_stock = 1.0 / len(high_growth_stocks) if high_growth_stocks else 0
        
        # Apply portfolio allocation percentages
        dividend_allocation = {}
        for symbol, weight in dividend_weights.items():
            dividend_allocation[symbol] = weight * 0.70  # 70% of portfolio
        
        growth_allocation = {}
        for stock in high_growth_stocks:
            growth_allocation[stock['symbol']] = growth_weight_per_stock * 0.30  # 30% of portfolio
        
        return {
            'dividend_aristocrats': dividend_allocation,
            'high_growth_unprofitable': growth_allocation,
            'total_allocation': sum(dividend_allocation.values()) + sum(growth_allocation.values())
        } 