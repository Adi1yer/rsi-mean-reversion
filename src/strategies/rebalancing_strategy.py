"""
Monthly Rebalancing Strategy
Maintains target allocations and rebalances portfolio monthly.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RebalancingStrategy:
    """Implements monthly rebalancing to maintain target allocations."""
    
    def __init__(self, 
                 dividend_target: float = 0.70,
                 growth_target: float = 0.30,
                 rebalancing_threshold: float = 0.05):
        """
        Initialize rebalancing strategy.
        
        Args:
            dividend_target: Target allocation for dividend aristocrats
            growth_target: Target allocation for high-growth stocks
            rebalancing_threshold: Threshold for triggering rebalancing (5%)
        """
        self.dividend_target = dividend_target
        self.growth_target = growth_target
        self.rebalancing_threshold = rebalancing_threshold
        self.last_rebalance_date = None
        
    def should_rebalance(self, 
                        current_dividend_allocation: float,
                        current_growth_allocation: float,
                        current_date: datetime) -> bool:
        """
        Determine if rebalancing is needed.
        
        Args:
            current_dividend_allocation: Current dividend allocation
            current_growth_allocation: Current growth allocation
            current_date: Current date
            
        Returns:
            True if rebalancing is needed
        """
        # Check if it's been a month since last rebalance
        if self.last_rebalance_date is None:
            return True
        
        days_since_rebalance = (current_date - self.last_rebalance_date).days
        if days_since_rebalance < 30:
            return False
        
        # Check if allocations have drifted beyond threshold
        dividend_drift = abs(current_dividend_allocation - self.dividend_target)
        growth_drift = abs(current_growth_allocation - self.growth_target)
        
        return dividend_drift > self.rebalancing_threshold or growth_drift > self.rebalancing_threshold
    
    def calculate_rebalancing_trades(self,
                                   positions: Dict,
                                   current_prices: Dict[str, float],
                                   target_weights: Dict[str, float],
                                   total_portfolio_value: float) -> List[Dict]:
        """
        Calculate rebalancing trades needed.
        
        Args:
            positions: Current positions
            current_prices: Current prices for all symbols
            target_weights: Target weights for each symbol
            total_portfolio_value: Total portfolio value
            
        Returns:
            List of rebalancing trades
        """
        trades = []
        
        for symbol, target_weight in target_weights.items():
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            current_shares = positions.get(symbol, {}).get('shares', 0)
            current_value = current_shares * current_price
            target_value = total_portfolio_value * target_weight
            
            # Calculate shares needed
            target_shares = target_value / current_price
            shares_to_trade = target_shares - current_shares
            
            if abs(shares_to_trade) > 0.01:  # Minimum trade size
                trade = {
                    'symbol': symbol,
                    'action': 'BUY' if shares_to_trade > 0 else 'SELL',
                    'shares': abs(shares_to_trade),
                    'price': current_price,
                    'value': abs(shares_to_trade) * current_price,
                    'reason': 'rebalancing'
                }
                trades.append(trade)
        
        return trades
    
    def update_rebalance_date(self, rebalance_date: datetime):
        """Update the last rebalance date."""
        self.last_rebalance_date = rebalance_date
    
    def get_allocation_drift(self,
                           current_dividend_allocation: float,
                           current_growth_allocation: float) -> Dict[str, float]:
        """
        Calculate allocation drift from targets.
        
        Args:
            current_dividend_allocation: Current dividend allocation
            current_growth_allocation: Current growth allocation
            
        Returns:
            Dictionary with drift information
        """
        dividend_drift = current_dividend_allocation - self.dividend_target
        growth_drift = current_growth_allocation - self.growth_target
        
        return {
            'dividend_drift': dividend_drift,
            'growth_drift': growth_drift,
            'total_drift': abs(dividend_drift) + abs(growth_drift)
        }

class DynamicRebalancingStrategy(RebalancingStrategy):
    """Enhanced rebalancing with dynamic thresholds and momentum consideration."""
    
    def __init__(self,
                 dividend_target: float = 0.70,
                 growth_target: float = 0.30,
                 base_rebalancing_threshold: float = 0.05,
                 momentum_factor: float = 0.1):
        """
        Initialize dynamic rebalancing strategy.
        
        Args:
            dividend_target: Target allocation for dividend aristocrats
            growth_target: Target allocation for high-growth stocks
            base_rebalancing_threshold: Base threshold for rebalancing
            momentum_factor: Factor to adjust threshold based on momentum
        """
        super().__init__(dividend_target, growth_target, base_rebalancing_threshold)
        self.momentum_factor = momentum_factor
        self.price_history = {}
        
    def update_price_history(self, symbol: str, price: float, date: datetime):
        """Update price history for momentum calculation."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            'price': price,
            'date': date
        })
        
        # Keep only last 30 days of history
        cutoff_date = date - timedelta(days=30)
        self.price_history[symbol] = [
            entry for entry in self.price_history[symbol]
            if entry['date'] >= cutoff_date
        ]
    
    def calculate_momentum(self, symbol: str) -> float:
        """Calculate price momentum for a symbol."""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
            return 0.0
        
        prices = [entry['price'] for entry in self.price_history[symbol]]
        if len(prices) < 2:
            return 0.0
        
        # Calculate momentum as percentage change over last 10 days
        momentum = ((prices[-1] - prices[0]) / prices[0]) * 100
        return momentum
    
    def get_dynamic_threshold(self, symbol: str) -> float:
        """Get dynamic rebalancing threshold based on momentum."""
        momentum = self.calculate_momentum(symbol)
        
        # Adjust threshold based on momentum
        # Higher momentum = higher threshold (less frequent rebalancing)
        momentum_adjustment = momentum * self.momentum_factor
        dynamic_threshold = self.rebalancing_threshold + momentum_adjustment
        
        # Clamp between 0.02 and 0.10
        return max(0.02, min(0.10, dynamic_threshold))
    
    def should_rebalance(self,
                        current_dividend_allocation: float,
                        current_growth_allocation: float,
                        current_date: datetime,
                        symbols: List[str]) -> bool:
        """
        Enhanced rebalancing decision with momentum consideration.
        
        Args:
            current_dividend_allocation: Current dividend allocation
            current_growth_allocation: Current growth allocation
            current_date: Current date
            symbols: List of symbols to consider
            
        Returns:
            True if rebalancing is needed
        """
        # Check time-based rebalancing
        if self.last_rebalance_date is None:
            return True
        
        days_since_rebalance = (current_date - self.last_rebalance_date).days
        if days_since_rebalance < 30:
            return False
        
        # Check allocation drift with dynamic thresholds
        dividend_drift = abs(current_dividend_allocation - self.dividend_target)
        growth_drift = abs(current_growth_allocation - self.growth_target)
        
        # Use average dynamic threshold for simplicity
        avg_threshold = float(np.mean([self.get_dynamic_threshold(symbol) for symbol in symbols]))
        
        return dividend_drift > avg_threshold or growth_drift > avg_threshold

def test_rebalancing_strategy():
    """Test the rebalancing strategy."""
    print("Testing Rebalancing Strategy:")
    print("=" * 40)
    
    # Test basic rebalancing
    strategy = RebalancingStrategy()
    
    # Simulate current allocations
    current_dividend = 0.75  # 5% drift
    current_growth = 0.25   # 5% drift
    
    should_rebalance = strategy.should_rebalance(
        current_dividend, current_growth, datetime.now()
    )
    
    print(f"Current Dividend Allocation: {current_dividend:.2f}")
    print(f"Current Growth Allocation: {current_growth:.2f}")
    print(f"Should Rebalance: {should_rebalance}")
    
    # Test drift calculation
    drift = strategy.get_allocation_drift(current_dividend, current_growth)
    print(f"Dividend Drift: {drift['dividend_drift']:.3f}")
    print(f"Growth Drift: {drift['growth_drift']:.3f}")
    print(f"Total Drift: {drift['total_drift']:.3f}")
    
    # Test dynamic rebalancing
    print("\nTesting Dynamic Rebalancing:")
    print("=" * 40)
    
    dynamic_strategy = DynamicRebalancingStrategy()
    
    # Simulate price history
    test_date = datetime.now()
    for i in range(30):
        price = 100 + i * 0.5  # Upward trend
        dynamic_strategy.update_price_history("AAPL", price, test_date - timedelta(days=30-i))
    
    momentum = dynamic_strategy.calculate_momentum("AAPL")
    threshold = dynamic_strategy.get_dynamic_threshold("AAPL")
    
    print(f"AAPL Momentum: {momentum:.2f}%")
    print(f"Dynamic Threshold: {threshold:.3f}")

if __name__ == "__main__":
    test_rebalancing_strategy() 