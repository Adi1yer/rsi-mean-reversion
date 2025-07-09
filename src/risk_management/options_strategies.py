"""
Advanced Options Strategies
Implements sophisticated options strategies for enhanced risk management and income generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class OptionsStrategy:
    """Base class for options strategies."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def calculate_implied_volatility(self, 
                                   option_price: float,
                                   stock_price: float,
                                   strike_price: float,
                                   time_to_expiry: float,
                                   option_type: str) -> float:
        """
        Calculate implied volatility using Black-Scholes approximation.
        
        Args:
            option_price: Current option price
            stock_price: Current stock price
            strike_price: Option strike price
            time_to_expiry: Time to expiry in years
            option_type: 'call' or 'put'
            
        Returns:
            Implied volatility
        """
        try:
            # Simplified implied volatility calculation
            # In practice, you'd use a more sophisticated method
            moneyness = stock_price / strike_price
            time_factor = np.sqrt(time_to_expiry)
            
            if option_type.lower() == 'call':
                if moneyness > 1:  # In-the-money
                    iv = (option_price / stock_price) / time_factor
                else:  # Out-of-the-money
                    iv = (option_price / strike_price) / time_factor
            else:  # Put
                if moneyness < 1:  # In-the-money
                    iv = (option_price / strike_price) / time_factor
                else:  # Out-of-the-money
                    iv = (option_price / stock_price) / time_factor
            
            return max(0.1, min(2.0, iv))  # Clamp between 10% and 200%
            
        except Exception as e:
            logger.error(f"Error calculating implied volatility: {e}")
            return 0.3  # Default 30% volatility

class CoveredCallStrategy(OptionsStrategy):
    """Implements covered call strategy for income generation."""
    
    def __init__(self, 
                 delta_threshold: float = 0.3,
                 min_premium_ratio: float = 0.02):
        """
        Initialize covered call strategy.
        
        Args:
            delta_threshold: Maximum delta for covered calls (0.3 = 30 delta)
            min_premium_ratio: Minimum premium as ratio of stock price
        """
        super().__init__()
        self.delta_threshold = delta_threshold
        self.min_premium_ratio = min_premium_ratio
        
    def should_sell_covered_call(self,
                               stock_price: float,
                               rsi: float,
                               implied_volatility: float,
                               dividend_yield: float) -> bool:
        """
        Determine if covered call should be sold.
        
        Args:
            stock_price: Current stock price
            rsi: Current RSI value
            implied_volatility: Current implied volatility
            dividend_yield: Current dividend yield
            
        Returns:
            True if covered call should be sold
        """
        # Sell covered calls when:
        # 1. RSI is high (overbought)
        # 2. Implied volatility is high
        # 3. Dividend yield is low (less downside from missing dividends)
        
        rsi_condition = rsi > 70
        vol_condition = implied_volatility > 0.25  # 25% volatility
        dividend_condition = dividend_yield < 0.03  # 3% dividend yield
        
        return rsi_condition and (vol_condition or dividend_condition)
    
    def calculate_covered_call_premium(self,
                                     stock_price: float,
                                     strike_price: float,
                                     time_to_expiry: float,
                                     implied_volatility: float) -> float:
        """
        Calculate covered call premium.
        
        Args:
            stock_price: Current stock price
            strike_price: Strike price
            time_to_expiry: Time to expiry in years
            implied_volatility: Implied volatility
            
        Returns:
            Premium amount
        """
        try:
            # Simplified Black-Scholes for call option
            d1 = (np.log(stock_price / strike_price) + 
                  (self.risk_free_rate + 0.5 * implied_volatility**2) * time_to_expiry) / \
                 (implied_volatility * np.sqrt(time_to_expiry))
            
            d2 = d1 - implied_volatility * np.sqrt(time_to_expiry)
            
            # Call option price
            call_price = (stock_price * self._normal_cdf(d1) - 
                         strike_price * np.exp(-self.risk_free_rate * time_to_expiry) * 
                         self._normal_cdf(d2))
            
            return call_price
            
        except Exception as e:
            logger.error(f"Error calculating covered call premium: {e}")
            return stock_price * 0.05  # 5% of stock price as fallback
    
    def _normal_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function."""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))

class BullPutSpreadStrategy(OptionsStrategy):
    """Implements bull put spread strategy for defined risk bullish positions."""
    
    def __init__(self, 
                 max_risk_per_trade: float = 0.02,
                 min_reward_risk_ratio: float = 2.0):
        """
        Initialize bull put spread strategy.
        
        Args:
            max_risk_per_trade: Maximum risk as percentage of portfolio
            min_reward_risk_ratio: Minimum reward-to-risk ratio
        """
        super().__init__()
        self.max_risk_per_trade = max_risk_per_trade
        self.min_reward_risk_ratio = min_reward_risk_ratio
        
    def should_buy_bull_put_spread(self,
                                  stock_price: float,
                                  rsi: float,
                                  implied_volatility: float,
                                  support_level: float) -> bool:
        """
        Determine if bull put spread should be bought.
        
        Args:
            stock_price: Current stock price
            rsi: Current RSI value
            implied_volatility: Current implied volatility
            support_level: Technical support level
            
        Returns:
            True if bull put spread should be bought
        """
        # Buy bull put spreads when:
        # 1. RSI is low (oversold)
        # 2. Stock is near support
        # 3. Implied volatility is high (cheaper puts)
        
        rsi_condition = rsi < 30
        support_condition = stock_price <= support_level * 1.05  # Within 5% of support
        vol_condition = implied_volatility > 0.20  # 20% volatility
        
        return rsi_condition and support_condition and vol_condition
    
    def calculate_bull_put_spread_risk_reward(self,
                                            stock_price: float,
                                            short_strike: float,
                                            long_strike: float,
                                            time_to_expiry: float,
                                            implied_volatility: float) -> Dict[str, float]:
        """
        Calculate bull put spread risk and reward.
        
        Args:
            stock_price: Current stock price
            short_strike: Short put strike price
            long_strike: Long put strike price
            time_to_expiry: Time to expiry in years
            implied_volatility: Implied volatility
            
        Returns:
            Dictionary with risk, reward, and ratio
        """
        try:
            # Calculate put prices
            short_put_price = self._calculate_put_price(stock_price, short_strike, 
                                                       time_to_expiry, implied_volatility)
            long_put_price = self._calculate_put_price(stock_price, long_strike, 
                                                     time_to_expiry, implied_volatility)
            
            # Risk is the net debit
            risk = short_put_price - long_put_price
            
            # Reward is the difference in strikes minus the net debit
            reward = (short_strike - long_strike) - risk
            
            ratio = reward / risk if risk > 0 else 0
            
            return {
                'risk': risk,
                'reward': reward,
                'ratio': ratio,
                'short_put_price': short_put_price,
                'long_put_price': long_put_price
            }
            
        except Exception as e:
            logger.error(f"Error calculating bull put spread risk/reward: {e}")
            return {'risk': 0, 'reward': 0, 'ratio': 0}
    
    def _calculate_put_price(self,
                           stock_price: float,
                           strike_price: float,
                           time_to_expiry: float,
                           implied_volatility: float) -> float:
        """Calculate put option price using Black-Scholes."""
        try:
            d1 = (np.log(stock_price / strike_price) + 
                  (self.risk_free_rate + 0.5 * implied_volatility**2) * time_to_expiry) / \
                 (implied_volatility * np.sqrt(time_to_expiry))
            
            d2 = d1 - implied_volatility * np.sqrt(time_to_expiry)
            
            # Put option price
            put_price = (strike_price * np.exp(-self.risk_free_rate * time_to_expiry) * 
                        self._normal_cdf(-d2) - stock_price * self._normal_cdf(-d1))
            
            return put_price
            
        except Exception as e:
            logger.error(f"Error calculating put price: {e}")
            return max(0, strike_price - stock_price)  # Intrinsic value as fallback
    
    def _normal_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function."""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))

class OptionsPortfolioManager:
    """Manages options positions and risk."""
    
    def __init__(self, max_options_allocation: float = 0.20):
        """
        Initialize options portfolio manager.
        
        Args:
            max_options_allocation: Maximum allocation to options (20%)
        """
        self.max_options_allocation = max_options_allocation
        self.covered_call_strategy = CoveredCallStrategy()
        self.bull_put_spread_strategy = BullPutSpreadStrategy()
        self.positions = {}
        
    def generate_options_signals(self,
                               stock_data: Dict[str, Dict],
                               portfolio_value: float) -> List[Dict]:
        """
        Generate options trading signals.
        
        Args:
            stock_data: Dictionary with stock data including price, RSI, etc.
            portfolio_value: Total portfolio value
            
        Returns:
            List of options trading signals
        """
        signals = []
        
        for symbol, data in stock_data.items():
            stock_price = data.get('price', 0)
            rsi = data.get('rsi', 50)
            implied_vol = data.get('implied_volatility', 0.3)
            dividend_yield = data.get('dividend_yield', 0)
            support_level = data.get('support_level', stock_price * 0.9)
            
            # Check for covered call opportunities
            if self.covered_call_strategy.should_sell_covered_call(
                stock_price, rsi, implied_vol, dividend_yield):
                
                # Calculate strike price (slightly out-of-the-money)
                strike_price = stock_price * 1.05
                premium = self.covered_call_strategy.calculate_covered_call_premium(
                    stock_price, strike_price, 0.25, implied_vol)  # 3 months
                
                signals.append({
                    'symbol': symbol,
                    'strategy': 'covered_call',
                    'action': 'SELL',
                    'strike_price': strike_price,
                    'premium': premium,
                    'expiry': '3M',
                    'reason': f'RSI={rsi:.1f}, IV={implied_vol:.1%}'
                })
            
            # Check for bull put spread opportunities
            elif self.bull_put_spread_strategy.should_buy_bull_put_spread(
                stock_price, rsi, implied_vol, support_level):
                
                # Calculate strikes
                short_strike = stock_price * 0.95  # 5% below current price
                long_strike = stock_price * 0.90   # 10% below current price
                
                risk_reward = self.bull_put_spread_strategy.calculate_bull_put_spread_risk_reward(
                    stock_price, short_strike, long_strike, 0.25, implied_vol)
                
                if risk_reward['ratio'] >= self.bull_put_spread_strategy.min_reward_risk_ratio:
                    signals.append({
                        'symbol': symbol,
                        'strategy': 'bull_put_spread',
                        'action': 'BUY',
                        'short_strike': short_strike,
                        'long_strike': long_strike,
                        'risk': risk_reward['risk'],
                        'reward': risk_reward['reward'],
                        'ratio': risk_reward['ratio'],
                        'expiry': '3M',
                        'reason': f'RSI={rsi:.1f}, Support={support_level:.1f}'
                    })
        
        return signals
    
    def calculate_portfolio_options_risk(self) -> Dict[str, float]:
        """Calculate current options portfolio risk metrics."""
        total_options_value = sum(pos.get('value', 0) for pos in self.positions.values())
        max_loss = sum(pos.get('max_loss', 0) for pos in self.positions.values())
        
        return {
            'total_options_value': total_options_value,
            'max_loss': max_loss,
            'options_allocation': total_options_value / 100000 if total_options_value > 0 else 0,
            'position_count': len(self.positions)
        }

def test_options_strategies():
    """Test the options strategies."""
    print("Testing Options Strategies:")
    print("=" * 40)
    
    # Test covered call strategy
    cc_strategy = CoveredCallStrategy()
    
    # Test conditions
    should_sell_cc = cc_strategy.should_sell_covered_call(
        stock_price=100,
        rsi=75,
        implied_volatility=0.3,
        dividend_yield=0.02
    )
    
    print(f"Should Sell Covered Call: {should_sell_cc}")
    
    # Test bull put spread strategy
    bps_strategy = BullPutSpreadStrategy()
    
    should_buy_bps = bps_strategy.should_buy_bull_put_spread(
        stock_price=100,
        rsi=25,
        implied_volatility=0.25,
        support_level=95
    )
    
    print(f"Should Buy Bull Put Spread: {should_buy_bps}")
    
    # Test risk/reward calculation
    risk_reward = bps_strategy.calculate_bull_put_spread_risk_reward(
        stock_price=100,
        short_strike=95,
        long_strike=90,
        time_to_expiry=0.25,
        implied_volatility=0.25
    )
    
    print(f"Risk: ${risk_reward['risk']:.2f}")
    print(f"Reward: ${risk_reward['reward']:.2f}")
    print(f"Ratio: {risk_reward['ratio']:.2f}")

if __name__ == "__main__":
    test_options_strategies() 