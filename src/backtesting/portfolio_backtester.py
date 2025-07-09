"""
Portfolio Backtester
Comprehensive backtesting engine for dividend aristocrat + high-growth portfolio strategy.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

from src.data.fundamental_analyzer import FundamentalAnalyzer
from src.data.dividend_aristocrat_analyzer import DividendAristocratAnalyzer
from src.strategies.portfolio_strategy import PortfolioStrategy
from src.backtesting.analyzers import calculate_rsi, calculate_sharpe_ratio, calculate_max_drawdown

logger = logging.getLogger(__name__)

class TradeType(Enum):
    """Types of trades."""
    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"
    REBALANCE = "rebalance"

@dataclass
class Trade:
    """Trade record."""
    timestamp: datetime
    symbol: str
    trade_type: TradeType
    shares: float
    price: float
    value: float
    rsi: Optional[float] = None
    notes: str = ""

@dataclass
class PortfolioPosition:
    """Portfolio position."""
    symbol: str
    shares: float
    avg_price: float
    current_value: float
    weight: float
    category: str  # 'dividend_aristocrat' or 'high_growth_unprofitable'

class PortfolioBacktester:
    """Comprehensive portfolio backtester."""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, PortfolioPosition] = {}
        self.trades: List[Trade] = []
        self.cash_history: List[float] = []
        self.portfolio_value_history: List[float] = []
        self.spy_history: List[float] = []
        
        # Initialize analyzers
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.dividend_analyzer = DividendAristocratAnalyzer()
        self.portfolio_strategy = PortfolioStrategy()
        
        # Strategy parameters
        self.dividend_alloc = 0.70
        self.growth_alloc = 0.30
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
    def get_operating_income_growth(self, symbol: str, lookback_years: int = 1) -> float:
        """Get year-over-year operating income growth."""
        try:
            ticker = yf.Ticker(symbol)
            income_stmt = ticker.income_stmt
            
            if income_stmt.empty or 'Operating Income' not in income_stmt.index:
                return 0.0
            
            operating_income = income_stmt.loc['Operating Income']
            
            if len(operating_income) < 4:  # Need at least 4 quarters
                return 0.0
            
            # Get most recent quarter vs same quarter last year
            current = operating_income.iloc[0]
            previous_year = operating_income.iloc[4] if len(operating_income) >= 4 else operating_income.iloc[0]
            
            if previous_year == 0:
                return 0.0
            
            growth_rate = ((current - previous_year) / abs(previous_year)) * 100
            return growth_rate
            
        except Exception as e:
            logger.error(f"Error getting operating income growth for {symbol}: {e}")
            return 0.0
    
    def get_eligible_companies(self) -> Tuple[List[Dict], List[Dict]]:
        """Get eligible dividend aristocrats and high-growth stocks."""
        # Get dividend aristocrats from our scan results (focus on well-established companies)
        dividend_aristocrats = [
            "KO", "PG", "JNJ", "PEP", "MMM", "T", "WMT", "JPM", "BAC", "CVX",
            "XOM", "IBM", "CAT", "DUK", "USB", "LOW", "TGT", "SO", "SYK", "INTU",
            "AVGO", "QCOM", "TXN", "NUE", "FDS", "ETN", "AOS", "UPS", "DOV", "WBA",
            "DE", "ESS", "O", "AWR", "FRT", "MKC", "EV", "COST", "HRL", "GPC",
            "SHW", "BEN", "BF-B", "PBCT", "PPG", "EMR", "CINF", "MCD", "SWK"
        ]
        
        # Get high-growth unprofitable stocks (focus on companies that existed in 2014)
        high_growth_stocks = ["UBER", "LYFT", "AFRM", "ZS"]  # Removed RIVN, PLTR as they weren't public in 2014
        
        # Get operating income growth for dividend aristocrats
        dividend_data = []
        for symbol in dividend_aristocrats[:30]:  # Limit to top 30 for performance
            growth = self.get_operating_income_growth(symbol)
            dividend_data.append({
                'symbol': symbol,
                'operating_income_growth': growth
            })
        
        # Get high-growth stock data
        growth_data = []
        for symbol in high_growth_stocks:
            growth_data.append({
                'symbol': symbol,
                'operating_income_growth': 0  # Not used for equal weighting
            })
        
        return dividend_data, growth_data
    
    def get_portfolio_allocation(self, dividend_data: List[Dict], 
                               growth_data: List[Dict]) -> Dict:
        """Get current portfolio allocation."""
        return self.portfolio_strategy.get_latest_portfolio_allocation(
            dividend_data, growth_data
        )
    
    def calculate_rsi_signals(self, symbol: str, date: datetime) -> Dict:
        """Calculate RSI signals for a stock on a given date."""
        try:
            # Get historical data up to the date
            ticker = yf.Ticker(symbol)
            end_date = date.strftime('%Y-%m-%d')
            start_date = (date - timedelta(days=100)).strftime('%Y-%m-%d')
            
            data = ticker.history(start=start_date, end=end_date)
            
            if len(data) < 14:
                return {'rsi': None, 'signal': 'hold'}
            
            # Calculate RSI
            rsi = calculate_rsi(data['Close'], period=14)
            current_rsi = rsi.iloc[-1]
            
            # Generate signals
            if current_rsi < self.rsi_oversold:
                signal = 'buy'
            elif current_rsi > self.rsi_overbought:
                signal = 'sell'
            else:
                signal = 'hold'
            
            return {
                'rsi': current_rsi,
                'signal': signal
            }
            
        except Exception as e:
            logger.error(f"Error calculating RSI for {symbol}: {e}")
            return {'rsi': None, 'signal': 'hold'}
    
    def execute_trade(self, timestamp: datetime, symbol: str, trade_type: TradeType,
                     shares: float, price: float, rsi: Optional[float] = None,
                     notes: str = "") -> None:
        """Execute a trade and update portfolio."""
        value = shares * price
        
        # Record the trade
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            trade_type=trade_type,
            shares=shares,
            price=price,
            value=value,
            rsi=rsi,
            notes=notes
        )
        self.trades.append(trade)
        
        # Update portfolio
        if trade_type == TradeType.BUY:
            if symbol in self.positions:
                # Add to existing position
                pos = self.positions[symbol]
                total_shares = pos.shares + shares
                total_cost = (pos.shares * pos.avg_price) + value
                pos.shares = total_shares
                pos.avg_price = total_cost / total_shares
                pos.current_value = total_shares * price
            else:
                # Create new position
                self.positions[symbol] = PortfolioPosition(
                    symbol=symbol,
                    shares=shares,
                    avg_price=price,
                    current_value=value,
                    weight=0.0,
                    category='unknown'
                )
            self.current_capital -= value
            
        elif trade_type == TradeType.SELL:
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.shares -= shares
                pos.current_value = pos.shares * price
                if pos.shares <= 0:
                    del self.positions[symbol]
            self.current_capital += value
            
        elif trade_type == TradeType.DIVIDEND:
            # Reinvest dividend in the same stock
            if symbol in self.positions:
                pos = self.positions[symbol]
                new_shares = value / price
                pos.shares += new_shares
                pos.current_value = pos.shares * price
                # Update average price
                total_cost = (pos.shares * pos.avg_price) + value
                pos.avg_price = total_cost / pos.shares
    
    def rebalance_portfolio(self, date: datetime, allocation: Dict) -> None:
        """Rebalance portfolio according to target allocation."""
        total_value = self.current_capital + sum(pos.current_value for pos in self.positions.values())
        
        # Calculate target positions
        target_positions = {}
        for category, allocations in allocation.items():
            if category == 'total_allocation':
                continue
            for symbol, target_weight in allocations.items():
                target_value = total_value * target_weight
                target_positions[symbol] = target_value
        
        # Execute rebalancing trades
        for symbol, target_value in target_positions.items():
            current_value = 0
            if symbol in self.positions:
                current_value = self.positions[symbol].current_value
            
            if target_value > current_value:
                # Need to buy
                shares_to_buy = (target_value - current_value) / self.get_stock_price(symbol, date)
                if shares_to_buy > 0:
                    price = self.get_stock_price(symbol, date)
                    self.execute_trade(date, symbol, TradeType.BUY, shares_to_buy, price,
                                     notes="Rebalancing")
                    
            elif target_value < current_value:
                # Need to sell
                shares_to_sell = (current_value - target_value) / self.get_stock_price(symbol, date)
                if shares_to_sell > 0:
                    price = self.get_stock_price(symbol, date)
                    self.execute_trade(date, symbol, TradeType.SELL, shares_to_sell, price,
                                     notes="Rebalancing")
    
    def get_stock_price(self, symbol: str, date: datetime) -> float:
        """Get stock price on a specific date."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=date, end=date + timedelta(days=1))
            if not data.empty:
                return data['Close'].iloc[0]
            else:
                # Try to get the closest available price
                data = ticker.history(start=date - timedelta(days=5), end=date + timedelta(days=5))
                if not data.empty:
                    return data['Close'].iloc[-1]
                else:
                    return 100.0  # Default price if data unavailable
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 100.0
    
    def process_dividends(self, date: datetime) -> None:
        """Process dividends and reinvest them."""
        for symbol, position in self.positions.items():
            if position.category == 'dividend_aristocrat':
                try:
                    ticker = yf.Ticker(symbol)
                    dividends = ticker.dividends
                    
                    # Check if there's a dividend on this date
                    if not dividends.empty:
                        dividend_date = pd.to_datetime(date).normalize()
                        if dividend_date in dividends.index:
                            dividend_amount = dividends[dividend_date]
                            dividend_value = position.shares * dividend_amount
                            
                            # Reinvest dividend
                            price = self.get_stock_price(symbol, date)
                            if price > 0:
                                self.execute_trade(date, symbol, TradeType.DIVIDEND, 
                                                 dividend_value / price, price,
                                                 notes=f"Dividend reinvestment: ${dividend_amount:.2f}")
                                
                except Exception as e:
                    logger.error(f"Error processing dividend for {symbol}: {e}")
    
    def run_backtest(self, start_date: str = "2014-01-01", 
                    end_date: str = "2024-12-31") -> Dict:
        """Run the complete backtest."""
        logger.info("🚀 Starting Portfolio Backtest")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        
        # Get eligible companies
        dividend_data, growth_data = self.get_eligible_companies()
        logger.info(f"Dividend Aristocrats: {len(dividend_data)}")
        logger.info(f"High-Growth Stocks: {len(growth_data)}")
        
        # Get initial allocation
        allocation = self.get_portfolio_allocation(dividend_data, growth_data)
        
        # Get SPY data for comparison
        spy_ticker = yf.Ticker("SPY")
        spy_data = spy_ticker.history(start=start_date, end=end_date)
        
        if spy_data.empty:
            logger.error("❌ No SPY data available for the specified period")
            return {}
        
        spy_initial_price = spy_data['Close'].iloc[0]
        
        # Initialize tracking
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Create monthly date range
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='MS')  # Month Start
        
        # Track portfolio value over time
        portfolio_values = []
        spy_values = []
        dates = []
        
        # Initial portfolio setup
        total_value = self.initial_capital
        for category, allocations in allocation.items():
            if category == 'total_allocation':
                continue
            for symbol, target_weight in allocations.items():
                target_value = total_value * target_weight
                price = self.get_stock_price(symbol, start_dt)
                if price > 0:
                    shares = target_value / price
                    self.execute_trade(start_dt, symbol, TradeType.BUY, shares, price,
                                     notes="Initial allocation")
        
        # Monthly rebalancing
        for current_date in date_range:
            try:
                # Check if it's a valid trading day (not weekend)
                if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                    # Rebalance portfolio
                    self.rebalance_portfolio(current_date, allocation)
                    
                    # Process dividends
                    self.process_dividends(current_date)
                    
                    # Calculate current portfolio value
                    portfolio_value = self.current_capital
                    for symbol, position in self.positions.items():
                        price = self.get_stock_price(symbol, current_date)
                        if price > 0:
                            position.current_value = position.shares * price
                            portfolio_value += position.current_value
                    
                    # Get SPY value for this date
                    spy_price = spy_data.loc[current_date, 'Close'] if current_date in spy_data.index else spy_initial_price
                    spy_value = (self.initial_capital / spy_initial_price) * spy_price
                    
                    # Record values
                    portfolio_values.append(portfolio_value)
                    spy_values.append(spy_value)
                    dates.append(current_date)
                    
                    logger.info(f"📊 {current_date.strftime('%Y-%m')}: Portfolio=${portfolio_value:,.0f}, SPY=${spy_value:,.0f}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {current_date}: {e}")
                continue
        
        if not portfolio_values:
            logger.error("❌ No portfolio values calculated")
            return {}
        
        # Calculate performance metrics
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
        spy_returns = pd.Series(spy_values).pct_change().dropna()
        
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        spy_total_return = (spy_values[-1] - self.initial_capital) / self.initial_capital
        
        sharpe_ratio = calculate_sharpe_ratio(portfolio_returns)
        max_drawdown = calculate_max_drawdown(portfolio_values)
        
        results = {
            'initial_capital': self.initial_capital,
            'final_portfolio_value': portfolio_values[-1],
            'final_spy_value': spy_values[-1],
            'total_return': total_return,
            'spy_total_return': spy_total_return,
            'excess_return': total_return - spy_total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': portfolio_returns.std() * np.sqrt(12),  # Annualized
            'total_trades': len(self.trades),
            'portfolio_values': portfolio_values,
            'spy_values': spy_values,
            'dates': dates,
            'trades': self.trades
        }
        
        logger.info("✅ Backtest completed successfully!")
        logger.info(f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
        logger.info(f"Final SPY Value: ${spy_values[-1]:,.2f}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"SPY Return: {spy_total_return:.2%}")
        logger.info(f"Excess Return: {total_return - spy_total_return:.2%}")
        
        return results 