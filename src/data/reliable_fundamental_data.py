"""
Reliable Fundamental Data Module
Provides consistent access to fundamental data for exponential weighting.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class ReliableFundamentalData:
    """Provides reliable fundamental data for exponential weighting calculations."""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
        
    def get_operating_income_growth(self, symbol: str) -> float:
        """Get reliable year-over-year operating income growth."""
        cache_key = f"{symbol}_operating_growth"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_duration:
                return cached_data['value']
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get income statement
            income_stmt = ticker.income_stmt
            if income_stmt is None or income_stmt.empty:
                logger.warning(f"No income statement data for {symbol}")
                return 0.0
            
            # Look for operating income in different possible formats
            operating_income = None
            possible_names = ['Operating Income', 'Operating Income Loss', 'Income from Operations']
            
            for name in possible_names:
                if name in income_stmt.index:
                    operating_income = income_stmt.loc[name]
                    break
            
            if operating_income is None:
                logger.warning(f"No operating income data found for {symbol}")
                return 0.0
            
            # Get quarterly data
            if len(operating_income) < 8:  # Need at least 2 years of quarterly data
                logger.warning(f"Insufficient operating income data for {symbol}")
                return 0.0
            
            # Calculate year-over-year growth
            current_year = operating_income.iloc[:4].sum()  # Last 4 quarters
            previous_year = operating_income.iloc[4:8].sum()  # Previous 4 quarters
            
            if previous_year == 0:
                logger.warning(f"Previous year operating income is zero for {symbol}")
                return 0.0
            
            growth_rate = ((current_year - previous_year) / abs(previous_year)) * 100
            
            # Cache the result
            self.cache[cache_key] = {
                'value': growth_rate,
                'timestamp': time.time()
            }
            
            return growth_rate
            
        except Exception as e:
            logger.error(f"Error getting operating income growth for {symbol}: {e}")
            return 0.0
    
    def get_revenue_growth(self, symbol: str) -> float:
        """Get reliable year-over-year revenue growth."""
        cache_key = f"{symbol}_revenue_growth"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_duration:
                return cached_data['value']
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get income statement
            income_stmt = ticker.income_stmt
            if income_stmt is None or income_stmt.empty:
                return 0.0
            
            # Look for revenue
            revenue = None
            possible_names = ['Total Revenue', 'Revenue', 'Sales Revenue']
            
            for name in possible_names:
                if name in income_stmt.index:
                    revenue = income_stmt.loc[name]
                    break
            
            if revenue is None:
                return 0.0
            
            # Get quarterly data
            if len(revenue) < 8:
                return 0.0
            
            # Calculate year-over-year growth
            current_year = revenue.iloc[:4].sum()
            previous_year = revenue.iloc[4:8].sum()
            
            if previous_year == 0:
                return 0.0
            
            growth_rate = ((current_year - previous_year) / abs(previous_year)) * 100
            
            # Cache the result
            self.cache[cache_key] = {
                'value': growth_rate,
                'timestamp': time.time()
            }
            
            return growth_rate
            
        except Exception as e:
            logger.error(f"Error getting revenue growth for {symbol}: {e}")
            return 0.0
    
    def get_free_cash_flow_growth(self, symbol: str) -> float:
        """Get reliable year-over-year free cash flow growth."""
        cache_key = f"{symbol}_fcf_growth"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_duration:
                return cached_data['value']
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get cash flow statement
            cash_flow = ticker.cashflow
            if cash_flow is None or cash_flow.empty:
                return 0.0
            
            # Look for free cash flow
            fcf = None
            possible_names = ['Free Cash Flow', 'Operating Cash Flow', 'Cash Flow From Operations']
            
            for name in possible_names:
                if name in cash_flow.index:
                    fcf = cash_flow.loc[name]
                    break
            
            if fcf is None:
                return 0.0
            
            # Get quarterly data
            if len(fcf) < 8:
                return 0.0
            
            # Calculate year-over-year growth
            current_year = fcf.iloc[:4].sum()
            previous_year = fcf.iloc[4:8].sum()
            
            if previous_year == 0:
                return 0.0
            
            growth_rate = ((current_year - previous_year) / abs(previous_year)) * 100
            
            # Cache the result
            self.cache[cache_key] = {
                'value': growth_rate,
                'timestamp': time.time()
            }
            
            return growth_rate
            
        except Exception as e:
            logger.error(f"Error getting FCF growth for {symbol}: {e}")
            return 0.0
    
    def calculate_composite_growth_score(self, symbol: str) -> float:
        """Calculate a composite growth score based on multiple metrics."""
        try:
            operating_growth = self.get_operating_income_growth(symbol)
            revenue_growth = self.get_revenue_growth(symbol)
            fcf_growth = self.get_free_cash_flow_growth(symbol)
            
            # Weight the metrics (operating income growth is most important)
            composite_score = (operating_growth * 0.5) + (revenue_growth * 0.3) + (fcf_growth * 0.2)
            
            return composite_score
            
        except Exception as e:
            logger.error(f"Error calculating composite growth score for {symbol}: {e}")
            return 0.0
    
    def get_exponential_weights(self, symbols: List[str]) -> Dict[str, float]:
        """Calculate exponential weights based on composite growth scores."""
        if not symbols:
            return {}
        
        # Get growth scores for all symbols
        growth_scores = {}
        for symbol in symbols:
            score = self.calculate_composite_growth_score(symbol)
            growth_scores[symbol] = score
            logger.info(f"{symbol}: Growth Score = {score:.2f}%")
        
        # Calculate exponential weights
        # Use exp(score / 100) to normalize and emphasize differences
        exp_weights = {}
        total_weight = 0
        
        for symbol, score in growth_scores.items():
            # Ensure positive weights even for negative growth
            exp_weight = np.exp(max(score, -50) / 100)
            exp_weights[symbol] = exp_weight
            total_weight += exp_weight
        
        # Normalize to sum to 1
        if total_weight > 0:
            normalized_weights = {}
            for symbol, weight in exp_weights.items():
                normalized_weights[symbol] = weight / total_weight
            return normalized_weights
        else:
            # Fallback to equal weights
            weight_per_symbol = 1.0 / len(symbols)
            return {symbol: weight_per_symbol for symbol in symbols}

def test_reliable_fundamental_data():
    """Test the reliable fundamental data module."""
    fundamental_data = ReliableFundamentalData()
    
    # Test symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    print("Testing Reliable Fundamental Data Module:")
    print("=" * 50)
    
    for symbol in test_symbols:
        print(f"\n{symbol}:")
        print(f"  Operating Income Growth: {fundamental_data.get_operating_income_growth(symbol):.2f}%")
        print(f"  Revenue Growth: {fundamental_data.get_revenue_growth(symbol):.2f}%")
        print(f"  FCF Growth: {fundamental_data.get_free_cash_flow_growth(symbol):.2f}%")
        print(f"  Composite Score: {fundamental_data.calculate_composite_growth_score(symbol):.2f}%")
    
    # Test exponential weights
    weights = fundamental_data.get_exponential_weights(test_symbols)
    print(f"\nExponential Weights:")
    for symbol, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {symbol}: {weight:.3f}")

if __name__ == "__main__":
    test_reliable_fundamental_data() 