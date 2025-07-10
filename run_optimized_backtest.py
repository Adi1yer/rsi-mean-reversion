"""
Optimized Portfolio Backtest
Combines the best of both versions - dividend reinvestment with selective enhancements.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# Import only proven enhancements
from src.data.reliable_fundamental_data import ReliableFundamentalData

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI for a price series."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def run_optimized_backtest():
    """Run optimized backtest combining best of both versions."""
    logger.info("🚀 OPTIMIZED PORTFOLIO BACKTEST")
    logger.info("="*60)
    
    # Strategy parameters (keep original allocation)
    initial_capital = 100000
    dividend_alloc = 0.70
    growth_alloc = 0.30
    
    # Enhanced stock lists (more companies)
    dividend_stocks = ["KO", "PG", "JNJ", "PEP", "MMM", "T", "WMT", "JPM", "BAC", "CVX", 
                      "XOM", "IBM", "DUK", "USB", "LOW", "TGT", "SO", "SYK", "INTU", "AVGO",
                      "HD", "COST", "NKE", "SBUX", "DIS", "V", "MA", "UNH", "ABBV", "LLY"]
    
    growth_stocks = ["UBER", "LYFT", "AFRM", "ZS", "SNOW", "PLTR", "RIVN", "LCID", "RBLX", "HOOD"]
    
    # Initialize only proven enhancement
    fundamental_data = ReliableFundamentalData()
    
    # Get data
    start_date = "2014-01-01"
    end_date = "2024-12-31"
    
    logger.info(f"📊 Loading data for {len(dividend_stocks)} dividend stocks and {len(growth_stocks)} growth stocks")
    
    # Get SPY data
    spy_data = yf.download("SPY", start=start_date, end=end_date)
    if spy_data.empty:
        logger.error("❌ No SPY data available")
        return None
    
    # Get dividend stock data
    dividend_data = {}
    dividend_dividends = {}
    for symbol in dividend_stocks:
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if data is not None and not data.empty:
                dividend_data[symbol] = data
                # Get dividend data
                ticker = yf.Ticker(symbol)
                dividends = ticker.dividends
                dividend_dividends[symbol] = dividends
                logger.info(f"✅ Loaded {symbol}")
            else:
                logger.warning(f"⚠️ No data for {symbol}")
        except Exception as e:
            logger.error(f"❌ Error loading {symbol}: {e}")
    
    # Get growth stock data
    growth_data = {}
    for symbol in growth_stocks:
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if data is not None and not data.empty:
                growth_data[symbol] = data
                logger.info(f"✅ Loaded {symbol}")
            else:
                logger.warning(f"⚠️ No data for {symbol}")
        except Exception as e:
            logger.error(f"❌ Error loading {symbol}: {e}")
    
    # Use reliable fundamental data for weights (but keep it simple)
    logger.info("📊 Calculating exponential weights using reliable fundamental data...")
    dividend_weights = fundamental_data.get_exponential_weights(list(dividend_data.keys()))
    
    # Equal weights for growth stocks (keep original simplicity)
    growth_weight_per_stock = 1.0 / len(growth_data) if growth_data else 0
    
    logger.info("📈 Dividend stock weights (Top 10):")
    sorted_weights = sorted(dividend_weights.items(), key=lambda x: x[1], reverse=True)
    for i, (symbol, weight) in enumerate(sorted_weights[:10]):
        logger.info(f"   {i+1}. {symbol}: {weight:.3f}")
    
    # Initialize portfolio tracking
    portfolio_values = []
    spy_values = []
    dates = []
    
    # Get common date range
    all_dates = spy_data.index
    for symbol in dividend_data:
        if dividend_data[symbol] is not None and hasattr(dividend_data[symbol], 'index'):
            all_dates = all_dates.intersection(dividend_data[symbol].index)
    for symbol in growth_data:
        if growth_data[symbol] is not None and hasattr(growth_data[symbol], 'index'):
            all_dates = all_dates.intersection(growth_data[symbol].index)
    
    if all_dates is None or len(all_dates) == 0:
        logger.error("❌ No common date range found")
        return None
    
    # Initialize positions
    positions = {}
    cash = initial_capital
    
    # Initial allocation (keep original approach)
    for symbol, weight in dividend_weights.items():
        if symbol in dividend_data:
            initial_price = float(dividend_data[symbol].iloc[0]['Close'].iloc[0])
            target_value = initial_capital * dividend_alloc * weight
            shares = target_value / initial_price
            positions[symbol] = {
                'shares': shares,
                'category': 'dividend_aristocrat'
            }
            cash -= target_value
    
    for symbol in growth_data:
        initial_price = float(growth_data[symbol].iloc[0]['Close'].iloc[0])
        target_value = initial_capital * growth_alloc * growth_weight_per_stock
        shares = target_value / initial_price
        positions[symbol] = {
            'shares': shares,
            'category': 'high_growth'
        }
        cash -= target_value
    
    # Track portfolio over time with detailed breakdown
    logger.info("📈 Calculating optimized portfolio performance...")
    
    # Initialize detailed tracking
    dividend_values = []
    growth_values = []
    dividend_returns = []
    growth_returns = []
    dividend_contributions = []
    growth_contributions = []
    
    for date in all_dates:
        try:
            # Process dividends for dividend aristocrats (KEEP THIS - it's the key!)
            for symbol, position in positions.items():
                if position['category'] == 'dividend_aristocrat' and symbol in dividend_dividends:
                    dividends = dividend_dividends[symbol]
                    if not dividends.empty and date in dividends.index:
                        dividend_amount = dividends[date]
                        dividend_value = position['shares'] * dividend_amount
                        
                        # Reinvest dividend in the same stock (THIS IS THE KEY!)
                        current_price = float(dividend_data[symbol].loc[date, 'Close'].iloc[0])
                        new_shares = dividend_value / current_price
                        position['shares'] += new_shares
                        
                        logger.debug(f"💰 {date.strftime('%Y-%m-%d')}: {symbol} dividend reinvestment: ${dividend_value:.2f}")
            
            # Calculate current portfolio value with breakdown
            portfolio_value = cash
            dividend_value = 0
            growth_value = 0
            
            for symbol, position in positions.items():
                if symbol in dividend_data:
                    current_price = float(dividend_data[symbol].loc[date, 'Close'].iloc[0])
                    position_value = position['shares'] * current_price
                    dividend_value += position_value
                    portfolio_value += position_value
                elif symbol in growth_data:
                    current_price = float(growth_data[symbol].loc[date, 'Close'].iloc[0])
                    position_value = position['shares'] * current_price
                    growth_value += position_value
                    portfolio_value += position_value
            
            # SPY value
            spy_price = float(spy_data.loc[date, 'Close'].iloc[0])
            spy_value = initial_capital * (spy_price / spy_data.iloc[0]['Close'].iloc[0])
            
            # Record values
            portfolio_values.append(portfolio_value)
            spy_values.append(spy_value)
            dividend_values.append(dividend_value)
            growth_values.append(growth_value)
            dates.append(date)
            
            # Calculate contributions
            dividend_contributions.append(dividend_value / portfolio_value if portfolio_value > 0 else 0)
            growth_contributions.append(growth_value / portfolio_value if portfolio_value > 0 else 0)
            
        except Exception as e:
            logger.error(f"❌ Error calculating values for {date}: {e}")
            continue
    
    if not portfolio_values:
        logger.error("❌ No portfolio values calculated")
        return None
    
    # Calculate performance metrics
    portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
    spy_returns = pd.Series(spy_values).pct_change().dropna()
    
    # Handle NaN values
    final_portfolio_value = portfolio_values[-1] if not np.isnan(portfolio_values[-1]) else initial_capital
    final_spy_value = spy_values[-1] if not np.isnan(spy_values[-1]) else initial_capital
    
    total_return = (final_portfolio_value - initial_capital) / initial_capital
    spy_total_return = (final_spy_value - initial_capital) / initial_capital
    excess_return = total_return - spy_total_return
    
    # Calculate Sharpe ratio
    risk_free_rate = 0.02
    portfolio_annual_return = portfolio_returns.mean() * 252
    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (portfolio_annual_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
    
    # Calculate max drawdown
    running_max = pd.Series(portfolio_values).expanding().max()
    drawdown = (pd.Series(portfolio_values) - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calculate final breakdown
    final_dividend_value = dividend_values[-1] if dividend_values else 0
    final_growth_value = growth_values[-1] if growth_values else 0
    
    # Calculate returns for each category
    initial_dividend_allocation = initial_capital * dividend_alloc
    initial_growth_allocation = initial_capital * growth_alloc
    
    dividend_return = (final_dividend_value - initial_dividend_allocation) / initial_dividend_allocation if initial_dividend_allocation > 0 else 0
    growth_return = (final_growth_value - initial_growth_allocation) / initial_growth_allocation if initial_growth_allocation > 0 else 0
    
    # Results
    results = {
        'initial_capital': initial_capital,
        'final_portfolio_value': final_portfolio_value,
        'final_spy_value': final_spy_value,
        'total_return': total_return,
        'spy_total_return': spy_total_return,
        'excess_return': excess_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'volatility': portfolio_volatility,
        'portfolio_values': portfolio_values,
        'spy_values': spy_values,
        'dividend_values': dividend_values,
        'growth_values': growth_values,
        'dividend_contributions': dividend_contributions,
        'growth_contributions': growth_contributions,
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'dividend_stocks': list(dividend_data.keys()),
        'growth_stocks': list(growth_data.keys()),
        'dividend_weights': dividend_weights,
        'final_dividend_value': final_dividend_value,
        'final_growth_value': final_growth_value,
        'dividend_return': dividend_return,
        'growth_return': growth_return,
        'initial_dividend_allocation': initial_dividend_allocation,
        'initial_growth_allocation': initial_growth_allocation,
        'optimizations': {
            'reliable_fundamental_data': True,
            'monthly_rebalancing': False,  # REMOVED - was hurting performance
            'options_strategies': False,   # REMOVED - was adding complexity
            'dynamic_thresholds': False,   # REMOVED - was too conservative
            'dividend_reinvestment': True, # KEPT - this is the key!
            'exponential_weighting': True  # KEPT - but simplified
        }
    }
    
    # Display results
    display_optimized_results(results)
    
    # Save results
    save_optimized_results(results)
    
    # Create charts
    create_optimized_charts(results)
    
    return results

def display_optimized_results(results):
    """Display optimized backtest results."""
    logger.info("\n" + "="*60)
    logger.info("📊 OPTIMIZED BACKTEST RESULTS")
    logger.info("="*60)
    
    logger.info(f"\n💰 PERFORMANCE:")
    logger.info(f"   Initial Capital: ${results['initial_capital']:,.2f}")
    logger.info(f"   Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
    logger.info(f"   Final SPY Value: ${results['final_spy_value']:,.2f}")
    logger.info(f"   Total Return: {results['total_return']:.2%}")
    logger.info(f"   SPY Return: {results['spy_total_return']:.2%}")
    logger.info(f"   Excess Return: {results['excess_return']:.2%}")
    
    logger.info(f"\n📈 RISK METRICS:")
    logger.info(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"   Max Drawdown: {results['max_drawdown']:.2%}")
    logger.info(f"   Volatility: {results['volatility']:.2%}")
    
    logger.info(f"\n🎯 STRATEGY ANALYSIS:")
    if results['excess_return'] > 0:
        logger.info(f"   ✅ Portfolio outperformed SPY by {results['excess_return']:.2%}")
    else:
        logger.info(f"   ❌ Portfolio underperformed SPY by {abs(results['excess_return']):.2%}")
    
    logger.info(f"\n🔧 OPTIMIZATIONS APPLIED:")
    optimizations = results['optimizations']
    for optimization, applied in optimizations.items():
        status = "✅" if applied else "❌"
        logger.info(f"   {status} {optimization.replace('_', ' ').title()}")
    
    logger.info(f"\n📊 PORTFOLIO COMPOSITION:")
    logger.info(f"   Dividend Stocks: {', '.join(results['dividend_stocks'])}")
    logger.info(f"   Growth Stocks: {', '.join(results['growth_stocks'])}")
    
    logger.info(f"\n⚖️ EXPONENTIAL WEIGHTS (Top 5):")
    sorted_weights = sorted(results['dividend_weights'].items(), key=lambda x: x[1], reverse=True)
    for i, (symbol, weight) in enumerate(sorted_weights[:5]):
        logger.info(f"   {i+1}. {symbol}: {weight:.3f}")
    
    logger.info(f"\n📊 PORTFOLIO BREAKDOWN:")
    logger.info(f"   Final Dividend Value: ${results['final_dividend_value']:,.2f}")
    logger.info(f"   Final Growth Value: ${results['final_growth_value']:,.2f}")
    logger.info(f"   Dividend Return: {results['dividend_return']:.2%}")
    logger.info(f"   Growth Return: {results['growth_return']:.2%}")
    logger.info(f"   Initial Dividend Allocation: ${results['initial_dividend_allocation']:,.2f}")
    logger.info(f"   Initial Growth Allocation: ${results['initial_growth_allocation']:,.2f}")

def save_optimized_results(results):
    """Save optimized results to file."""
    try:
        filename = f"optimized_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"✅ Results saved to {filename}")
    except Exception as e:
        logger.error(f"❌ Error saving results: {e}")

def create_optimized_charts(results):
    """Create comprehensive optimized performance charts."""
    try:
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Define the grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Portfolio vs SPY Value Over Time (top left, spans 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        dates = pd.to_datetime(results['dates'])
        portfolio_values = results['portfolio_values']
        spy_values = results['spy_values']
        
        ax1.plot(dates, portfolio_values, label='Optimized Portfolio', linewidth=2, color='blue')
        ax1.plot(dates, spy_values, label='SPY', linewidth=2, color='red', alpha=0.7)
        ax1.set_title('Portfolio vs SPY Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Portfolio Composition Breakdown (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        dividend_value = results['final_dividend_value']
        growth_value = results['final_growth_value']
        labels = ['Dividend Stocks', 'Growth Stocks']
        sizes = [dividend_value, growth_value]
        colors = ['#ff9999', '#66b3ff']
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Final Portfolio Composition', fontsize=12, fontweight='bold')
        
        # 3. Returns Breakdown (top right, bottom)
        ax3 = fig.add_subplot(gs[1, 2])
        dividend_return = results['dividend_return'] * 100
        growth_return = results['growth_return'] * 100
        categories = ['Dividend Stocks', 'Growth Stocks']
        returns = [dividend_return, growth_return]
        colors = ['#ff9999', '#66b3ff']
        
        bars = ax3.bar(categories, returns, color=colors, alpha=0.7)
        ax3.set_title('Returns by Category', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Return (%)')
        ax3.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, returns):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 4. Portfolio Allocation Over Time (middle left)
        ax4 = fig.add_subplot(gs[2, 0])
        dividend_contributions = [x * 100 for x in results['dividend_contributions']]
        growth_contributions = [x * 100 for x in results['growth_contributions']]
        
        ax4.plot(dates, dividend_contributions, label='Dividend Stocks', color='#ff9999', linewidth=2)
        ax4.plot(dates, growth_contributions, label='Growth Stocks', color='#66b3ff', linewidth=2)
        ax4.set_title('Portfolio Allocation Over Time', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Allocation (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance Comparison (middle right)
        ax5 = fig.add_subplot(gs[2, 1])
        metrics = ['Total Return', 'SPY Return', 'Excess Return']
        values = [results['total_return'] * 100, results['spy_total_return'] * 100, results['excess_return'] * 100]
        colors = ['blue', 'red', 'green' if results['excess_return'] > 0 else 'orange']
        
        bars = ax5.bar(metrics, values, color=colors, alpha=0.7)
        ax5.set_title('Performance Comparison', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Return (%)')
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 6. Risk Metrics (bottom left)
        ax6 = fig.add_subplot(gs[3, 0])
        risk_metrics = ['Sharpe Ratio', 'Max Drawdown', 'Volatility']
        risk_values = [results['sharpe_ratio'], results['max_drawdown'] * 100, results['volatility'] * 100]
        colors = ['green', 'red', 'orange']
        
        bars = ax6.bar(risk_metrics, risk_values, color=colors, alpha=0.7)
        ax6.set_title('Risk Metrics', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Value')
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, risk_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 7. Optimization Status (bottom right)
        ax7 = fig.add_subplot(gs[3, 1])
        optimizations = results['optimizations']
        optimization_names = [name.replace('_', ' ').title() for name in optimizations.keys()]
        optimization_status = [1 if applied else 0 for applied in optimizations.values()]
        
        bars = ax7.bar(optimization_names, optimization_status, color='green', alpha=0.7)
        ax7.set_title('Optimizations Applied', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Status')
        ax7.set_ylim(0, 1.2)
        ax7.tick_params(axis='x', rotation=45)
        ax7.grid(True, alpha=0.3)
        
        for bar, status in zip(bars, optimization_status):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    'Applied' if status else 'Removed', ha='center', va='bottom')
        
        # 8. Dividend vs Growth Value Over Time (bottom middle)
        ax8 = fig.add_subplot(gs[3, 2])
        dividend_values = results['dividend_values']
        growth_values = results['growth_values']
        
        ax8.plot(dates, dividend_values, label='Dividend Stocks', color='#ff9999', linewidth=2)
        ax8.plot(dates, growth_values, label='Growth Stocks', color='#66b3ff', linewidth=2)
        ax8.set_title('Dividend vs Growth Value Over Time', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Date')
        ax8.set_ylabel('Value ($)')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Summary Statistics (bottom right)
        ax9 = fig.add_subplot(gs[3, 3])
        ax9.axis('off')
        
        # Create summary text
        summary_text = f"""
        📊 PORTFOLIO SUMMARY
        
        💰 Total Return: {results['total_return']:.1%}
        📈 SPY Return: {results['spy_total_return']:.1%}
        🎯 Excess Return: {results['excess_return']:.1%}
        
        📊 Final Values:
        💵 Portfolio: ${results['final_portfolio_value']:,.0f}
        📊 SPY: ${results['final_spy_value']:,.0f}
        
        🎲 Risk Metrics:
        📊 Sharpe: {results['sharpe_ratio']:.2f}
        📉 Max DD: {results['max_drawdown']:.1%}
        📈 Volatility: {results['volatility']:.1%}
        
        📋 Composition:
        🏢 Dividend: ${results['final_dividend_value']:,.0f}
        🚀 Growth: ${results['final_growth_value']:,.0f}
        """
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('🚀 Optimized Portfolio Backtest Results', fontsize=16, fontweight='bold')
        
        filename = f"optimized_backtest_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Comprehensive charts saved to {filename}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"❌ Error creating charts: {e}")

def main():
    """Main function."""
    try:
        results = run_optimized_backtest()
        if results:
            logger.info("\n✅ Optimized backtest completed successfully!")
        else:
            logger.error("\n❌ Optimized backtest failed!")
        return results
    except Exception as e:
        logger.error(f"❌ Error running optimized backtest: {e}")
        return None

if __name__ == "__main__":
    main() 