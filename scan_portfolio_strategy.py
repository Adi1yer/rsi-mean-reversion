#!/usr/bin/env python3
"""
Portfolio Strategy Scanner
Implements 70% dividend aristocrats / 30% high-growth allocation with specific trading rules.
"""

import sys
import os
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.strategies.portfolio_strategy import PortfolioStrategy, PortfolioAllocation, TradingAction
from src.data.dividend_aristocrat_analyzer import DividendAristocratAnalyzer
from src.data.fundamental_analyzer import FundamentalAnalyzer

def get_comprehensive_universe():
    """Get a comprehensive universe of US stocks for portfolio analysis."""
    
    # Dividend aristocrat candidates (known dividend payers)
    dividend_candidates = [
        # S&P 500 Dividend Aristocrats (companies with 25+ years of dividend increases)
        'JNJ', 'KO', 'PG', 'MMM', 'T', 'VZ', 'XOM', 'CVX', 'JPM', 'BAC',
        'WMT', 'HD', 'COST', 'LOW', 'UPS', 'CAT', 'DE', 'EMR', 'AFL',
        'AOS', 'AWR', 'BEN', 'BF-B', 'BRO', 'CB', 'CINF', 'CTAS', 'DOV',
        'ECL', 'ESS', 'EV', 'FDS', 'FRT', 'GPC', 'HRL', 'ITW', 'LEG',
        'MCD', 'MDT', 'MKC', 'NUE', 'O', 'PBCT', 'PEP', 'PFE', 'PNR',
        'PPG', 'ROST', 'SHW', 'SJM', 'SWK', 'TGT', 'TROW', 'VFC', 'WBA',
        
        # Additional dividend payers
        'ABT', 'TMO', 'AVGO', 'ACN', 'DHR', 'LLY', 'NEE', 'TXN', 'HON',
        'UNP', 'RTX', 'INTU', 'QCOM', 'IBM', 'GS', 'MS', 'AXP', 'GE',
        'BMY', 'TFC', 'USB', 'COF', 'SCHW', 'BLK', 'AMGN', 'GILD', 'DUK',
        'SO', 'PLD', 'CCI', 'REGN', 'BDX', 'TJX', 'NOC', 'MMC', 'ETN',
        'AON', 'ICE', 'APD', 'ADI', 'KLAC', 'HUM', 'ISRG', 'SYK', 'VRTX',
        'BIIB', 'ALGN', 'DXCM', 'IDXX', 'WST', 'RMD', 'ZTS', 'ILMN'
    ]
    
    # High-growth unprofitable candidates
    growth_candidates = [
        # Known high-growth unprofitable
        'LYFT',  # Already confirmed eligible
        
        # Tech companies with high growth potential
        'SNOW', 'CRWD', 'NET', 'ZS', 'OKTA', 'TWLO', 'SQ', 'SHOP', 'ZM', 'DOCU',
        'DDOG', 'MDB', 'ESTC', 'TEAM', 'FTNT', 'CDNS', 'SNPS', 'MTCH', 'ILMN',
        'BIIB', 'ALGN', 'DXCM', 'IDXX', 'WST', 'RMD', 'ZTS', 'VRTX', 'ISRG',
        
        # EV and clean energy
        'NIO', 'XPEV', 'LI', 'LCID', 'CHPT', 'PLUG', 'FCEL', 'RIVN',
        
        # Fintech and crypto
        'COIN', 'HOOD', 'SOFI', 'AFRM', 'UPST', 'LC', 'OPRT',
        
        # E-commerce and digital
        'ETSY', 'PINS', 'SNAP', 'SPOT', 'ROKU', 'TTD', 'CRWD',
        
        # Biotech and healthcare
        'MRNA', 'BNTX', 'NVAX', 'INO', 'CRSP', 'EDIT', 'NTLA',
        
        # Cloud and SaaS
        'SNOW', 'NET', 'ZS', 'TWLO', 'SQ', 'SHOP', 'ZM', 'DOCU', 'OKTA',
        'DDOG', 'MDB', 'ESTC', 'TEAM', 'FTNT', 'CDNS', 'SNPS', 'MTCH',
        
        # Gaming and entertainment
        'ROKU', 'SPOT', 'SNAP', 'PINS', 'ETSY',
        
        # Emerging tech
        'PLTR', 'UBER', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI'
    ]
    
    return {
        'dividend_candidates': list(set(dividend_candidates)),
        'growth_candidates': list(set(growth_candidates))
    }

def analyze_portfolio_strategy():
    """Analyze portfolio strategy with 70/30 allocation."""
    logger.info("🎯 PORTFOLIO STRATEGY ANALYSIS")
    logger.info("="*60)
    
    # Get comprehensive universe
    universe = get_comprehensive_universe()
    
    # Initialize portfolio strategy
    portfolio_strategy = PortfolioStrategy(
        dividend_alloc=0.70,  # 70% dividend aristocrats
        growth_alloc=0.30,    # 30% high-growth unprofitable
        dividend_rsi_threshold=70.0,  # RSI > 70 for covered calls
        growth_oversold=30.0,  # RSI < 30 for bull put spreads
        growth_overbought=70.0  # RSI > 70 for bear call spreads
    )
    
    # Combine all candidates
    all_symbols = universe['dividend_candidates'] + universe['growth_candidates']
    
    logger.info(f"📊 Analyzing {len(all_symbols)} companies for portfolio eligibility...")
    
    try:
        # Scan for eligible companies
        scan_results = portfolio_strategy.scan_for_eligible_companies(all_symbols)
        
        # Get portfolio allocation
        allocation = portfolio_strategy.get_portfolio_allocation()
        
        # Get trading rules summary
        trading_rules = portfolio_strategy.get_trading_rules_summary()
        
        # Validate portfolio balance
        validation = portfolio_strategy.validate_portfolio_balance()
        
        # Display results
        display_portfolio_results(scan_results, allocation, trading_rules, validation)
        
        # Save results
        save_portfolio_results(scan_results, allocation, trading_rules, validation)
        
        logger.info("\n✅ Portfolio strategy analysis completed successfully!")
        
        return {
            'scan_results': scan_results,
            'allocation': allocation,
            'trading_rules': trading_rules,
            'validation': validation
        }
        
    except Exception as e:
        logger.error(f"❌ Error during portfolio analysis: {e}")
        return None

def display_portfolio_results(scan_results, allocation, trading_rules, validation):
    """Display comprehensive portfolio results."""
    logger.info("\n" + "="*60)
    logger.info("📊 PORTFOLIO STRATEGY RESULTS")
    logger.info("="*60)
    
    # Portfolio Allocation
    logger.info(f"\n📈 PORTFOLIO ALLOCATION:")
    logger.info(f"   Dividend Aristocrats: {trading_rules['portfolio_allocation']['dividend_aristocrats']}")
    logger.info(f"   High-Growth Unprofitable: {trading_rules['portfolio_allocation']['high_growth_unprofitable']}")
    
    # Scan Results
    logger.info(f"\n🔍 SCAN RESULTS:")
    logger.info(f"   Dividend Aristocrats Found: {scan_results['dividend_aristocrats']['count']}")
    logger.info(f"   High-Growth Unprofitable Found: {scan_results['high_growth_unprofitable']['count']}")
    logger.info(f"   Total Eligible Companies: {scan_results['total_eligible']}")
    
    # Trading Rules
    logger.info(f"\n⚙️ TRADING RULES:")
    
    dividend_rules = trading_rules['trading_rules']['dividend_aristocrats']
    logger.info(f"   Dividend Aristocrats:")
    logger.info(f"     Strategy: {dividend_rules['strategy']}")
    logger.info(f"     RSI Threshold: {dividend_rules['rsi_threshold']}")
    logger.info(f"     Actions: {', '.join(dividend_rules['actions'])}")
    
    growth_rules = trading_rules['trading_rules']['high_growth_unprofitable']
    logger.info(f"   High-Growth Unprofitable:")
    logger.info(f"     Strategy: {growth_rules['strategy']}")
    logger.info(f"     Oversold Threshold: {growth_rules['oversold_threshold']}")
    logger.info(f"     Overbought Threshold: {growth_rules['overbought_threshold']}")
    logger.info(f"     Actions: {', '.join(growth_rules['actions'])}")
    
    # Validation Results
    logger.info(f"\n✅ VALIDATION RESULTS:")
    current_balance = validation['current_balance']
    logger.info(f"   Current Dividend Aristocrats: {current_balance['dividend_aristocrats']}")
    logger.info(f"   Current High-Growth Unprofitable: {current_balance['high_growth_unprofitable']}")
    logger.info(f"   Total Eligible: {current_balance['total_eligible']}")
    
    if validation['recommendations']:
        logger.info(f"\n💡 RECOMMENDATIONS:")
        for rec in validation['recommendations']:
            logger.info(f"   • {rec}")
    
    # Show top candidates
    if allocation['dividend_aristocrats']['symbols']:
        logger.info(f"\n🏛️ TOP DIVIDEND ARISTOCRAT CANDIDATES:")
        for i, symbol in enumerate(allocation['dividend_aristocrats']['symbols'][:10], 1):
            logger.info(f"   {i}. {symbol}")
    
    if allocation['high_growth_unprofitable']['symbols']:
        logger.info(f"\n📈 TOP HIGH-GROWTH UNPROFITABLE CANDIDATES:")
        for i, symbol in enumerate(allocation['high_growth_unprofitable']['symbols'][:10], 1):
            logger.info(f"   {i}. {symbol}")

def save_portfolio_results(scan_results, allocation, trading_rules, validation, 
                         filename="portfolio_strategy_results.json"):
    """Save portfolio strategy results."""
    try:
        # Add metadata
        output = {
            'timestamp': datetime.now().isoformat(),
            'strategy_version': 'portfolio_v1.0',
            'allocation_targets': {
                'dividend_aristocrats': '70%',
                'high_growth_unprofitable': '30%'
            },
            'trading_rules': {
                'dividend_aristocrats': 'Buy and hold, sell covered calls when RSI > 70',
                'high_growth_unprofitable': 'Aggressive spreads based on RSI thresholds'
            },
            'scan_results': scan_results,
            'allocation': allocation,
            'trading_rules': trading_rules,
            'validation': validation
        }
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"✅ Results saved to {filename}")
        
    except Exception as e:
        logger.error(f"❌ Error saving results: {e}")

def demonstrate_strategy():
    """Demonstrate the portfolio strategy rules."""
    logger.info("\n📋 PORTFOLIO STRATEGY RULES")
    logger.info("="*50)
    
    logger.info("🎯 PORTFOLIO ALLOCATION:")
    logger.info("  • 70% Dividend Aristocrats (stability)")
    logger.info("  • 30% High-Growth Unprofitable (growth)")
    
    logger.info("\n🏛️ DIVIDEND ARISTOCRAT RULES:")
    logger.info("  • Must have 10+ years of consistent dividend payments")
    logger.info("  • Dividend yield > 2.5%")
    logger.info("  • Payout ratio < 60%")
    logger.info("  • ROE > 10%")
    logger.info("  • Trading: Buy and hold, sell covered calls when RSI > 70")
    
    logger.info("\n📈 HIGH-GROWTH UNPROFITABLE RULES:")
    logger.info("  • Revenue growth > 30% year-over-year")
    logger.info("  • Operating income < 0 (unprofitable)")
    logger.info("  • Free cash flow > 0 (positive cash generation)")
    logger.info("  • Trading: Aggressive spreads based on RSI")
    logger.info("    - RSI < 30: Buy bull put spread")
    logger.info("    - RSI > 70: Sell bear call spread")
    
    logger.info("\n⚙️ RSI THRESHOLDS:")
    logger.info("  • Dividend Aristocrats: RSI > 70 (sell covered calls)")
    logger.info("  • High-Growth Unprofitable: RSI < 30 (bull put) or RSI > 70 (bear call)")

def main():
    """Main function."""
    logger.info("🚀 PORTFOLIO STRATEGY SCANNER")
    logger.info("="*60)
    
    # Demonstrate strategy rules
    demonstrate_strategy()
    
    # Analyze portfolio strategy
    results = analyze_portfolio_strategy()
    
    if results:
        logger.info("\n✅ Portfolio strategy analysis completed!")
        logger.info("📊 The system is ready to implement your 70/30 allocation strategy")
        logger.info("🎯 Only companies meeting strict empirical criteria will be included")
    else:
        logger.error("❌ Portfolio strategy analysis failed")

if __name__ == "__main__":
    main() 