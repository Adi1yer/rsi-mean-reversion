#!/usr/bin/env python3
"""
Installation Test Script
Verifies that all components of the RSI Mean Reversion Trading Bot work correctly.
"""

import sys
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all core modules can be imported."""
    logger.info("Testing module imports...")
    
    test_modules = [
        "src.data.fundamental_analyzer",
        "src.data.dividend_aristocrat_analyzer", 
        "src.data.reliable_fundamental_data",
        "src.data.stock_universe_scanner",
        "src.data.alternative_data",
        "src.backtesting.portfolio_backtester",
        "src.strategies.rebalancing_strategy",
        "src.ml.signal_generator",
        "src.risk_management.options_strategies"
    ]
    
    failed_imports = []
    
    for module in test_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module}")
        except ImportError as e:
            logger.error(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_dependencies():
    """Test that all required dependencies are installed."""
    logger.info("Testing dependencies...")
    
    required_packages = [
        "pandas", "numpy", "yfinance", "matplotlib", 
        "plotly", "requests", "scipy", "sklearn"
    ]
    
    failed_deps = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            logger.error(f"❌ {package}")
            failed_deps.append(package)
    
    return len(failed_deps) == 0

def test_data_analysis():
    """Test that data analysis components work."""
    logger.info("Testing data analysis components...")
    
    try:
        from src.data.fundamental_analyzer import FundamentalAnalyzer
        from src.data.dividend_aristocrat_analyzer import DividendAristocratAnalyzer
        
        # Test analyzers can be instantiated
        fundamental = FundamentalAnalyzer()
        dividend = DividendAristocratAnalyzer()
        
        logger.info("✅ Data analyzers instantiated successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data analysis test failed: {e}")
        return False

def test_backtesting():
    """Test that backtesting components work."""
    logger.info("Testing backtesting components...")
    
    try:
        from src.backtesting.portfolio_backtester import PortfolioBacktester
        
        # Test backtester can be instantiated
        backtester = PortfolioBacktester(initial_capital=100000)
        
        logger.info("✅ Portfolio backtester instantiated successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Backtesting test failed: {e}")
        return False

def test_main_script():
    """Test that the main script can be imported."""
    logger.info("Testing main script...")
    
    try:
        import run_optimized_backtest
        logger.info("✅ Main backtesting script imports successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Main script test failed: {e}")
        return False

def test_directories():
    """Test that required directories exist or can be created."""
    logger.info("Testing directory structure...")
    
    required_dirs = ['data', 'results', 'src']
    
    for directory in required_dirs:
        if os.path.exists(directory):
            logger.info(f"✅ {directory}/ exists")
        else:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"✅ {directory}/ created")
            except Exception as e:
                logger.error(f"❌ Could not create {directory}/: {e}")
                return False
    
    return True

def run_comprehensive_test():
    """Run all tests and provide a summary."""
    logger.info("="*60)
    logger.info("RSI MEAN REVERSION TRADING BOT - INSTALLATION TEST")
    logger.info("="*60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Module Imports", test_imports),
        ("Directory Structure", test_directories),
        ("Data Analysis", test_data_analysis),
        ("Backtesting", test_backtesting),
        ("Main Script", test_main_script)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{test_name} Test:")
        logger.info("-" * 40)
        results[test_name] = test_func()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:20} {status}")
    
    logger.info("-" * 40)
    logger.info(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("Your installation is working correctly.")
        logger.info("You can now run: python run_optimized_backtest.py")
        return True
    else:
        logger.error(f"\n⚠️  {total - passed} tests failed.")
        logger.error("Please check the error messages above and fix any issues.")
        logger.error("Try running: pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1) 