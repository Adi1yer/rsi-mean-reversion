"""
Stock Universe Scanner
Scans all US stocks on major exchanges to identify companies meeting fundamental criteria.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
import logging
from datetime import datetime, timedelta
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from src.data.fundamental_analyzer import FundamentalAnalyzer, CompanyType, FundamentalMetrics

logger = logging.getLogger(__name__)

@dataclass
class StockInfo:
    """Basic stock information."""
    symbol: str
    name: str
    exchange: str
    market_cap: float
    sector: str
    industry: str
    country: str

class StockUniverseScanner:
    """Scans all US stocks to identify eligible companies."""
    
    def __init__(self, fundamental_analyzer: Optional[FundamentalAnalyzer] = None):
        self.fundamental_analyzer = fundamental_analyzer or FundamentalAnalyzer()
        self.cache_file = "data/stock_universe_cache.json"
        self.cache_duration = 86400  # 24 hours
        
        # Major US exchanges
        self.us_exchanges = ['NYSE', 'NASDAQ', 'AMEX']
        
        # Market cap thresholds (in billions)
        self.min_market_cap = 1.0  # $1B minimum
        self.max_market_cap = 1000.0  # $1T maximum
        
        # Rate limiting
        self.request_delay = 0.1  # 100ms between requests
        self.max_concurrent = 5
        
    def get_all_us_stocks(self) -> List[StockInfo]:
        """Get all US stocks from major exchanges."""
        logger.info("Scanning US stock universe...")
        
        all_stocks = []
        
        try:
            # Get tickers from yfinance
            tickers = yf.Tickers('^GSPC ^DJI ^IXIC')  # Get major indices first
            indices = tickers.tickers
            
            # Get all tickers from major exchanges
            for exchange in self.us_exchanges:
                logger.info(f"Scanning {exchange}...")
                
                try:
                    # Get tickers for this exchange
                    exchange_tickers = self._get_exchange_tickers(exchange)
                    
                    for ticker in exchange_tickers:
                        try:
                            # Get basic info
                            stock_info = self._get_stock_info(ticker)
                            
                            if stock_info and self._is_eligible_stock(stock_info):
                                all_stocks.append(stock_info)
                                
                            time.sleep(self.request_delay)  # Rate limiting
                            
                        except Exception as e:
                            logger.debug(f"Error processing {ticker}: {e}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"Error scanning {exchange}: {e}")
                    continue
            
            logger.info(f"Found {len(all_stocks)} eligible US stocks")
            return all_stocks
            
        except Exception as e:
            logger.error(f"❌ Error scanning stock universe: {e}")
            return []
    
    def _get_exchange_tickers(self, exchange: str) -> List[str]:
        """Get tickers for a specific exchange."""
        try:
            # Use yfinance to get tickers
            if exchange == 'NYSE':
                # NYSE tickers (common ones)
                return self._get_nyse_tickers()
            elif exchange == 'NASDAQ':
                # NASDAQ tickers (common ones)
                return self._get_nasdaq_tickers()
            elif exchange == 'AMEX':
                # AMEX tickers (common ones)
                return self._get_amex_tickers()
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting {exchange} tickers: {e}")
            return []
    
    def _get_nyse_tickers(self) -> List[str]:
        """Get common NYSE tickers."""
        # Common NYSE stocks (focus on liquid, well-known companies)
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'JPM', 'JNJ',
            'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC', 'ADBE', 'CRM', 'NFLX', 'KO',
            'PEP', 'ABT', 'TMO', 'AVGO', 'COST', 'ACN', 'DHR', 'LLY', 'NEE', 'TXN', 'HON',
            'UNP', 'RTX', 'LOW', 'UPS', 'SPGI', 'INTU', 'QCOM', 'T', 'CAT', 'IBM', 'GS',
            'MS', 'AXP', 'GE', 'CVX', 'XOM', 'WMT', 'MRK', 'PFE', 'ABBV', 'VZ', 'CMCSA',
            'BMY', 'TFC', 'USB', 'COF', 'SCHW', 'BLK', 'AMGN', 'GILD', 'DUK', 'SO', 'PLD',
            'CCI', 'REGN', 'BDX', 'TJX', 'NOC', 'ITW', 'MMC', 'ETN', 'AON', 'ICE', 'SHW',
            'APD', 'ADI', 'KLAC', 'HUM', 'ISRG', 'SYK', 'VRTX', 'BIIB', 'ALGN', 'DXCM',
            'IDXX', 'WST', 'RMD', 'ZTS', 'ILMN', 'MTCH', 'FTNT', 'CDNS', 'SNPS', 'KLAC',
            'MCHP', 'LRCX', 'AMAT', 'ASML', 'MU', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN'
        ]
    
    def _get_nasdaq_tickers(self) -> List[str]:
        """Get common NASDAQ tickers."""
        # Common NASDAQ stocks (focus on tech and growth companies)
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'ADBE', 'CRM', 'NFLX',
            'PYPL', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT', 'LRCX', 'KLAC',
            'ASML', 'SNPS', 'CDNS', 'FTNT', 'MTCH', 'ILMN', 'BIIB', 'ALGN', 'DXCM', 'IDXX',
            'WST', 'RMD', 'ZTS', 'VRTX', 'ISRG', 'HUM', 'ADI', 'MCHP', 'KLAC', 'LRCX',
            'AMAT', 'ASML', 'MU', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'NVDA', 'TSLA',
            'UBER', 'LYFT', 'PLTR', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'CHPT', 'PLUG',
            'FCEL', 'COIN', 'HOOD', 'SOFI', 'AFRM', 'UPST', 'LC', 'OPRT', 'ETSY', 'PINS',
            'SNAP', 'SPOT', 'ROKU', 'TTD', 'CRWD', 'DDOG', 'MDB', 'ESTC', 'TEAM', 'ZM',
            'DOCU', 'OKTA', 'SNOW', 'NET', 'ZS', 'TWLO', 'SQ', 'SHOP', 'MRNA', 'BNTX',
            'NVAX', 'INO', 'CRSP', 'EDIT', 'NTLA', 'NIO', 'XPEV', 'LI', 'LCID', 'CHPT',
            'PLUG', 'FCEL', 'COIN', 'HOOD', 'SOFI', 'AFRM', 'UPST', 'LC', 'OPRT', 'ETSY',
            'PINS', 'SNAP', 'SPOT', 'ROKU', 'TTD', 'CRWD', 'DDOG', 'MDB', 'ESTC', 'TEAM',
            'ZM', 'DOCU', 'OKTA', 'SNOW', 'NET', 'ZS', 'TWLO', 'SQ', 'SHOP', 'MRNA',
            'BNTX', 'NVAX', 'INO', 'CRSP', 'EDIT', 'NTLA'
        ]
    
    def _get_amex_tickers(self) -> List[str]:
        """Get common AMEX tickers."""
        # Common AMEX stocks (mostly ETFs and some stocks)
        return [
            'SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'GLD', 'SLV', 'USO', 'UNG',
            'XLK', 'XLV', 'XLF', 'XLE', 'XLI', 'XLP', 'XLU', 'XLB', 'XLY', 'XLRE',
            'VTI', 'VOO', 'VEA', 'VWO', 'BND', 'AGG', 'LQD', 'HYG', 'JNK', 'EMB'
        ]
    
    def _get_stock_info(self, symbol: str) -> Optional[StockInfo]:
        """Get basic stock information."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract basic info
            name = info.get('longName', symbol)
            exchange = info.get('exchange', 'Unknown')
            market_cap = info.get('marketCap', 0)
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            country = info.get('country', 'Unknown')
            
            # Only include US stocks
            if country != 'US':
                return None
            
            return StockInfo(
                symbol=symbol,
                name=name,
                exchange=exchange,
                market_cap=market_cap,
                sector=sector,
                industry=industry,
                country=country
            )
            
        except Exception as e:
            logger.debug(f"Error getting info for {symbol}: {e}")
            return None
    
    def _is_eligible_stock(self, stock_info: StockInfo) -> bool:
        """Check if stock meets basic eligibility criteria."""
        # Market cap filter
        if stock_info.market_cap < self.min_market_cap * 1e9:  # Less than $1B
            return False
        if stock_info.market_cap > self.max_market_cap * 1e9:  # More than $1T
            return False
        
        # Exchange filter
        if stock_info.exchange not in self.us_exchanges:
            return False
        
        # Country filter
        if stock_info.country != 'US':
            return False
        
        return True
    
    def scan_for_eligible_companies(self, max_companies: int = 100) -> Dict:
        """Scan all US stocks to find eligible companies."""
        logger.info("Scanning US stock universe for eligible companies...")
        
        # Get all US stocks
        all_stocks = self.get_all_us_stocks()
        
        if not all_stocks:
            logger.error("❌ No stocks found")
            return {'error': 'No stocks found'}
        
        # Limit to max_companies for performance
        if len(all_stocks) > max_companies:
            logger.info(f"Limiting analysis to {max_companies} stocks for performance")
            all_stocks = all_stocks[:max_companies]
        
        # Extract symbols
        symbols = [stock.symbol for stock in all_stocks]
        
        logger.info(f"Analyzing {len(symbols)} US stocks for fundamental eligibility...")
        
        # Analyze fundamentals
        try:
            analysis_results = self.fundamental_analyzer.analyze_portfolio(symbols)
            
            # Create detailed results
            results = {
                'scan_summary': {
                    'total_stocks_scanned': len(symbols),
                    'eligible_companies': 0,
                    'high_growth_unprofitable': 0,
                    'dividend_aristocrats': 0,
                    'profitable_growth': 0,
                    'unqualified': 0
                },
                'eligible_companies': [],
                'unqualified_companies': [],
                'detailed_analysis': analysis_results,
                'recommendations': []
            }
            
            # Process results
            for symbol, metrics in analysis_results.items():
                if metrics.is_eligible:
                    results['scan_summary']['eligible_companies'] += 1
                    results['eligible_companies'].append({
                        'symbol': symbol,
                        'name': next((s.name for s in all_stocks if s.symbol == symbol), symbol),
                        'category': metrics.company_type.value,
                        'revenue_growth': metrics.revenue_growth_yoy,
                        'operating_income': metrics.operating_income,
                        'free_cash_flow': metrics.free_cash_flow
                    })
                    
                    # Update category counts
                    if metrics.company_type == CompanyType.HIGH_GROWTH_UNPROFITABLE:
                        results['scan_summary']['high_growth_unprofitable'] += 1
                    elif metrics.company_type == CompanyType.DIVIDEND_ARISTOCRAT:
                        results['scan_summary']['dividend_aristocrats'] += 1
                    elif metrics.company_type == CompanyType.PROFITABLE_GROWTH:
                        results['scan_summary']['profitable_growth'] += 1
                else:
                    results['scan_summary']['unqualified'] += 1
                    results['unqualified_companies'].append({
                        'symbol': symbol,
                        'name': next((s.name for s in all_stocks if s.symbol == symbol), symbol),
                        'reason': metrics.reason
                    })
            
            # Generate recommendations
            if results['scan_summary']['high_growth_unprofitable'] < 5:
                results['recommendations'].append(
                    "Consider expanding universe to find more high-growth unprofitable companies"
                )
            
            if results['scan_summary']['dividend_aristocrats'] < 5:
                results['recommendations'].append(
                    "Consider adding more dividend aristocrats for portfolio stability"
                )
            
            logger.info(f"Scan completed: {results['scan_summary']['eligible_companies']} eligible companies found")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during fundamental analysis: {e}")
            return {'error': str(e)}
    
    def get_high_growth_unprofitable_candidates(self) -> List[Dict]:
        """Get specifically high-growth unprofitable candidates."""
        logger.info("Finding high-growth unprofitable candidates...")
        
        # Focus on sectors likely to have high-growth unprofitable companies
        target_sectors = [
            'Technology', 'Healthcare', 'Consumer Cyclical', 'Communication Services'
        ]
        
        # Get all stocks
        all_stocks = self.get_all_us_stocks()
        
        # Filter by target sectors
        target_stocks = [
            stock for stock in all_stocks 
            if stock.sector in target_sectors and stock.market_cap < 50 * 1e9  # < $50B
        ]
        
        symbols = [stock.symbol for stock in target_stocks]
        
        logger.info(f"Analyzing {len(symbols)} potential high-growth candidates...")
        
        # Analyze fundamentals
        analysis_results = self.fundamental_analyzer.analyze_portfolio(symbols)
        
        # Filter for high-growth unprofitable
        candidates = []
        for symbol, metrics in analysis_results.items():
            if (metrics.is_eligible and 
                metrics.company_type == CompanyType.HIGH_GROWTH_UNPROFITABLE):
                stock_info = next((s for s in target_stocks if s.symbol == symbol), None)
                candidates.append({
                    'symbol': symbol,
                    'name': stock_info.name if stock_info else symbol,
                    'sector': stock_info.sector if stock_info else 'Unknown',
                    'market_cap': stock_info.market_cap if stock_info else 0,
                    'revenue_growth': metrics.revenue_growth_yoy,
                    'operating_income': metrics.operating_income,
                    'free_cash_flow': metrics.free_cash_flow,
                    'roe': metrics.roe,
                    'debt_to_equity': metrics.debt_to_equity
                })
        
        # Sort by revenue growth
        candidates.sort(key=lambda x: x['revenue_growth'], reverse=True)
        
        logger.info(f"Found {len(candidates)} high-growth unprofitable candidates")
        return candidates
    
    def save_scan_results(self, results: Dict, filename: str = "us_stock_scan_results.json"):
        """Save scan results to file."""
        try:
            # Add timestamp
            results['scan_timestamp'] = datetime.now().isoformat()
            results['scanner_version'] = "v1.0"
            
            # Ensure data directory exists
            os.makedirs('data', exist_ok=True)
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Scan results saved to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Error saving scan results: {e}")
    
    def display_scan_results(self, results: Dict):
        """Display comprehensive scan results."""
        logger.info("\n" + "="*60)
        logger.info("US STOCK UNIVERSE SCAN RESULTS")
        logger.info("="*60)
        
        if 'error' in results:
            logger.error(f"❌ Scan failed: {results['error']}")
            return
        
        summary = results.get('scan_summary', {})
        logger.info(f"\nSCAN SUMMARY:")
        logger.info(f"   Total Stocks Scanned: {summary.get('total_stocks_scanned', 0)}")
        logger.info(f"   Eligible Companies: {summary.get('eligible_companies', 0)}")
        logger.info(f"   High-Growth Unprofitable: {summary.get('high_growth_unprofitable', 0)}")
        logger.info(f"   Dividend Aristocrats: {summary.get('dividend_aristocrats', 0)}")
        logger.info(f"   Profitable Growth: {summary.get('profitable_growth', 0)}")
        logger.info(f"   Unqualified: {summary.get('unqualified', 0)}")
        
        # Show eligible companies
        eligible = results.get('eligible_companies', [])
        if eligible:
            logger.info(f"\n✅ ELIGIBLE COMPANIES:")
            for company in eligible[:10]:  # Show first 10
                logger.info(f"   {company['symbol']} ({company['name']}): {company['category']}")
                logger.info(f"     Revenue Growth: {company['revenue_growth']:.1f}%")
                logger.info(f"     Operating Income: ${company['operating_income']:,.0f}")
                logger.info(f"     Free Cash Flow: ${company['free_cash_flow']:,.0f}")
        
        # Show recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            logger.info(f"\nRECOMMENDATIONS:")
            for rec in recommendations:
                logger.info(f"   • {rec}")
        
        logger.info("\n" + "="*60) 