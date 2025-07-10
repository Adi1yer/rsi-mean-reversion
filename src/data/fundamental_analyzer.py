"""
Fundamental Analysis Module
Analyzes financial statements to determine company classification and eligibility.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CompanyType(Enum):
    """Company classification based on fundamental analysis."""
    DIVIDEND_ARISTOCRAT = "dividend_aristocrat"
    HIGH_GROWTH_UNPROFITABLE = "high_growth_unprofitable"
    PROFITABLE_GROWTH = "profitable_growth"
    UNQUALIFIED = "unqualified"

@dataclass
class FundamentalMetrics:
    """Fundamental analysis metrics."""
    symbol: str
    company_type: CompanyType
    revenue_growth_yoy: float  # Year-over-year revenue growth
    operating_income: float  # Operating income (profitability)
    free_cash_flow: float  # Free cash flow
    net_income: float  # Net income
    total_revenue: float  # Total revenue
    debt_to_equity: float  # Debt to equity ratio
    current_ratio: float  # Current ratio
    roe: float  # Return on equity
    roa: float  # Return on assets
    pe_ratio: float  # Price to earnings ratio
    pb_ratio: float  # Price to book ratio
    dividend_yield: float  # Dividend yield
    payout_ratio: float  # Dividend payout ratio
    analysis_date: datetime
    is_eligible: bool
    reason: str

class FundamentalAnalyzer:
    """Analyzes fundamental data to classify companies and determine eligibility."""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
        
    def analyze_company(self, symbol: str) -> FundamentalMetrics:
        """
        Analyze a company's fundamentals to determine classification and eligibility.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            FundamentalMetrics object with analysis results
        """
        try:
            # Get company data
            ticker = yf.Ticker(symbol)
            
            # Get financial statements
            income_stmt = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            
            if income_stmt.empty or balance_sheet.empty or cash_flow.empty:
                return self._create_unqualified_metrics(symbol, "Missing financial data")
            
            # Calculate key metrics
            metrics = self._calculate_metrics(symbol, income_stmt, balance_sheet, cash_flow)
            
            # Determine company type and eligibility
            company_type = self._classify_company(metrics)
            is_eligible = company_type != CompanyType.UNQUALIFIED
            
            # Create metrics object
            fundamental_metrics = FundamentalMetrics(
                symbol=symbol,
                company_type=company_type,
                revenue_growth_yoy=metrics['revenue_growth_yoy'],
                operating_income=metrics['operating_income'],
                free_cash_flow=metrics['free_cash_flow'],
                net_income=metrics['net_income'],
                total_revenue=metrics['total_revenue'],
                debt_to_equity=metrics['debt_to_equity'],
                current_ratio=metrics['current_ratio'],
                roe=metrics['roe'],
                roa=metrics['roa'],
                pe_ratio=metrics['pe_ratio'],
                pb_ratio=metrics['pb_ratio'],
                dividend_yield=metrics['dividend_yield'],
                payout_ratio=metrics['payout_ratio'],
                analysis_date=datetime.now(),
                is_eligible=is_eligible,
                reason="Meets classification criteria" if is_eligible else "Does not meet classification criteria"
            )
            
            logger.info(f"Analyzed {symbol}: {company_type.value}, Eligible: {is_eligible}")
            return fundamental_metrics
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return self._create_unqualified_metrics(symbol, f"Analysis error: {e}")
    
    def _calculate_metrics(self, symbol: str, income_stmt: pd.DataFrame, 
                          balance_sheet: pd.DataFrame, cash_flow: pd.DataFrame) -> Dict:
        """Calculate key financial metrics."""
        metrics = {}
        
        try:
            # Revenue growth (year-over-year)
            if 'Total Revenue' in income_stmt.index:
                revenue_data = income_stmt.loc['Total Revenue']
                if len(revenue_data) >= 2:
                    current_revenue = revenue_data.iloc[0]
                    previous_revenue = revenue_data.iloc[1]
                    metrics['revenue_growth_yoy'] = ((current_revenue - previous_revenue) / previous_revenue) * 100
                    metrics['total_revenue'] = current_revenue
                else:
                    metrics['revenue_growth_yoy'] = 0
                    metrics['total_revenue'] = revenue_data.iloc[0] if len(revenue_data) > 0 else 0
            else:
                metrics['revenue_growth_yoy'] = 0
                metrics['total_revenue'] = 0
            
            # Operating income (profitability)
            if 'Operating Income' in income_stmt.index:
                metrics['operating_income'] = income_stmt.loc['Operating Income'].iloc[0]
            else:
                metrics['operating_income'] = 0
            
            # Net income
            if 'Net Income' in income_stmt.index:
                metrics['net_income'] = income_stmt.loc['Net Income'].iloc[0]
            else:
                metrics['net_income'] = 0
            
            # Free cash flow
            if 'Free Cash Flow' in cash_flow.index:
                metrics['free_cash_flow'] = cash_flow.loc['Free Cash Flow'].iloc[0]
            else:
                # Calculate FCF: Operating Cash Flow - Capital Expenditure
                if 'Operating Cash Flow' in cash_flow.index and 'Capital Expenditure' in cash_flow.index:
                    ocf = cash_flow.loc['Operating Cash Flow'].iloc[0]
                    capex = cash_flow.loc['Capital Expenditure'].iloc[0]
                    metrics['free_cash_flow'] = ocf - abs(capex)  # Capex is typically negative
                else:
                    metrics['free_cash_flow'] = 0
            
            # Balance sheet metrics
            if 'Total Debt' in balance_sheet.index and 'Total Stockholder Equity' in balance_sheet.index:
                debt = balance_sheet.loc['Total Debt'].iloc[0]
                equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                metrics['debt_to_equity'] = debt / equity if equity != 0 else float('inf')
            else:
                metrics['debt_to_equity'] = 0
            
            if 'Total Current Assets' in balance_sheet.index and 'Total Current Liabilities' in balance_sheet.index:
                current_assets = balance_sheet.loc['Total Current Assets'].iloc[0]
                current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0]
                metrics['current_ratio'] = current_assets / current_liabilities if current_liabilities != 0 else float('inf')
            else:
                metrics['current_ratio'] = 0
            
            # ROE and ROA
            if 'Total Stockholder Equity' in balance_sheet.index and metrics['net_income'] != 0:
                equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                metrics['roe'] = (metrics['net_income'] / equity) * 100 if equity != 0 else 0
            else:
                metrics['roe'] = 0
            
            if 'Total Assets' in balance_sheet.index and metrics['net_income'] != 0:
                assets = balance_sheet.loc['Total Assets'].iloc[0]
                metrics['roa'] = (metrics['net_income'] / assets) * 100 if assets != 0 else 0
            else:
                metrics['roa'] = 0
            
            # Market ratios (from yfinance)
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            metrics['pe_ratio'] = info.get('trailingPE', 0)
            metrics['pb_ratio'] = info.get('priceToBook', 0)
            metrics['dividend_yield'] = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            
            # Payout ratio
            if metrics['dividend_yield'] > 0 and metrics['pe_ratio'] > 0:
                metrics['payout_ratio'] = (metrics['dividend_yield'] / 100) * metrics['pe_ratio'] * 100
            else:
                metrics['payout_ratio'] = 0
                
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {e}")
            # Set default values
            for key in ['revenue_growth_yoy', 'operating_income', 'free_cash_flow', 'net_income', 
                       'total_revenue', 'debt_to_equity', 'current_ratio', 'roe', 'roa', 
                       'pe_ratio', 'pb_ratio', 'dividend_yield', 'payout_ratio']:
                if key not in metrics:
                    metrics[key] = 0
        
        return metrics
    
    def _classify_company(self, metrics: Dict) -> CompanyType:
        """Classify company based on fundamental metrics."""
        
        # High-Growth Unprofitable: >25% revenue growth, negative operating income, positive FCF
        if (metrics['revenue_growth_yoy'] > 25.0 and 
            metrics['operating_income'] < 0 and 
            metrics['free_cash_flow'] > 0):
            return CompanyType.HIGH_GROWTH_UNPROFITABLE
        
        # Dividend Aristocrat: >1.5% dividend yield, <75% payout ratio, profitable, >5% ROE
        if (metrics['dividend_yield'] > 1.5 and 
            metrics['payout_ratio'] < 75.0 and 
            metrics['operating_income'] > 0 and 
            metrics['roe'] > 5.0):
            return CompanyType.DIVIDEND_ARISTOCRAT
        
        # Profitable Growth: >10% revenue growth, profitable, >10% ROE
        if (metrics['revenue_growth_yoy'] > 10.0 and 
            metrics['operating_income'] > 0 and 
            metrics['roe'] > 10.0):
            return CompanyType.PROFITABLE_GROWTH
        
        return CompanyType.UNQUALIFIED
    
    def _create_unqualified_metrics(self, symbol: str, reason: str) -> FundamentalMetrics:
        """Create unqualified metrics for companies that can't be analyzed."""
        return FundamentalMetrics(
            symbol=symbol,
            company_type=CompanyType.UNQUALIFIED,
            revenue_growth_yoy=0,
            operating_income=0,
            free_cash_flow=0,
            net_income=0,
            total_revenue=0,
            debt_to_equity=0,
            current_ratio=0,
            roe=0,
            roa=0,
            pe_ratio=0,
            pb_ratio=0,
            dividend_yield=0,
            payout_ratio=0,
            analysis_date=datetime.now(),
            is_eligible=False,
            reason=reason
        )
    
    def analyze_portfolio(self, symbols: List[str]) -> Dict[str, FundamentalMetrics]:
        """Analyze multiple companies and return results."""
        results = {}
        
        logger.info(f"Analyzing {len(symbols)} companies for fundamental eligibility...")
        
        for symbol in symbols:
            try:
                metrics = self.analyze_company(symbol)
                results[symbol] = metrics
                
                # Log detailed results
                logger.info(f"  {symbol}:")
                logger.info(f"    Type: {metrics.company_type.value}")
                logger.info(f"    Revenue Growth: {metrics.revenue_growth_yoy:.1f}%")
                logger.info(f"    Operating Income: ${metrics.operating_income:,.0f}")
                logger.info(f"    Free Cash Flow: ${metrics.free_cash_flow:,.0f}")
                logger.info(f"    Eligible: {metrics.is_eligible}")
                if not metrics.is_eligible:
                    logger.info(f"    Reason: {metrics.reason}")
                
            except Exception as e:
                logger.error(f"❌ Error analyzing {symbol}: {e}")
                results[symbol] = self._create_unqualified_metrics(symbol, f"Analysis failed: {e}")
        
        return results
    
    def get_eligible_companies(self, symbols: List[str]) -> Dict[CompanyType, List[str]]:
        """Get eligible companies by category."""
        analysis = self.analyze_portfolio(symbols)
        
        eligible = {
            CompanyType.DIVIDEND_ARISTOCRAT: [],
            CompanyType.HIGH_GROWTH_UNPROFITABLE: [],
            CompanyType.PROFITABLE_GROWTH: [],
            CompanyType.UNQUALIFIED: []
        }
        
        for symbol, metrics in analysis.items():
            if metrics.is_eligible:
                eligible[metrics.company_type].append(symbol)
            else:
                eligible[CompanyType.UNQUALIFIED].append(symbol)
        
        return eligible
    
    def generate_analysis_report(self, symbols: List[str]) -> Dict:
        """Generate a comprehensive analysis report."""
        analysis = self.analyze_portfolio(symbols)
        eligible = self.get_eligible_companies(symbols)
        
        report = {
            'summary': {
                'total_companies': len(symbols),
                'eligible_companies': sum(len(companies) for companies in eligible.values() if companies),
                'dividend_aristocrats': len(eligible[CompanyType.DIVIDEND_ARISTOCRAT]),
                'high_growth_unprofitable': len(eligible[CompanyType.HIGH_GROWTH_UNPROFITABLE]),
                'profitable_growth': len(eligible[CompanyType.PROFITABLE_GROWTH]),
                'unqualified': len(eligible[CompanyType.UNQUALIFIED])
            },
            'detailed_analysis': analysis,
            'eligible_companies': eligible,
            'recommendations': []
        }
        
        # Generate recommendations
        if len(eligible[CompanyType.HIGH_GROWTH_UNPROFITABLE]) < 3:
            report['recommendations'].append(
                "Consider adding more high-growth unprofitable companies to diversify the portfolio"
            )
        
        if len(eligible[CompanyType.DIVIDEND_ARISTOCRAT]) < 3:
            report['recommendations'].append(
                "Consider adding more dividend aristocrats for stability"
            )
        
        if len(eligible[CompanyType.UNQUALIFIED]) > len(symbols) * 0.5:
            report['recommendations'].append(
                "High number of unqualified companies - review selection criteria"
            )
        
        return report 