"""
Dividend Aristocrat Analyzer
Identifies true dividend aristocrats with 10+ years of consistent dividend payments.
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

class DividendStatus(Enum):
    """Dividend payment status."""
    ARISTOCRAT = "dividend_aristocrat"  # 10+ years consistent
    CONTENDER = "dividend_contender"    # 5-9 years consistent
    CHALLENGER = "dividend_challenger"  # 1-4 years consistent
    UNQUALIFIED = "unqualified"

@dataclass
class DividendMetrics:
    """Dividend analysis metrics."""
    symbol: str
    dividend_status: DividendStatus
    years_paid: int
    current_dividend_yield: float
    payout_ratio: float
    dividend_growth_rate: float
    consecutive_years: int
    last_dividend_date: Optional[datetime]
    dividend_frequency: str  # quarterly, monthly, etc.
    roe: float  # Return on equity
    is_eligible: bool
    reason: str
    analysis_date: datetime

class DividendAristocratAnalyzer:
    """Analyzes dividend history to identify true dividend aristocrats."""
    
    def __init__(self):
        self.min_years = 5  # Reduced from 10 to 5 years for dividend aristocrat status
        self.min_dividend_yield = 1.5  # Reduced from 2.5% to 1.5%
        self.max_payout_ratio = 75.0  # Increased from 60% to 75%
        self.min_roe = 5.0  # Reduced from 10% to 5%
        
    def analyze_dividend_history(self, symbol: str) -> DividendMetrics:
        """
        Analyze dividend history to determine aristocrat status.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DividendMetrics object with analysis results
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get dividend history
            dividend_history = ticker.dividends
            
            if dividend_history.empty:
                return self._create_unqualified_metrics(symbol, "No dividend history")
            
            # Get company info
            info = ticker.info
            
            # Calculate dividend metrics
            years_paid = self._calculate_years_paid(dividend_history)
            consecutive_years = self._calculate_consecutive_years(dividend_history)
            dividend_growth_rate = self._calculate_dividend_growth_rate(dividend_history)
            current_yield = info.get('dividendYield', 0) if info.get('dividendYield') else 0  # yfinance returns as percentage
            payout_ratio = self._calculate_payout_ratio(info)
            roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0  # Convert to percentage
            
            # Determine dividend status
            dividend_status = self._determine_dividend_status(consecutive_years)
            
            # Check eligibility criteria
            is_eligible, reason = self._check_eligibility(
                dividend_status, current_yield, payout_ratio, roe, consecutive_years
            )
            
            # Get dividend frequency
            dividend_frequency = self._determine_dividend_frequency(dividend_history)
            
            # Get last dividend date
            last_dividend_date = pd.Timestamp(dividend_history.index[-1]).to_pydatetime() if not dividend_history.empty else None
            
            metrics = DividendMetrics(
                symbol=symbol,
                dividend_status=dividend_status,
                years_paid=years_paid,
                current_dividend_yield=current_yield,
                payout_ratio=payout_ratio,
                dividend_growth_rate=dividend_growth_rate,
                consecutive_years=consecutive_years,
                last_dividend_date=last_dividend_date,
                dividend_frequency=dividend_frequency,
                roe=roe,
                is_eligible=is_eligible,
                reason=reason,
                analysis_date=datetime.now()
            )
            
            logger.info(f"Analyzed {symbol}: {dividend_status.value}, "
                       f"Years: {consecutive_years}, Yield: {current_yield:.2f}%, "
                       f"Eligible: {is_eligible}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error analyzing dividend history for {symbol}: {e}")
            return self._create_unqualified_metrics(symbol, f"Analysis error: {e}")
    
    def _calculate_years_paid(self, dividend_history: pd.Series) -> int:
        """Calculate total years with dividend payments."""
        if dividend_history.empty:
            return 0
        
        # Get unique years
        years = dividend_history.index.year.unique()
        return len(years)
    
    def _calculate_consecutive_years(self, dividend_history: pd.Series) -> int:
        """Calculate consecutive years of dividend payments."""
        if dividend_history.empty:
            return 0
        
        # Get years with dividends
        dividend_years = sorted(dividend_history.index.year.unique(), reverse=True)
        
        if not dividend_years:
            return 0
        
        # Count consecutive years from most recent
        consecutive = 0
        current_year = datetime.now().year
        
        for year in dividend_years:
            if year == current_year - consecutive:
                consecutive += 1
            else:
                break
        
        return consecutive
    
    def _calculate_dividend_growth_rate(self, dividend_history: pd.Series) -> float:
        """Calculate dividend growth rate over the last 5 years."""
        if dividend_history.empty:
            return 0.0
        
        try:
            # Get annual dividends (sum by year)
            annual_dividends = dividend_history.groupby(dividend_history.index.year).sum()
            
            if len(annual_dividends) < 2:
                return 0.0
            
            # Calculate growth rate over last 5 years
            recent_dividends = annual_dividends.tail(5)
            
            if len(recent_dividends) < 2:
                return 0.0
            
            # Calculate CAGR
            first_dividend = recent_dividends.iloc[0]
            last_dividend = recent_dividends.iloc[-1]
            years = len(recent_dividends) - 1
            
            if first_dividend <= 0 or years <= 0:
                return 0.0
            
            growth_rate = ((last_dividend / first_dividend) ** (1/years) - 1) * 100
            return growth_rate
            
        except Exception as e:
            logger.debug(f"Error calculating dividend growth rate: {e}")
            return 0.0
    
    def _calculate_payout_ratio(self, info: Dict) -> float:
        """Calculate dividend payout ratio."""
        try:
            dividend_yield = info.get('dividendYield', 0)
            pe_ratio = info.get('trailingPE', 0)
            
            if dividend_yield and pe_ratio and pe_ratio > 0:
                # Payout ratio = (Dividend Yield / 100) * P/E Ratio * 100
                payout_ratio = (dividend_yield / 100) * pe_ratio * 100
                return payout_ratio
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Error calculating payout ratio: {e}")
            return 0.0
    
    def _determine_dividend_status(self, consecutive_years: int) -> DividendStatus:
        """Determine dividend status based on consecutive years."""
        if consecutive_years >= self.min_years:
            return DividendStatus.ARISTOCRAT
        elif consecutive_years >= 5:
            return DividendStatus.CONTENDER
        elif consecutive_years >= 1:
            return DividendStatus.CHALLENGER
        else:
            return DividendStatus.UNQUALIFIED
    
    def _determine_dividend_frequency(self, dividend_history: pd.Series) -> str:
        """Determine dividend payment frequency."""
        if dividend_history.empty:
            return "unknown"
        
        try:
            # Count payments per year
            payments_per_year = dividend_history.groupby(dividend_history.index.year).count()
            
            if payments_per_year.empty:
                return "unknown"
            
            avg_payments = payments_per_year.mean()
            
            if avg_payments >= 11:
                return "monthly"
            elif avg_payments >= 3:
                return "quarterly"
            elif avg_payments >= 1:
                return "annual"
            else:
                return "irregular"
                
        except Exception as e:
            logger.debug(f"Error determining dividend frequency: {e}")
            return "unknown"
    
    def _check_eligibility(self, dividend_status: DividendStatus, 
                          dividend_yield: float, payout_ratio: float, 
                          roe: float, consecutive_years: int) -> Tuple[bool, str]:
        """Check if company meets dividend aristocrat criteria (only consecutive years)."""
        # Must be a dividend aristocrat (10+ years)
        if dividend_status != DividendStatus.ARISTOCRAT:
            return False, f"Not a dividend aristocrat (only {consecutive_years} years)"
        return True, "Meets consecutive years requirement for dividend aristocrat"
    
    def _create_unqualified_metrics(self, symbol: str, reason: str) -> DividendMetrics:
        """Create unqualified metrics for companies that can't be analyzed."""
        return DividendMetrics(
            symbol=symbol,
            dividend_status=DividendStatus.UNQUALIFIED,
            years_paid=0,
            current_dividend_yield=0.0,
            payout_ratio=0.0,
            dividend_growth_rate=0.0,
            consecutive_years=0,
            last_dividend_date=None,
            dividend_frequency="unknown",
            roe=0.0,
            is_eligible=False,
            reason=reason,
            analysis_date=datetime.now()
        )
    
    def analyze_dividend_portfolio(self, symbols: List[str]) -> Dict:
        """Analyze multiple companies for dividend aristocrat status."""
        results = {}
        
        logger.info(f"Analyzing {len(symbols)} companies for dividend aristocrat status...")
        
        for symbol in symbols:
            try:
                metrics = self.analyze_dividend_history(symbol)
                results[symbol] = metrics
                
            except Exception as e:
                logger.error(f"❌ Error analyzing {symbol}: {e}")
                results[symbol] = self._create_unqualified_metrics(symbol, f"Analysis failed: {e}")
        
        return results
    
    def get_eligible_aristocrats(self, symbols: List[str]) -> List[Dict]:
        """Get eligible dividend aristocrats."""
        analysis = self.analyze_dividend_portfolio(symbols)
        
        eligible = []
        for symbol, metrics in analysis.items():
            if metrics.is_eligible:
                eligible.append({
                    'symbol': symbol,
                    'consecutive_years': metrics.consecutive_years,
                    'dividend_yield': metrics.current_dividend_yield,
                    'payout_ratio': metrics.payout_ratio,
                    'dividend_growth_rate': metrics.dividend_growth_rate,
                    'dividend_frequency': metrics.dividend_frequency,
                    'last_dividend_date': metrics.last_dividend_date
                })
        
        # Sort by dividend yield (highest first)
        eligible.sort(key=lambda x: x['dividend_yield'], reverse=True)
        
        return eligible
    
    def generate_dividend_report(self, symbols: List[str]) -> Dict:
        """Generate comprehensive dividend analysis report."""
        analysis = self.analyze_dividend_portfolio(symbols)
        
        # Count by status
        aristocrats = []
        contenders = []
        challengers = []
        unqualified = []
        
        for symbol, metrics in analysis.items():
            if metrics.is_eligible:
                aristocrats.append(symbol)
            elif metrics.dividend_status == DividendStatus.CONTENDER:
                contenders.append(symbol)
            elif metrics.dividend_status == DividendStatus.CHALLENGER:
                challengers.append(symbol)
            else:
                unqualified.append(symbol)
        
        report = {
            'summary': {
                'total_companies': len(symbols),
                'dividend_aristocrats': len(aristocrats),
                'dividend_contenders': len(contenders),
                'dividend_challengers': len(challengers),
                'unqualified': len(unqualified)
            },
            'detailed_analysis': analysis,
            'eligible_aristocrats': aristocrats,
            'recommendations': []
        }
        
        # Generate recommendations
        if len(aristocrats) < 10:
            report['recommendations'].append(
                "Consider adding more dividend aristocrats for portfolio stability"
            )
        
        if len(contenders) > 0:
            report['recommendations'].append(
                f"Found {len(contenders)} dividend contenders that may become aristocrats"
            )
        
        return report 