"""
Alternative Data Integration
Integrates sentiment analysis, news feeds, and alternative data sources.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import requests
import json
import logging
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class AlternativeDataManager:
    """Manages alternative data sources for enhanced analysis."""
    
    def __init__(self):
        self.sentiment_cache = {}
        self.news_cache = {}
        self.cache_duration = 3600  # 1 hour
        
    def get_sentiment_score(self, symbol: str) -> float:
        """
        Get sentiment score for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Sentiment score between -1 and 1
        """
        cache_key = f"{symbol}_sentiment"
        
        # Check cache
        if cache_key in self.sentiment_cache:
            cached_data = self.sentiment_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_duration:
                return cached_data['value']
        
        try:
            # Simulate sentiment analysis (in practice, use real API)
            # This is a simplified version - you'd integrate with actual sentiment APIs
            
            # Generate synthetic sentiment based on recent price action
            # In reality, you'd use news sentiment, social media sentiment, etc.
            sentiment = self._generate_synthetic_sentiment(symbol)
            
            # Cache result
            self.sentiment_cache[cache_key] = {
                'value': sentiment,
                'timestamp': time.time()
            }
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {e}")
            return 0.0
    
    def _generate_synthetic_sentiment(self, symbol: str) -> float:
        """Generate synthetic sentiment score for testing."""
        # In practice, this would call real sentiment APIs
        # For now, generate based on symbol hash for consistency
        import hashlib
        hash_val = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
        sentiment = (hash_val % 200 - 100) / 100.0  # Range: -1 to 1
        return sentiment
    
    def get_news_sentiment(self, symbol: str, days: int = 7) -> Dict[str, float]:
        """
        Get news sentiment for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            Dictionary with sentiment metrics
        """
        cache_key = f"{symbol}_news_{days}"
        
        # Check cache
        if cache_key in self.news_cache:
            cached_data = self.news_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_duration:
                return cached_data['value']
        
        try:
            # Simulate news sentiment analysis
            # In practice, you'd use news APIs like NewsAPI, Alpha Vantage News, etc.
            
            sentiment_data = self._generate_synthetic_news_sentiment(symbol, days)
            
            # Cache result
            self.news_cache[cache_key] = {
                'value': sentiment_data,
                'timestamp': time.time()
            }
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {e}")
            return {
                'positive_ratio': 0.5,
                'negative_ratio': 0.5,
                'neutral_ratio': 0.0,
                'overall_sentiment': 0.0,
                'news_count': 0
            }
    
    def _generate_synthetic_news_sentiment(self, symbol: str, days: int) -> Dict[str, float]:
        """Generate synthetic news sentiment for testing."""
        # In practice, this would analyze real news articles
        import hashlib
        hash_val = int(hashlib.md5(f"{symbol}_{days}".encode()).hexdigest()[:8], 16)
        
        # Generate realistic sentiment distribution
        positive_ratio = 0.3 + (hash_val % 40) / 100.0  # 30-70%
        negative_ratio = 0.2 + (hash_val % 30) / 100.0  # 20-50%
        neutral_ratio = 1.0 - positive_ratio - negative_ratio
        
        overall_sentiment = (positive_ratio - negative_ratio) / (positive_ratio + negative_ratio + neutral_ratio)
        news_count = 10 + (hash_val % 20)  # 10-30 articles
        
        return {
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'neutral_ratio': neutral_ratio,
            'overall_sentiment': overall_sentiment,
            'news_count': news_count
        }
    
    def get_social_sentiment(self, symbol: str) -> Dict[str, float]:
        """
        Get social media sentiment for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with social sentiment metrics
        """
        try:
            # Simulate social media sentiment
            # In practice, you'd use Twitter API, Reddit API, etc.
            
            import hashlib
            hash_val = int(hashlib.md5(f"{symbol}_social".encode()).hexdigest()[:8], 16)
            
            # Generate social sentiment metrics
            bullish_mentions = 50 + (hash_val % 100)
            bearish_mentions = 30 + (hash_val % 80)
            total_mentions = bullish_mentions + bearish_mentions
            
            if total_mentions > 0:
                bullish_ratio = bullish_mentions / total_mentions
                bearish_ratio = bearish_mentions / total_mentions
                sentiment_score = (bullish_mentions - bearish_mentions) / total_mentions
            else:
                bullish_ratio = 0.5
                bearish_ratio = 0.5
                sentiment_score = 0.0
            
            return {
                'bullish_mentions': bullish_mentions,
                'bearish_mentions': bearish_mentions,
                'total_mentions': total_mentions,
                'bullish_ratio': bullish_ratio,
                'bearish_ratio': bearish_ratio,
                'sentiment_score': sentiment_score
            }
            
        except Exception as e:
            logger.error(f"Error getting social sentiment for {symbol}: {e}")
            return {
                'bullish_mentions': 0,
                'bearish_mentions': 0,
                'total_mentions': 0,
                'bullish_ratio': 0.5,
                'bearish_ratio': 0.5,
                'sentiment_score': 0.0
            }
    
    def get_insider_trading_data(self, symbol: str) -> Dict[str, any]:
        """
        Get insider trading data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with insider trading metrics
        """
        try:
            # Simulate insider trading data
            # In practice, you'd use SEC filings, insider trading APIs, etc.
            
            import hashlib
            hash_val = int(hashlib.md5(f"{symbol}_insider".encode()).hexdigest()[:8], 16)
            
            # Generate insider trading metrics
            buy_volume = 10000 + (hash_val % 50000)
            sell_volume = 5000 + (hash_val % 30000)
            net_volume = buy_volume - sell_volume
            
            return {
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'net_volume': net_volume,
                'buy_transactions': 5 + (hash_val % 10),
                'sell_transactions': 3 + (hash_val % 8),
                'insider_sentiment': 'bullish' if net_volume > 0 else 'bearish'
            }
            
        except Exception as e:
            logger.error(f"Error getting insider trading data for {symbol}: {e}")
            return {
                'buy_volume': 0,
                'sell_volume': 0,
                'net_volume': 0,
                'buy_transactions': 0,
                'sell_transactions': 0,
                'insider_sentiment': 'neutral'
            }
    
    def get_composite_sentiment_score(self, symbol: str) -> float:
        """
        Get composite sentiment score combining multiple sources.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Composite sentiment score between -1 and 1
        """
        try:
            # Get sentiment from multiple sources
            general_sentiment = self.get_sentiment_score(symbol)
            news_sentiment = self.get_news_sentiment(symbol)['overall_sentiment']
            social_sentiment = self.get_social_sentiment(symbol)['sentiment_score']
            
            # Get insider trading sentiment
            insider_data = self.get_insider_trading_data(symbol)
            insider_sentiment = 0.1 if insider_data['insider_sentiment'] == 'bullish' else -0.1
            
            # Weight the different sentiment sources
            weights = {
                'general': 0.2,
                'news': 0.3,
                'social': 0.3,
                'insider': 0.2
            }
            
            composite_score = (
                general_sentiment * weights['general'] +
                news_sentiment * weights['news'] +
                social_sentiment * weights['social'] +
                insider_sentiment * weights['insider']
            )
            
            # Clamp between -1 and 1
            return max(-1.0, min(1.0, composite_score))
            
        except Exception as e:
            logger.error(f"Error calculating composite sentiment for {symbol}: {e}")
            return 0.0

class AlternativeDataSignalGenerator:
    """Generates trading signals based on alternative data."""
    
    def __init__(self):
        self.data_manager = AlternativeDataManager()
        
    def generate_sentiment_signals(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Generate sentiment-based trading signals.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with sentiment signals for each symbol
        """
        signals = {}
        
        for symbol in symbols:
            try:
                # Get composite sentiment
                sentiment_score = self.data_manager.get_composite_sentiment_score(symbol)
                
                # Get detailed sentiment data
                news_sentiment = self.data_manager.get_news_sentiment(symbol)
                social_sentiment = self.data_manager.get_social_sentiment(symbol)
                insider_data = self.data_manager.get_insider_trading_data(symbol)
                
                # Generate signal based on sentiment
                signal_strength = abs(sentiment_score)
                signal_direction = 'BUY' if sentiment_score > 0.1 else 'SELL' if sentiment_score < -0.1 else 'HOLD'
                
                signals[symbol] = {
                    'sentiment_score': sentiment_score,
                    'signal_direction': signal_direction,
                    'signal_strength': signal_strength,
                    'news_sentiment': news_sentiment,
                    'social_sentiment': social_sentiment,
                    'insider_data': insider_data,
                    'confidence': min(1.0, signal_strength * 2)  # Scale to 0-1
                }
                
            except Exception as e:
                logger.error(f"Error generating sentiment signal for {symbol}: {e}")
                signals[symbol] = {
                    'sentiment_score': 0.0,
                    'signal_direction': 'HOLD',
                    'signal_strength': 0.0,
                    'confidence': 0.0
                }
        
        return signals

def test_alternative_data():
    """Test the alternative data integration."""
    print("Testing Alternative Data Integration:")
    print("=" * 50)
    
    # Initialize alternative data manager
    data_manager = AlternativeDataManager()
    signal_generator = AlternativeDataSignalGenerator()
    
    # Test symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    
    for symbol in test_symbols:
        print(f"\n{symbol}:")
        
        # Get sentiment scores
        sentiment = data_manager.get_sentiment_score(symbol)
        news_sentiment = data_manager.get_news_sentiment(symbol)
        social_sentiment = data_manager.get_social_sentiment(symbol)
        insider_data = data_manager.get_insider_trading_data(symbol)
        composite_sentiment = data_manager.get_composite_sentiment_score(symbol)
        
        print(f"  General Sentiment: {sentiment:.3f}")
        print(f"  News Sentiment: {news_sentiment['overall_sentiment']:.3f}")
        print(f"  Social Sentiment: {social_sentiment['sentiment_score']:.3f}")
        print(f"  Insider Sentiment: {insider_data['insider_sentiment']}")
        print(f"  Composite Sentiment: {composite_sentiment:.3f}")
    
    # Test signal generation
    print(f"\nGenerating Sentiment Signals:")
    print("=" * 30)
    
    signals = signal_generator.generate_sentiment_signals(test_symbols)
    
    for symbol, signal in signals.items():
        print(f"{symbol}: {signal['signal_direction']} (Strength: {signal['signal_strength']:.3f}, Confidence: {signal['confidence']:.3f})")

if __name__ == "__main__":
    test_alternative_data() 