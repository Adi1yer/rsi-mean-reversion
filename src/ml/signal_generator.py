"""
Machine Learning Signal Generator
Uses ML models to generate trading signals and improve prediction accuracy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging
import joblib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MLSignalGenerator:
    """Machine learning-based signal generator."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize ML signal generator.
        
        Args:
            model_type: Type of ML model ('random_forest' or 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical and fundamental features for ML model.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with features
        """
        features = data.copy()
        
        # Price-based features
        features['returns'] = features['Close'].pct_change()
        features['log_returns'] = np.log(features['Close'] / features['Close'].shift(1))
        features['price_change'] = features['Close'] - features['Close'].shift(1)
        features['price_change_pct'] = features['price_change'] / features['Close'].shift(1)
        
        # Moving averages
        features['sma_5'] = features['Close'].rolling(window=5).mean()
        features['sma_20'] = features['Close'].rolling(window=20).mean()
        features['sma_50'] = features['Close'].rolling(window=50).mean()
        features['ema_12'] = features['Close'].ewm(span=12).mean()
        features['ema_26'] = features['Close'].ewm(span=26).mean()
        
        # Price relative to moving averages
        features['price_vs_sma5'] = features['Close'] / features['sma_5'] - 1
        features['price_vs_sma20'] = features['Close'] / features['sma_20'] - 1
        features['price_vs_sma50'] = features['Close'] / features['sma_50'] - 1
        
        # Volatility features
        features['volatility_5'] = features['returns'].rolling(window=5).std()
        features['volatility_20'] = features['returns'].rolling(window=20).std()
        features['volatility_50'] = features['returns'].rolling(window=50).std()
        
        # RSI
        features['rsi'] = self._calculate_rsi(features['Close'])
        
        # MACD
        features['macd'] = features['ema_12'] - features['ema_26']
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        features['bb_middle'] = features['Close'].rolling(window=20).mean()
        bb_std = features['Close'].rolling(window=20).std()
        features['bb_upper'] = features['bb_middle'] + (bb_std * 2)
        features['bb_lower'] = features['bb_middle'] - (bb_std * 2)
        features['bb_position'] = (features['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Volume features
        features['volume_sma'] = features['Volume'].rolling(window=20).mean()
        features['volume_ratio'] = features['Volume'] / features['volume_sma']
        
        # Momentum features
        features['momentum_5'] = features['Close'] / features['Close'].shift(5) - 1
        features['momentum_10'] = features['Close'] / features['Close'].shift(10) - 1
        features['momentum_20'] = features['Close'] / features['Close'].shift(20) - 1
        
        # Support and resistance
        features['support_level'] = features['Close'].rolling(window=20).min()
        features['resistance_level'] = features['Close'].rolling(window=20).max()
        features['price_vs_support'] = features['Close'] / features['support_level'] - 1
        features['price_vs_resistance'] = features['Close'] / features['resistance_level'] - 1
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_target(self, data: pd.DataFrame, forward_period: int = 5) -> pd.Series:
        """
        Create target variable for ML model.
        
        Args:
            data: DataFrame with price data
            forward_period: Number of periods to look forward
            
        Returns:
            Target series (1 for positive return, 0 for negative)
        """
        future_returns = data['Close'].shift(-forward_period) / data['Close'] - 1
        target = (future_returns > 0).astype(int)
        return target
    
    def prepare_data(self, data: pd.DataFrame, forward_period: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for ML model training.
        
        Args:
            data: Raw price data
            forward_period: Forward period for target
            
        Returns:
            Tuple of features and target
        """
        # Create features
        features = self.create_features(data)
        
        # Create target
        target = self.create_target(data, forward_period)
        
        # Remove rows with NaN values
        valid_idx = features.notna().all(axis=1) & target.notna()
        features = features.loc[valid_idx]
        target = target.loc[valid_idx]
        
        # Select feature columns (exclude price and volume columns)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.feature_columns = [col for col in features.columns if col not in exclude_cols]
        
        return features[self.feature_columns], target
    
    def train_model(self, data: pd.DataFrame, forward_period: int = 5):
        """
        Train the ML model.
        
        Args:
            data: Training data
            forward_period: Forward period for target
        """
        logger.info(f"Training ML model ({self.model_type})...")
        
        # Prepare data
        features, target = self.prepare_data(data, forward_period)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model accuracy: {accuracy:.3f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        self.is_trained = True
        
        return accuracy
    
    def predict_signal(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate trading signal using trained model.
        
        Args:
            data: Recent price data
            
        Returns:
            Dictionary with prediction probabilities
        """
        if not self.is_trained:
            logger.warning("Model not trained. Returning neutral signal.")
            return {'buy_probability': 0.5, 'sell_probability': 0.5}
        
        # Create features for recent data
        features = self.create_features(data)
        
        # Get latest features
        latest_features = features[self.feature_columns].iloc[-1:].values
        
        # Scale features
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(latest_features_scaled)[0]
        
        return {
            'buy_probability': probabilities[1],
            'sell_probability': probabilities[0]
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_trained or self.model is None:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_columns, importance))
        else:
            return {}
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        if self.is_trained:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'model_type': self.model_type
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
        else:
            logger.warning("No trained model to save.")
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_type = model_data['model_type']
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

def test_ml_signal_generator():
    """Test the ML signal generator."""
    import yfinance as yf
    
    print("Testing ML Signal Generator:")
    print("=" * 40)
    
    # Get sample data
    data = yf.download("AAPL", start="2020-01-01", end="2023-12-31")
    
    # Initialize ML signal generator
    ml_generator = MLSignalGenerator(model_type='random_forest')
    
    # Train model
    accuracy = ml_generator.train_model(data, forward_period=5)
    print(f"Model Accuracy: {accuracy:.3f}")
    
    # Get feature importance
    importance = ml_generator.get_feature_importance()
    print("\nTop 10 Most Important Features:")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, imp) in enumerate(sorted_importance[:10]):
        print(f"  {i+1}. {feature}: {imp:.3f}")
    
    # Generate signal
    signal = ml_generator.predict_signal(data)
    print(f"\nCurrent Signal:")
    print(f"  Buy Probability: {signal['buy_probability']:.3f}")
    print(f"  Sell Probability: {signal['sell_probability']:.3f}")

if __name__ == "__main__":
    test_ml_signal_generator() 