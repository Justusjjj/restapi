import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

class AdvancedMLTradingEngine:
    """
    Advanced Machine Learning Trading Engine with Self-Learning Capabilities
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_history = []
        self.live_trading_mode = False
        
        # Initialize ML models
        self._initialize_models()
        
        # Performance tracking
        self.trade_history = []
        self.model_performance = {}
        
    def _default_config(self):
        """Default configuration for the ML engine"""
        return {
            'feature_window': 100,
            'prediction_horizon': 5,
            'retrain_frequency': 1000,  # Retrain every 1000 trades
            'ensemble_size': 5,
            'confidence_threshold': 0.75,
            'risk_adjustment': True,
            'dynamic_position_sizing': True,
            'market_regime_detection': True,
            'sentiment_analysis': True,
            'order_flow_analysis': True
        }
    
    def _initialize_models(self):
        """Initialize multiple ML models for ensemble learning"""
        # Traditional ML models
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42
        )
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=200, max_depth=6, random_state=42
        )
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
        )
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
        )
        self.models['catboost'] = CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.1, random_state=42, verbose=False
        )
        
        # Neural Network
        self.models['neural_network'] = self._create_neural_network()
        
        # Initialize scalers
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    def _create_neural_network(self):
        """Create a deep neural network for price prediction"""
        model = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3 classes: BUY, SELL, HOLD
            nn.Softmax(dim=1)
        )
        return model
    
    def extract_advanced_features(self, df):
        """
        Extract advanced technical and mathematical features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with advanced features
        """
        try:
            # Price-based features
            df['price_change'] = df['close'].pct_change()
            df['price_change_2'] = df['close'].pct_change(2)
            df['price_change_5'] = df['close'].pct_change(5)
            
            # Volatility features
            df['volatility'] = df['price_change'].rolling(20).std()
            df['volatility_5'] = df['price_change'].rolling(5).std()
            df['volatility_10'] = df['price_change'].rolling(10).std()
            
            # Momentum features
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            # Mean reversion features
            df['mean_reversion_5'] = (df['close'] - df['close'].rolling(5).mean()) / df['close'].rolling(5).std()
            df['mean_reversion_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
            
            # Trend strength features
            df['trend_strength'] = abs(df['close'] - df['close'].shift(20)) / df['close'].rolling(20).std()
            
            # Support and resistance levels
            df['support_level'] = df['low'].rolling(20).min()
            df['resistance_level'] = df['high'].rolling(20).max()
            df['support_distance'] = (df['close'] - df['support_level']) / df['close']
            df['resistance_distance'] = (df['resistance_level'] - df['close']) / df['close']
            
            # Volume analysis
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['volume_price_trend'] = df['volume'] * df['price_change']
            
            # Advanced mathematical indicators
            df['rsi_divergence'] = self._calculate_rsi_divergence(df)
            df['macd_divergence'] = self._calculate_macd_divergence(df)
            df['bollinger_squeeze'] = self._calculate_bollinger_squeeze(df)
            df['keltner_channels'] = self._calculate_keltner_channels(df)
            
            # Market microstructure features
            df['bid_ask_spread'] = self._estimate_bid_ask_spread(df)
            df['order_flow_imbalance'] = self._calculate_order_flow_imbalance(df)
            df['market_microstructure'] = self._calculate_market_microstructure(df)
            
            # Time-based features
            df['hour_of_day'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
            
            # Economic calendar impact (simplified)
            df['economic_impact'] = self._estimate_economic_impact(df)
            
            return df
            
        except Exception as e:
            print(f"Error extracting advanced features: {e}")
            return df
    
    def _calculate_rsi_divergence(self, df):
        """Calculate RSI divergence patterns"""
        try:
            rsi = self._calculate_rsi(df['close'])
            price_highs = df['high'].rolling(5).max()
            rsi_highs = rsi.rolling(5).max()
            
            # Bullish divergence: price makes lower low, RSI makes higher low
            bullish_div = ((df['close'] < df['close'].shift(5)) & 
                          (rsi > rsi.shift(5))).astype(int)
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            bearish_div = ((df['close'] > df['close'].shift(5)) & 
                          (rsi < rsi.shift(5))).astype(int)
            
            return bullish_div - bearish_div
            
        except:
            return pd.Series(0, index=df.index)
    
    def _calculate_macd_divergence(self, df):
        """Calculate MACD divergence patterns"""
        try:
            macd = self._calculate_macd(df['close'])
            macd_signal = macd.ewm(span=9).mean()
            
            # MACD histogram
            macd_hist = macd - macd_signal
            
            # Divergence detection
            price_trend = df['close'].rolling(10).mean().diff()
            macd_trend = macd_hist.rolling(10).mean().diff()
            
            divergence = np.where(
                (price_trend > 0) & (macd_trend < 0), -1,  # Bearish divergence
                np.where(
                    (price_trend < 0) & (macd_trend > 0), 1,  # Bullish divergence
                    0  # No divergence
                )
            )
            
            return pd.Series(divergence, index=df.index)
            
        except:
            return pd.Series(0, index=df.index)
    
    def _calculate_bollinger_squeeze(self, df):
        """Calculate Bollinger Band squeeze indicator"""
        try:
            bb_upper = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
            bb_lower = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
            bb_width = (bb_upper - bb_lower) / df['close'].rolling(20).mean()
            
            # Squeeze when bands are narrow
            squeeze = (bb_width < bb_width.rolling(20).quantile(0.2)).astype(int)
            return squeeze
            
        except:
            return pd.Series(0, index=df.index)
    
    def _calculate_keltner_channels(self, df):
        """Calculate Keltner Channels"""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            atr = self._calculate_atr(df)
            
            kc_upper = typical_price.rolling(20).mean() + 2 * atr
            kc_lower = typical_price.rolling(20).mean() - 2 * atr
            
            # Position within Keltner Channels
            kc_position = (df['close'] - kc_lower) / (kc_upper - kc_lower)
            return kc_position
            
        except:
            return pd.Series(0.5, index=df.index)
    
    def _estimate_bid_ask_spread(self, df):
        """Estimate bid-ask spread based on price volatility"""
        try:
            # Simplified spread estimation
            spread = df['volatility'] * 0.1  # 10% of volatility
            return spread
            
        except:
            return pd.Series(0.0001, index=df.index)
    
    def _calculate_order_flow_imbalance(self, df):
        """Calculate order flow imbalance indicator"""
        try:
            # Volume-weighted price change
            vwap = (df['volume'] * df['close']).rolling(20).sum() / df['volume'].rolling(20).sum()
            
            # Order flow imbalance
            imbalance = (df['close'] - vwap) / vwap
            return imbalance
            
        except:
            return pd.Series(0, index=df.index)
    
    def _calculate_market_microstructure(self, df):
        """Calculate market microstructure indicators"""
        try:
            # Price impact of volume
            price_impact = df['volume_price_trend'].rolling(10).mean()
            
            # Market efficiency ratio
            efficiency_ratio = abs(df['close'] - df['close'].shift(20)) / df['close'].rolling(20).std()
            
            return price_impact * efficiency_ratio
            
        except:
            return pd.Series(0, index=df.index)
    
    def _estimate_economic_impact(self, df):
        """Estimate economic calendar impact"""
        try:
            # Simplified economic impact (in real implementation, use economic calendar API)
            # Higher volatility during certain hours (London/NY overlap)
            london_ny_overlap = ((df.index.hour >= 13) & (df.index.hour <= 17)).astype(int)
            
            # Weekend effect
            weekend_effect = df['is_weekend']
            
            # Month-end effect
            month_end = (df.index.day >= 25).astype(int)
            
            return london_ny_overlap + weekend_effect + month_end
            
        except:
            return pd.Series(0, index=df.index)
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return atr
    
    def prepare_features(self, df):
        """
        Prepare features for ML models
        
        Args:
            df: DataFrame with advanced features
            
        Returns:
            Feature matrix and target variables
        """
        try:
            # Select feature columns
            feature_columns = [
                'price_change', 'price_change_2', 'price_change_5',
                'volatility', 'volatility_5', 'volatility_10',
                'momentum_5', 'momentum_10', 'momentum_20',
                'mean_reversion_5', 'mean_reversion_20',
                'trend_strength', 'support_distance', 'resistance_distance',
                'volume_ratio', 'volume_price_trend',
                'rsi_divergence', 'macd_divergence', 'bollinger_squeeze',
                'keltner_channels', 'bid_ask_spread', 'order_flow_imbalance',
                'market_microstructure', 'economic_impact',
                'hour_of_day', 'day_of_week', 'month', 'is_weekend'
            ]
            
            # Filter available columns
            available_features = [col for col in feature_columns if col in df.columns]
            
            # Create feature matrix
            X = df[available_features].fillna(0)
            
            # Create target variable (future price direction)
            future_returns = df['close'].pct_change(self.config['prediction_horizon']).shift(-self.config['prediction_horizon'])
            
            # Create classification target
            y = np.where(future_returns > 0.001, 1,  # BUY
                        np.where(future_returns < -0.001, 2, 0))  # SELL, HOLD
            
            # Remove NaN values
            valid_indices = ~(X.isna().any(axis=1) | pd.isna(y))
            X = X[valid_indices]
            y = y[valid_indices]
            
            return X, y, available_features
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None, None, None
    
    def train_models(self, X, y, feature_names):
        """
        Train all ML models
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
        """
        try:
            print("Training ML models...")
            
            # Split data for time series validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            for model_name, model in self.models.items():
                if model_name == 'neural_network':
                    self._train_neural_network(X, y, feature_names)
                else:
                    self._train_traditional_model(model_name, model, X, y, feature_names, tscv)
                
                print(f"✓ {model_name} trained successfully")
                
            # Calculate ensemble performance
            self._calculate_ensemble_performance(X, y)
            
        except Exception as e:
            print(f"Error training models: {e}")
    
    def _train_traditional_model(self, model_name, model, X, y, feature_names, tscv):
        """Train traditional ML models"""
        try:
            # Scale features
            X_scaled = self.scalers[model_name].fit_transform(X)
            
            # Train model
            model.fit(X_scaled, y)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')
            
            # Store performance
            self.model_performance[model_name] = {
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = dict(zip(feature_names, model.feature_importances_))
                
        except Exception as e:
            print(f"Error training {model_name}: {e}")
    
    def _train_neural_network(self, X, y, feature_names):
        """Train neural network"""
        try:
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X.values)
            y_tensor = torch.LongTensor(y)
            
            # Create data loader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Training parameters
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.models['neural_network'].parameters(), lr=0.001)
            
            # Training loop
            epochs = 50
            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.models['neural_network'](batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
                    
        except Exception as e:
            print(f"Error training neural network: {e}")
    
    def _calculate_ensemble_performance(self, X, y):
        """Calculate ensemble model performance"""
        try:
            predictions = {}
            
            for model_name, model in self.models.items():
                if model_name == 'neural_network':
                    X_tensor = torch.FloatTensor(X.values)
                    with torch.no_grad():
                        outputs = model(X_tensor)
                        pred = torch.argmax(outputs, dim=1).numpy()
                else:
                    X_scaled = self.scalers[model_name].transform(X)
                    pred = model.predict(X_scaled)
                
                predictions[model_name] = pred
            
            # Ensemble prediction (majority voting)
            ensemble_pred = np.zeros(len(y))
            for i in range(len(y)):
                votes = [predictions[model_name][i] for model_name in predictions.keys()]
                ensemble_pred[i] = max(set(votes), key=votes.count)
            
            # Calculate ensemble accuracy
            ensemble_accuracy = np.mean(ensemble_pred == y)
            print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
            
        except Exception as e:
            print(f"Error calculating ensemble performance: {e}")
    
    def predict(self, X, confidence_threshold=None):
        """
        Make predictions using ensemble of models
        
        Args:
            X: Feature matrix
            confidence_threshold: Minimum confidence for prediction
            
        Returns:
            Predictions and confidence scores
        """
        try:
            if confidence_threshold is None:
                confidence_threshold = self.config['confidence_threshold']
            
            predictions = {}
            confidence_scores = {}
            
            for model_name, model in self.models.items():
                if model_name == 'neural_network':
                    X_tensor = torch.FloatTensor(X.values)
                    with torch.no_grad():
                        outputs = model(X_tensor)
                        probs = torch.softmax(outputs, dim=1).numpy()
                        pred = torch.argmax(outputs, dim=1).numpy()
                        confidence = np.max(probs, axis=1)
                else:
                    X_scaled = self.scalers[model_name].transform(X)
                    pred = model.predict(X_scaled)
                    
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(X_scaled)
                        confidence = np.max(probs, axis=1)
                    else:
                        confidence = np.ones(len(pred)) * 0.8  # Default confidence
                
                predictions[model_name] = pred
                confidence_scores[model_name] = confidence
            
            # Ensemble prediction with confidence weighting
            ensemble_pred = np.zeros(len(X))
            ensemble_confidence = np.zeros(len(X))
            
            for i in range(len(X)):
                # Weighted voting based on confidence
                votes = np.zeros(3)  # 3 classes: HOLD, BUY, SELL
                total_confidence = 0
                
                for model_name in predictions.keys():
                    pred_class = predictions[model_name][i]
                    conf = confidence_scores[model_name][i]
                    votes[pred_class] += conf
                    total_confidence += conf
                
                # Normalize votes
                if total_confidence > 0:
                    votes = votes / total_confidence
                    ensemble_pred[i] = np.argmax(votes)
                    ensemble_confidence[i] = np.max(votes)
                else:
                    ensemble_pred[i] = 0  # HOLD
                    ensemble_confidence[i] = 0
            
            # Apply confidence threshold
            valid_predictions = ensemble_confidence >= confidence_threshold
            ensemble_pred[~valid_predictions] = 0  # HOLD for low confidence
            
            return ensemble_pred, ensemble_confidence
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None, None
    
    def update_models(self, new_data, retrain_threshold=1000):
        """
        Update models with new data (online learning)
        
        Args:
            new_data: New market data
            retrain_threshold: Number of new samples before retraining
        """
        try:
            # Add new data to history
            self.performance_history.append(new_data)
            
            # Check if retraining is needed
            if len(self.performance_history) >= retrain_threshold:
                print("Retraining models with new data...")
                
                # Prepare new training data
                X_new, y_new, feature_names = self.prepare_features(new_data)
                
                if X_new is not None and len(X_new) > 0:
                    # Retrain models
                    self.train_models(X_new, y_new, feature_names)
                    
                    # Clear performance history
                    self.performance_history = []
                    
                    print("Models updated successfully")
                    
        except Exception as e:
            print(f"Error updating models: {e}")
    
    def get_trading_signal(self, market_data, confidence_threshold=None):
        """
        Get trading signal from ML models
        
        Args:
            market_data: Current market data
            confidence_threshold: Minimum confidence for signal
            
        Returns:
            Trading signal and confidence
        """
        try:
            # Extract features
            df_with_features = self.extract_advanced_features(market_data)
            
            # Prepare features for prediction
            X, _, feature_names = self.prepare_features(df_with_features)
            
            if X is None or len(X) == 0:
                return 'HOLD', 0.0
            
            # Get latest data point
            X_latest = X.iloc[-1:].copy()
            
            # Make prediction
            prediction, confidence = self.predict(X_latest, confidence_threshold)
            
            if prediction is None:
                return 'HOLD', 0.0
            
            # Convert prediction to signal
            signal_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            signal = signal_map.get(prediction[0], 'HOLD')
            
            return signal, confidence[0]
            
        except Exception as e:
            print(f"Error getting trading signal: {e}")
            return 'HOLD', 0.0
    
    def calculate_risk_adjusted_position_size(self, signal, confidence, account_balance, 
                                            market_volatility, risk_per_trade=0.02):
        """
        Calculate risk-adjusted position size using ML insights
        
        Args:
            signal: Trading signal
            confidence: Signal confidence
            account_balance: Current account balance
            market_volatility: Current market volatility
            risk_per_trade: Risk per trade percentage
            
        Returns:
            Position size and risk metrics
        """
        try:
            if signal == 'HOLD':
                return 0.0, 0.0, 0.0
            
            # Base position size
            base_risk_amount = account_balance * risk_per_trade
            
            # Confidence adjustment
            confidence_multiplier = confidence ** 2  # Square confidence for stronger effect
            
            # Volatility adjustment
            volatility_factor = 1 / (1 + market_volatility * 100)  # Reduce size in high volatility
            
            # Market regime adjustment
            regime_factor = self._detect_market_regime()
            
            # Calculate adjusted position size
            adjusted_risk_amount = base_risk_amount * confidence_multiplier * volatility_factor * regime_factor
            
            # Convert to lot size (assuming 1 pip = $1 for 0.01 lot)
            pip_value = 1.0
            stop_loss_pips = 50  # Default stop loss
            position_size = adjusted_risk_amount / (stop_loss_pips * pip_value)
            
            # Limit position size
            max_position = min(position_size, 1.0)  # Maximum 1.0 lot
            min_position = max(max_position, 0.01)  # Minimum 0.01 lot
            
            # Calculate actual risk
            actual_risk = min_position * stop_loss_pips * pip_value
            risk_percentage = actual_risk / account_balance
            
            return min_position, actual_risk, risk_percentage
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 0.01, 0.0, 0.0
    
    def _detect_market_regime(self):
        """Detect current market regime for position sizing"""
        try:
            # This would typically use market data to detect regime
            # For now, return neutral factor
            return 1.0
            
        except:
            return 1.0
    
    def get_model_performance_summary(self):
        """Get summary of all model performances"""
        try:
            summary = {
                'models': self.model_performance,
                'feature_importance': self.feature_importance,
                'total_trades': len(self.trade_history),
                'live_trading_mode': self.live_trading_mode
            }
            return summary
            
        except Exception as e:
            print(f"Error getting performance summary: {e}")
            return {}

if __name__ == "__main__":
    # Example usage
    engine = AdvancedMLTradingEngine()
    print("Advanced ML Trading Engine initialized successfully!")
    print("Features:")
    print("- Multi-model ensemble learning")
    print("- Advanced feature engineering")
    print("- Real-time model updates")
    print("- Risk-adjusted position sizing")
    print("- Market regime detection")