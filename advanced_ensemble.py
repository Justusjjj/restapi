import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime, timedelta
import MetaTrader5 as mt5


class AdvancedEnsembleModel:
	"""
	Advanced ensemble model with dynamic model selection based on market conditions.
	Automatically selects the best performing model for current market regime.
	"""
	
	def __init__(self, models_config: Dict, market_regime_detector=None):
		self.models_config = models_config
		self.market_regime_detector = market_regime_detector
		self.models = {}
		self.model_performance = {}
		self.regime_performance = {}
		self.current_regime = "normal"
		self.ensemble_weights = {}
		
		# Initialize models
		self._initialize_models()
	
	def _initialize_models(self):
		"""Initialize all configured models"""
		
		for model_name, config in self.models_config.items():
			if config["type"] == "lstm":
				from deep_model import LSTMModel
				self.models[model_name] = LSTMModel(
					input_dim=config["input_dim"],
					hidden_dim=config["hidden_dim"],
					num_layers=config["num_layers"],
					dropout=config["dropout"]
				)
			elif config["type"] == "hybrid":
				from hybrid_transformer_tree import HybridTransformerTree
				self.models[model_name] = HybridTransformerTree(
					input_dim=config["input_dim"],
					seq_len=config["seq_len"],
					tree_type=config["tree_type"]
				)
			elif config["type"] == "transformer":
				from transformers import AutoModel, AutoTokenizer
				self.models[model_name] = AutoModel.from_pretrained(config["model_name"])
			elif config["type"] == "sklearn":
				from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
				if config["model_class"] == "RandomForest":
					self.models[model_name] = RandomForestClassifier(**config["params"])
				elif config["model_class"] == "GradientBoosting":
					self.models[model_name] = GradientBoostingClassifier(**config["params"])
	
	def train_ensemble(self, X: np.ndarray, y: np.ndarray, 
					  validation_split: float = 0.2) -> Dict:
		"""
		Train all models in the ensemble and evaluate performance
		"""
		
		# Split data for validation
		split_idx = int(len(X) * (1 - validation_split))
		X_train, X_val = X[:split_idx], X[split_idx:]
		y_train, y_val = y[:split_idx], y[split_idx:]
		
		training_results = {}
		
		for model_name, model in self.models.items():
			print(f"Training {model_name}...")
			
			try:
				if hasattr(model, 'train'):
					# Custom training method
					result = model.train(X_train, y_train)
					training_results[model_name] = result
				else:
					# Sklearn models
					model.fit(X_train, y_train)
					training_results[model_name] = {"status": "trained"}
				
				# Evaluate on validation set
				if hasattr(model, 'predict'):
					predictions = model.predict(X_val)
					accuracy = accuracy_score(y_val, predictions)
					self.model_performance[model_name] = {
						"accuracy": accuracy,
						"last_updated": datetime.now(),
						"validation_size": len(X_val)
					}
				
			except Exception as e:
				print(f"Error training {model_name}: {e}")
				training_results[model_name] = {"error": str(e)}
		
		# Calculate ensemble weights based on performance
		self._calculate_ensemble_weights()
		
		return training_results
	
	def _calculate_ensemble_weights(self):
		"""Calculate dynamic weights for ensemble voting"""
		
		if not self.model_performance:
			# Equal weights if no performance data
			self.ensemble_weights = {name: 1.0/len(self.models) for name in self.models.keys()}
			return
		
		# Weight based on accuracy and recency
		total_weight = 0
		weights = {}
		
		for model_name, perf in self.model_performance.items():
			# Base weight from accuracy
			base_weight = perf["accuracy"]
			
			# Recency bonus (more recent = higher weight)
			days_old = (datetime.now() - perf["last_updated"]).days
			recency_factor = max(0.5, 1.0 - (days_old / 30))  # Decay over 30 days
			
			# Final weight
			weight = base_weight * recency_factor
			weights[model_name] = weight
			total_weight += weight
		
		# Normalize weights
		if total_weight > 0:
			self.ensemble_weights = {name: weight/total_weight for name, weight in weights.items()}
		else:
			self.ensemble_weights = {name: 1.0/len(self.models) for name in self.models.keys()}
	
	def predict_ensemble(self, X: np.ndarray, use_regime_selection: bool = True) -> Dict:
		"""
		Make ensemble prediction with optional regime-based model selection
		"""
		
		if use_regime_selection and self.market_regime_detector:
			# Detect current market regime
			self.current_regime = self.market_regime_detector.detect_regime(X)
			
			# Use regime-specific model if available
			if self.current_regime in self.regime_performance:
				best_model = max(self.regime_performance[self.current_regime].items(), 
								key=lambda x: x[1]["accuracy"])
				model_name = best_model[0]
				
				if model_name in self.models:
					prediction = self.models[model_name].predict(X)
					confidence = self.model_performance.get(model_name, {}).get("accuracy", 0.5)
					
					return {
						"prediction": prediction,
						"confidence": confidence,
						"method": "regime_selected",
						"model_used": model_name,
						"regime": self.current_regime
					}
		
		# Standard ensemble voting
		predictions = {}
		confidences = {}
		
		for model_name, model in self.models.items():
			try:
				if hasattr(model, 'predict'):
					pred = model.predict(X)
					predictions[model_name] = pred
					confidences[model_name] = self.model_performance.get(model_name, {}).get("accuracy", 0.5)
			except Exception as e:
				print(f"Error predicting with {model_name}: {e}")
				continue
		
		if not predictions:
			return {"prediction": [0], "confidence": 0.0, "method": "fallback"}
		
		# Weighted voting
		final_prediction = []
		total_confidence = 0
		
		for i in range(len(X)):
			votes = {}
			weighted_confidence = 0
			
			for model_name, pred in predictions.items():
				vote = pred[i] if isinstance(pred, np.ndarray) else pred
				weight = self.ensemble_weights.get(model_name, 0)
				confidence = confidences.get(model_name, 0)
				
				if vote not in votes:
					votes[vote] = 0
				votes[vote] += weight
				weighted_confidence += confidence * weight
			
			# Select most voted class
			final_vote = max(votes.items(), key=lambda x: x[1])[0]
			final_prediction.append(final_vote)
			total_confidence += weighted_confidence
		
		avg_confidence = total_confidence / len(X) if len(X) > 0 else 0
		
		return {
			"prediction": final_prediction,
			"confidence": avg_confidence,
			"method": "ensemble_voting",
			"weights": self.ensemble_weights,
			"individual_predictions": predictions
		}
	
	def update_model_performance(self, model_name: str, X: np.ndarray, y: np.ndarray):
		"""Update model performance with new data"""
		
		if model_name not in self.models:
			return
		
		try:
			model = self.models[model_name]
			predictions = model.predict(X)
			accuracy = accuracy_score(y, predictions)
			
			self.model_performance[model_name] = {
				"accuracy": accuracy,
				"last_updated": datetime.now(),
				"validation_size": len(X)
			}
			
			# Update regime-specific performance
			if self.market_regime_detector:
				regime = self.market_regime_detector.detect_regime(X)
				if regime not in self.regime_performance:
					self.regime_performance[regime] = {}
				
				self.regime_performance[regime][model_name] = {
					"accuracy": accuracy,
					"last_updated": datetime.now()
				}
			
			# Recalculate ensemble weights
			self._calculate_ensemble_weights()
			
		except Exception as e:
			print(f"Error updating performance for {model_name}: {e}")
	
	def get_ensemble_status(self) -> Dict:
		"""Get current status of the ensemble"""
		
		return {
			"models_count": len(self.models),
			"current_regime": self.current_regime,
			"ensemble_weights": self.ensemble_weights,
			"model_performance": self.model_performance,
			"regime_performance": self.regime_performance,
			"last_updated": datetime.now().isoformat()
		}
	
	def save_ensemble(self, path: str):
		"""Save the entire ensemble"""
		
		ensemble_data = {
			"models_config": self.models_config,
			"model_performance": self.model_performance,
			"regime_performance": self.regime_performance,
			"ensemble_weights": self.ensemble_weights,
			"current_regime": self.current_regime
		}
		
		# Save model states
		for model_name, model in self.models.items():
			if hasattr(model, 'state_dict'):
				torch.save(model.state_dict(), f"{path}_{model_name}.pt")
			elif hasattr(model, 'save_model'):
				model.save_model(f"{path}_{model_name}.pt")
			else:
				joblib.dump(model, f"{path}_{model_name}.pkl")
		
		# Save ensemble metadata
		joblib.dump(ensemble_data, f"{path}_ensemble_metadata.pkl")
	
	def load_ensemble(self, path: str):
		"""Load the entire ensemble"""
		
		# Load ensemble metadata
		ensemble_data = joblib.load(f"{path}_ensemble_metadata.pkl")
		
		self.models_config = ensemble_data["models_config"]
		self.model_performance = ensemble_data["model_performance"]
		self.regime_performance = ensemble_data["regime_performance"]
		self.ensemble_weights = ensemble_data["ensemble_weights"]
		self.current_regime = ensemble_data["current_regime"]
		
		# Reinitialize models
		self._initialize_models()
		
		# Load model states
		for model_name, model in self.models.items():
			try:
				if hasattr(model, 'load_state_dict'):
					state = torch.load(f"{path}_{model_name}.pt")
					model.load_state_dict(state)
				elif hasattr(model, 'load_model'):
					model.load_model(f"{path}_{model_name}.pt")
				else:
					model = joblib.load(f"{path}_{model_name}.pkl")
					self.models[model_name] = model
			except Exception as e:
				print(f"Error loading {model_name}: {e}")


class MarketRegimeDetector:
	"""
	Advanced market regime detection using multiple indicators
	"""
	
	def __init__(self, lookback_period: int = 100):
		self.lookback_period = lookback_period
		self.regime_thresholds = {
			"trending": 0.6,
			"ranging": 0.4,
			"volatile": 0.7,
			"calm": 0.3
		}
	
	def detect_regime(self, data: np.ndarray) -> str:
		"""
		Detect current market regime based on price data
		"""
		
		if len(data) < self.lookback_period:
			return "normal"
		
		# Extract price data (assuming close prices are in column 3)
		prices = data[-self.lookback_period:, 3] if data.shape[1] > 3 else data[-self.lookback_period:, 0]
		
		# Calculate regime indicators
		trend_strength = self._calculate_trend_strength(prices)
		volatility = self._calculate_volatility(prices)
		range_ratio = self._calculate_range_ratio(prices)
		
		# Determine regime
		if trend_strength > self.regime_thresholds["trending"]:
			if volatility > self.regime_thresholds["volatile"]:
				return "trending_volatile"
			else:
				return "trending_calm"
		elif range_ratio < self.regime_thresholds["ranging"]:
			if volatility > self.regime_thresholds["volatile"]:
				return "ranging_volatile"
			else:
				return "ranging_calm"
		else:
			return "normal"
	
	def _calculate_trend_strength(self, prices: np.ndarray) -> float:
		"""Calculate trend strength using linear regression R²"""
		
		if len(prices) < 10:
			return 0.0
		
		x = np.arange(len(prices))
		coef = np.polyfit(x, prices, 1)
		y_pred = np.polyval(coef, x)
		
		# Calculate R²
		ss_res = np.sum((prices - y_pred) ** 2)
		ss_tot = np.sum((prices - np.mean(prices)) ** 2)
		
		if ss_tot == 0:
			return 0.0
		
		r_squared = 1 - (ss_res / ss_tot)
		return max(0, r_squared)
	
	def _calculate_volatility(self, prices: np.ndarray) -> float:
		"""Calculate normalized volatility"""
		
		if len(prices) < 2:
			return 0.0
		
		returns = np.diff(np.log(prices))
		volatility = np.std(returns)
		
		# Normalize to 0-1 scale
		return min(volatility * 100, 1.0)  # Scale factor for forex
	
	def _calculate_range_ratio(self, prices: np.ndarray) -> float:
		"""Calculate range ratio (high-low range vs price level)"""
		
		if len(prices) < 2:
			return 0.0
		
		price_range = np.max(prices) - np.min(prices)
		avg_price = np.mean(prices)
		
		if avg_price == 0:
			return 0.0
		
		return price_range / avg_price


def create_advanced_ensemble_config() -> Dict:
	"""
	Create configuration for advanced ensemble model
	"""
	
	return {
		"lstm_model": {
			"type": "lstm",
			"input_dim": 11,
			"hidden_dim": 128,
			"num_layers": 2,
			"dropout": 0.2
		},
		"hybrid_model": {
			"type": "hybrid",
			"input_dim": 11,
			"seq_len": 64,
			"tree_type": "random_forest"
		},
		"transformer_model": {
			"type": "transformer",
			"model_name": "microsoft/DialoGPT-medium"
		},
		"random_forest": {
			"type": "sklearn",
			"model_class": "RandomForest",
			"params": {
				"n_estimators": 200,
				"max_depth": 15,
				"min_samples_split": 5,
				"random_state": 42
			}
		},
		"gradient_boosting": {
			"type": "sklearn",
			"model_class": "GradientBoosting",
			"params": {
				"n_estimators": 200,
				"learning_rate": 0.1,
				"max_depth": 8,
				"random_state": 42
			}
		}
	}