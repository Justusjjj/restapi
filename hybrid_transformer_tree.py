import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, List, Tuple, Optional
import joblib


class TransformerFeatureExtractor(nn.Module):
	"""
	Transformer-based feature extractor for financial time series.
	Extracts meaningful features that decision trees can interpret.
	"""
	
	def __init__(self, input_dim: int = 11, d_model: int = 64, nhead: int = 4, 
				 num_layers: int = 2, seq_len: int = 64, feature_dim: int = 32):
		super().__init__()
		
		self.seq_len = seq_len
		self.feature_dim = feature_dim
		
		# Input projection
		self.input_projection = nn.Linear(input_dim, d_model)
		
		# Positional encoding
		self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))
		
		# Transformer encoder
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=d_model * 4,
			dropout=0.1,
			batch_first=True
		)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		
		# Feature extraction layers
		self.feature_extractor = nn.Sequential(
			nn.Linear(d_model, d_model // 2),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear(d_model // 2, feature_dim),
			nn.Tanh()  # Bounded output for decision tree stability
		)
		
		# Attention pooling
		self.attention_pool = nn.MultiheadAttention(d_model, nhead, batch_first=True)
		
	def forward(self, x):
		# Input projection
		x = self.input_projection(x)  # (batch, seq_len, d_model)
		
		# Add positional encoding
		x = x + self.pos_encoding.unsqueeze(0)
		
		# Transformer encoding
		encoded = self.transformer(x)  # (batch, seq_len, d_model)
		
		# Attention pooling to get global representation
		query = encoded.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
		attended, _ = self.attention_pool(query, encoded, encoded)
		global_repr = attended.squeeze(1)  # (batch, d_model)
		
		# Extract interpretable features
		features = self.feature_extractor(global_repr)  # (batch, feature_dim)
		
		return features


class HybridTransformerTree:
	"""
	Hybrid model combining Transformer feature extraction with Decision Tree classification.
	Transformer extracts complex temporal patterns, Decision Tree makes interpretable decisions.
	"""
	
	def __init__(self, input_dim: int = 11, seq_len: int = 64, 
				 transformer_features: int = 32, tree_type: str = "random_forest"):
		self.input_dim = input_dim
		self.seq_len = seq_len
		self.transformer_features = transformer_features
		self.tree_type = tree_type
		
		# Initialize models
		self.transformer = TransformerFeatureExtractor(
			input_dim=input_dim,
			seq_len=seq_len,
			feature_dim=transformer_features
		)
		
		# Decision tree options
		if tree_type == "random_forest":
			self.tree = RandomForestClassifier(
				n_estimators=100,
				max_depth=10,
				min_samples_split=5,
				min_samples_leaf=2,
				random_state=42
			)
		elif tree_type == "gradient_boosting":
			self.tree = GradientBoostingClassifier(
				n_estimators=100,
				learning_rate=0.1,
				max_depth=6,
				random_state=42
			)
		else:  # single decision tree
			self.tree = DecisionTreeClassifier(
				max_depth=10,
				min_samples_split=5,
				min_samples_leaf=2,
				random_state=42
			)
		
		# Feature scaler
		self.scaler = StandardScaler()
		
		# Training state
		self.is_fitted = False
		self.feature_names = [f"transformer_feat_{i}" for i in range(transformer_features)]
	
	def train(self, X: np.ndarray, y: np.ndarray, 
			  epochs: int = 50, batch_size: int = 64, lr: float = 1e-3,
			  device: str = None) -> Dict:
		"""
		Train the hybrid model in two stages:
		1. Train transformer to extract meaningful features
		2. Train decision tree on extracted features
		"""
		device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		
		# Stage 1: Train transformer feature extractor
		print("Training Transformer feature extractor...")
		self._train_transformer(X, y, epochs, batch_size, lr, device)
		
		# Stage 2: Extract features and train decision tree
		print("Extracting features and training Decision Tree...")
		features = self._extract_features(X, device)
		
		# Scale features for decision tree
		features_scaled = self.scaler.fit_transform(features)
		
		# Train decision tree
		self.tree.fit(features_scaled, y)
		
		self.is_fitted = True
		
		# Calculate performance metrics
		train_pred = self.tree.predict(features_scaled)
		train_acc = accuracy_score(y, train_pred)
		
		return {
			"training_accuracy": train_acc,
			"feature_importance": self._get_feature_importance(),
			"tree_depth": self._get_tree_depth(),
			"num_features": self.transformer_features
		}
	
	def _train_transformer(self, X: np.ndarray, y: np.ndarray, 
						  epochs: int, batch_size: int, lr: float, device: str):
		"""Train transformer using contrastive learning approach"""
		
		# Convert to tensors
		X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
		y_tensor = torch.tensor(y, dtype=torch.long).to(device)
		
		# Create data loader
		dataset = TensorDataset(X_tensor, y_tensor)
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
		
		# Optimizer and loss
		optimizer = torch.optim.AdamW(self.transformer.parameters(), lr=lr, weight_decay=1e-4)
		criterion = nn.CrossEntropyLoss()
		
		# Move transformer to device
		self.transformer.to(device)
		self.transformer.train()
		
		# Training loop
		for epoch in range(epochs):
			total_loss = 0.0
			
			for batch_X, batch_y in dataloader:
				optimizer.zero_grad()
				
				# Extract features
				features = self.transformer(batch_X)
				
				# Simple classification head for training
				logits = torch.randn(features.size(0), 2).to(device)  # Placeholder
				
				# Use contrastive loss on features
				loss = self._contrastive_loss(features, batch_y)
				
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
				optimizer.step()
				
				total_loss += loss.item()
			
			if (epoch + 1) % 10 == 0:
				print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
	
	def _contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		"""Contrastive loss to learn discriminative features"""
		
		# Normalize features
		features_norm = F.normalize(features, p=2, dim=1)
		
		# Compute similarity matrix
		similarity = torch.mm(features_norm, features_norm.t())
		
		# Create positive/negative masks
		labels_expanded = labels.unsqueeze(1)
		positive_mask = (labels_expanded == labels_expanded.t()).float()
		negative_mask = 1 - positive_mask
		
		# Remove diagonal
		positive_mask.fill_diagonal_(0)
		
		# Contrastive loss
		pos_loss = -torch.log(torch.sigmoid(similarity) + 1e-8) * positive_mask
		neg_loss = -torch.log(1 - torch.sigmoid(similarity) + 1e-8) * negative_mask
		
		loss = (pos_loss.sum() + neg_loss.sum()) / (positive_mask.sum() + negative_mask.sum())
		
		return loss
	
	def _extract_features(self, X: np.ndarray, device: str) -> np.ndarray:
		"""Extract features using trained transformer"""
		
		self.transformer.eval()
		features_list = []
		
		with torch.no_grad():
			X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
			
			# Process in batches
			batch_size = 64
			for i in range(0, len(X_tensor), batch_size):
				batch = X_tensor[i:i+batch_size]
				features = self.transformer(batch)
				features_list.append(features.cpu().numpy())
		
		return np.vstack(features_list)
	
	def predict(self, X: np.ndarray, device: str = None) -> np.ndarray:
		"""Make predictions using the hybrid model"""
		
		if not self.is_fitted:
			raise ValueError("Model must be fitted before making predictions")
		
		device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		
		# Extract features
		features = self._extract_features(X, device)
		
		# Scale features
		features_scaled = self.scaler.transform(features)
		
		# Make predictions
		predictions = self.tree.predict(features_scaled)
		
		return predictions
	
	def predict_proba(self, X: np.ndarray, device: str = None) -> np.ndarray:
		"""Get prediction probabilities"""
		
		if not self.is_fitted:
			raise ValueError("Model must be fitted before making predictions")
		
		device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		
		# Extract features
		features = self._extract_features(X, device)
		
		# Scale features
		features_scaled = self.scaler.transform(features)
		
		# Get probabilities
		probabilities = self.tree.predict_proba(features_scaled)
		
		return probabilities
	
	def _get_feature_importance(self) -> Dict[str, float]:
		"""Get feature importance from decision tree"""
		
		if hasattr(self.tree, 'feature_importances_'):
			importance_dict = dict(zip(self.feature_names, self.tree.feature_importances_))
		else:
			# For ensemble methods, get average importance
			if hasattr(self.tree, 'estimators_'):
				avg_importance = np.mean([est.feature_importances_ for est in self.tree.estimators_], axis=0)
				importance_dict = dict(zip(self.feature_names, avg_importance))
			else:
				importance_dict = {name: 0.0 for name in self.feature_names}
		
		return importance_dict
	
	def _get_tree_depth(self) -> int:
		"""Get maximum depth of decision tree"""
		
		if hasattr(self.tree, 'tree_'):
			return self.tree.tree_.max_depth
		elif hasattr(self.tree, 'estimators_'):
			return max([est.tree_.max_depth for est in self.tree.estimators_])
		else:
			return 0
	
	def get_decision_rules(self, max_depth: int = 3) -> str:
		"""Get interpretable decision rules from the tree"""
		
		if self.tree_type == "single_tree":
			return export_text(self.tree, max_depth=max_depth, feature_names=self.feature_names)
		else:
			return "Decision rules available only for single decision tree. Use tree_type='single_tree' for interpretable rules."
	
	def save_model(self, path: str):
		"""Save the hybrid model"""
		
		model_data = {
			'transformer_state': self.transformer.state_dict(),
			'tree': self.tree,
			'scaler': self.scaler,
			'input_dim': self.input_dim,
			'seq_len': self.seq_len,
			'transformer_features': self.transformer_features,
			'tree_type': self.tree_type,
			'feature_names': self.feature_names,
			'is_fitted': self.is_fitted
		}
		
		torch.save(model_data, path)
	
	def load_model(self, path: str):
		"""Load the hybrid model"""
		
		model_data = torch.load(path, map_location='cpu')
		
		self.transformer.load_state_dict(model_data['transformer_state'])
		self.tree = model_data['tree']
		self.scaler = model_data['scaler']
		self.input_dim = model_data['input_dim']
		self.seq_len = model_data['seq_len']
		self.transformer_features = model_data['transformer_features']
		self.tree_type = model_data['tree_type']
		self.feature_names = model_data['feature_names']
		self.is_fitted = model_data['is_fitted']


def train_hybrid_model(X: np.ndarray, y: np.ndarray, 
					  tree_type: str = "random_forest",
					  epochs: int = 50, batch_size: int = 64, lr: float = 1e-3) -> HybridTransformerTree:
	"""
	Train hybrid transformer + decision tree model
	"""
	
	model = HybridTransformerTree(
		input_dim=X.shape[-1],
		seq_len=X.shape[1],
		tree_type=tree_type
	)
	
	results = model.train(X, y, epochs=epochs, batch_size=batch_size, lr=lr)
	
	print(f"Training completed:")
	print(f"Accuracy: {results['training_accuracy']:.4f}")
	print(f"Tree Depth: {results['tree_depth']}")
	print(f"Feature Importance: {results['feature_importance']}")
	
	return model


def evaluate_hybrid_model(model: HybridTransformerTree, X: np.ndarray, y: np.ndarray) -> Dict:
	"""
	Evaluate hybrid model performance
	"""
	
	predictions = model.predict(X)
	probabilities = model.predict_proba(X)
	
	accuracy = accuracy_score(y, predictions)
	
	# Get feature importance
	feature_importance = model._get_feature_importance()
	
	# Get decision rules if possible
	decision_rules = model.get_decision_rules()
	
	return {
		"accuracy": accuracy,
		"feature_importance": feature_importance,
		"decision_rules": decision_rules,
		"predictions": predictions.tolist(),
		"probabilities": probabilities.tolist()
	}