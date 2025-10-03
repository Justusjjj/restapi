import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


class LSTMClassifier(nn.Module):
	def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
		super().__init__()
		self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
		self.head = nn.Sequential(
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, 2)
		)

	def forward(self, x):
		out, _ = self.lstm(x)
		last = out[:, -1, :]
		return self.head(last)


def train_model(X: np.ndarray, y: np.ndarray, epochs: int = 25, batch_size: int = 64, lr: float = 1e-3, device: str = None):
	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	X_tensor = torch.tensor(X, dtype=torch.float32)
	y_tensor = torch.tensor(y, dtype=torch.long)

	# Class weights for imbalance
	classes, counts = np.unique(y, return_counts=True)
	total = counts.sum()
	weights = torch.tensor([total / (2 * c) for c in counts], dtype=torch.float32)

	dataset = TensorDataset(X_tensor, y_tensor)
	sampler = WeightedRandomSampler(weights=weights[y_tensor], num_samples=len(y_tensor), replacement=True)
	dl = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

	model = LSTMClassifier(input_dim=X.shape[-1]).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
	criterion = nn.CrossEntropyLoss(weight=weights.to(device))

	best_loss = float("inf")
	patience = 5
	wait = 0

	for epoch in range(epochs):
		model.train()
		total_loss = 0.0
		for xb, yb in dl:
			xb = xb.to(device)
			yb = yb.to(device)
			optimizer.zero_grad()
			logits = model(xb)
			loss = criterion(logits, yb)
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			total_loss += loss.item()
		avg_loss = total_loss / max(1, len(dl))
		if avg_loss < best_loss - 1e-4:
			best_loss = avg_loss
			wait = 0
		else:
			wait += 1
		if wait >= patience:
			break
	return model


def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray, device: str = None):
	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	model.eval()
	with torch.no_grad():
		X_t = torch.tensor(X, dtype=torch.float32).to(device)
		logits = model(X_t)
		pred = torch.argmax(logits, dim=1).cpu().numpy()
		acc = (pred == y).mean() if len(y) else 0.0
		return {"accuracy": float(acc)}