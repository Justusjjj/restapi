import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import gym
from gym import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from streamlined_features import StreamlinedICTFeatures, get_economic_calendar_proxy


class TradingEnvironment(gym.Env):
	"""
	RL Trading Environment for PPO training.
	State: Multi-timeframe features + ICT signals + economic calendar
	Action: 0=HOLD, 1=BUY, 2=SELL
	Reward: Risk-adjusted returns with transaction costs
	"""
	
	def __init__(self, data: pd.DataFrame, symbol: str, initial_balance: float = 10000.0):
		super().__init__()
		
		self.data = data
		self.symbol = symbol
		self.initial_balance = initial_balance
		self.balance = initial_balance
		self.position = 0.0  # Current position size
		self.entry_price = 0.0
		self.current_step = 0
		self.max_steps = len(data) - 100  # Leave room for lookback
		
		# Feature dimensions
		self.feature_dim = 25  # Will be calculated dynamically
		self.ict_features = StreamlinedICTFeatures()
		
		# Action space: 0=HOLD, 1=BUY, 2=SELL
		self.action_space = spaces.Discrete(3)
		
		# Observation space: normalized features
		self.observation_space = spaces.Box(
			low=-1.0, high=1.0, shape=(self.feature_dim,), dtype=np.float32
		)
		
		# Trading parameters
		self.transaction_cost = 0.0001  # 1 pip spread
		self.max_position_size = 1.0
		self.risk_per_trade = 0.02
		
		self.reset()
	
	def reset(self):
		"""Reset environment to initial state"""
		self.balance = self.initial_balance
		self.position = 0.0
		self.entry_price = 0.0
		self.current_step = 100  # Start after warmup period
		
		return self._get_observation()
	
	def step(self, action):
		"""Execute action and return next state, reward, done, info"""
		
		# Get current price
		current_price = self.data['close'].iloc[self.current_step]
		
		# Execute action
		reward = self._execute_action(action, current_price)
		
		# Move to next step
		self.current_step += 1
		
		# Check if episode is done
		done = self.current_step >= self.max_steps
		
		# Get next observation
		next_obs = self._get_observation() if not done else np.zeros(self.feature_dim)
		
		# Info dictionary
		info = {
			"balance": self.balance,
			"position": self.position,
			"step": self.current_step,
			"price": current_price
		}
		
		return next_obs, reward, done, info
	
	def _execute_action(self, action: int, current_price: float) -> float:
		"""Execute trading action and return reward"""
		
		# Close existing position if any
		if self.position != 0:
			pnl = self._close_position(current_price)
		else:
			pnl = 0.0
		
		# Execute new action
		if action == 1:  # BUY
			self._open_position(1.0, current_price)
		elif action == 2:  # SELL
			self._open_position(-1.0, current_price)
		# action == 0: HOLD (do nothing)
		
		# Calculate reward
		reward = self._calculate_reward(pnl, current_price)
		
		return reward
	
	def _open_position(self, size: float, price: float):
		"""Open a new position"""
		# Calculate position size based on risk
		risk_amount = self.balance * self.risk_per_trade
		atr_val = self._calculate_atr()
		stop_distance = max(atr_val * 1.5, price * 0.01)  # 1% minimum stop
		
		position_size = min(risk_amount / stop_distance, self.max_position_size)
		position_size = max(position_size, 0.01)  # Minimum position size
		
		self.position = size * position_size
		self.entry_price = price
		
		# Deduct transaction cost
		self.balance -= abs(self.position) * price * self.transaction_cost
	
	def _close_position(self, current_price: float) -> float:
		"""Close current position and return P&L"""
		if self.position == 0:
			return 0.0
		
		# Calculate P&L
		if self.position > 0:  # Long position
			pnl = (current_price - self.entry_price) * abs(self.position)
		else:  # Short position
			pnl = (self.entry_price - current_price) * abs(self.position)
		
		# Deduct transaction cost
		self.balance -= abs(self.position) * current_price * self.transaction_cost
		
		# Update balance
		self.balance += pnl
		
		# Reset position
		self.position = 0.0
		self.entry_price = 0.0
		
		return pnl
	
	def _calculate_reward(self, pnl: float, current_price: float) -> float:
		"""Calculate reward for the action"""
		
		# Base reward from P&L
		reward = pnl / self.initial_balance
		
		# Penalty for holding positions too long
		if self.position != 0:
			reward -= 0.001  # Small penalty for holding
		
		# Bonus for high balance
		balance_ratio = self.balance / self.initial_balance
		if balance_ratio > 1.1:  # 10% profit
			reward += 0.1
		elif balance_ratio < 0.9:  # 10% loss
			reward -= 0.1
		
		return reward
	
	def _calculate_atr(self, period: int = 14) -> float:
		"""Calculate Average True Range"""
		if self.current_step < period:
			return 0.01
		
		recent_data = self.data.iloc[self.current_step-period:self.current_step]
		
		high_low = recent_data['high'] - recent_data['low']
		high_close = (recent_data['high'] - recent_data['close'].shift()).abs()
		low_close = (recent_data['low'] - recent_data['close'].shift()).abs()
		
		tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
		atr_val = tr.mean()
		
		return float(atr_val) if not pd.isna(atr_val) else 0.01
	
	def _get_observation(self) -> np.ndarray:
		"""Get current observation state"""
		
		if self.current_step >= len(self.data):
			return np.zeros(self.feature_dim)
		
		# Get recent data for feature calculation
		recent_data = self.data.iloc[max(0, self.current_step-100):self.current_step+1]
		
		# Calculate features
		features = []
		
		# Price features
		current_price = recent_data['close'].iloc[-1]
		features.extend([
			recent_data['close'].pct_change().iloc[-1] or 0,  # Returns
			recent_data['close'].pct_change().rolling(5).mean().iloc[-1] or 0,  # MA returns
			recent_data['close'].pct_change().rolling(20).std().iloc[-1] or 0,  # Volatility
		])
		
		# Technical indicators
		sma_20 = recent_data['close'].rolling(20).mean().iloc[-1] or current_price
		sma_50 = recent_data['close'].rolling(50).mean().iloc[-1] or current_price
		features.extend([
			(current_price - sma_20) / sma_20,  # Price vs SMA20
			(current_price - sma_50) / sma_50,  # Price vs SMA50
			(sma_20 - sma_50) / sma_50,  # SMA20 vs SMA50
		])
		
		# ICT features
		ict_signal = self.ict_features.get_composite_signal(recent_data)
		features.extend([
			ict_signal['composite_score'],
			ict_signal['confidence'],
			1.0 if ict_signal['signal'] == 'BULLISH' else -1.0 if ict_signal['signal'] == 'BEARISH' else 0.0,
		])
		
		# Order block features
		ob_data = ict_signal['order_blocks']
		features.extend([
			ob_data['bullish_obs'] / 10.0,  # Normalize
			ob_data['bearish_obs'] / 10.0,
			ob_data['ob_strength'],
		])
		
		# Fair value gap features
		fvg_data = ict_signal['fair_value_gaps']
		features.extend([
			fvg_data['bullish_fvgs'] / 10.0,
			fvg_data['bearish_fvgs'] / 10.0,
			fvg_data['fvg_strength'],
		])
		
		# Liquidity sweep features
		liq_data = ict_signal['liquidity_sweeps']
		features.extend([
			1.0 if liq_data['sweep_high'] else 0.0,
			1.0 if liq_data['sweep_low'] else 0.0,
			liq_data['sweep_strength'],
		])
		
		# Market structure features
		struct_data = ict_signal['market_structure']
		features.extend([
			1.0 if struct_data['break_bullish'] else 0.0,
			1.0 if struct_data['break_bearish'] else 0.0,
			struct_data['structure_strength'],
		])
		
		# Economic calendar features
		timestamp = recent_data.index[-1]
		econ_data = get_economic_calendar_proxy(self.symbol, timestamp)
		features.extend([
			econ_data['impact_score'],
			1.0 if econ_data['high_impact_time'] else 0.0,
		])
		
		# Account features
		features.extend([
			self.balance / self.initial_balance - 1.0,  # Balance ratio
			self.position,  # Current position
		])
		
		# Pad or truncate to feature_dim
		while len(features) < self.feature_dim:
			features.append(0.0)
		features = features[:self.feature_dim]
		
		return np.array(features, dtype=np.float32)


def train_rl_agent(data: pd.DataFrame, symbol: str, 
				  total_timesteps: int = 100000,
				  learning_rate: float = 3e-4,
				  n_steps: int = 2048,
				  batch_size: int = 64,
				  n_epochs: int = 10,
				  model_save_path: str = "ppo_trading_model") -> PPO:
	"""
	Train PPO agent on trading environment
	"""
	
	# Create environment
	env = TradingEnvironment(data, symbol)
	
	# Create PPO model
	model = PPO(
		"MlpPolicy",
		env,
		learning_rate=learning_rate,
		n_steps=n_steps,
		batch_size=batch_size,
		n_epochs=n_epochs,
		gamma=0.99,
		gae_lambda=0.95,
		clip_range=0.2,
		ent_coef=0.01,
		vf_coef=0.5,
		max_grad_norm=0.5,
		verbose=1
	)
	
	# Train the model
	model.learn(total_timesteps=total_timesteps)
	
	# Save the model
	model.save(model_save_path)
	
	return model


def evaluate_rl_agent(model: PPO, data: pd.DataFrame, symbol: str, 
					 num_episodes: int = 10) -> Dict:
	"""
	Evaluate trained RL agent
	"""
	
	env = TradingEnvironment(data, symbol)
	
	episode_rewards = []
	episode_returns = []
	
	for episode in range(num_episodes):
		obs = env.reset()
		episode_reward = 0
		
		while True:
			action, _ = model.predict(obs, deterministic=True)
			obs, reward, done, info = env.step(action)
			episode_reward += reward
			
			if done:
				break
		
		episode_rewards.append(episode_reward)
		episode_returns.append((info['balance'] - env.initial_balance) / env.initial_balance)
	
	return {
		"avg_reward": np.mean(episode_rewards),
		"std_reward": np.std(episode_rewards),
		"avg_return": np.mean(episode_returns),
		"std_return": np.std(episode_returns),
		"max_return": np.max(episode_returns),
		"min_return": np.min(episode_returns)
	}