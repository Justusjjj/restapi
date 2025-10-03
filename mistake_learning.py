import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import MetaTrader5 as mt5


@dataclass
class TradingMistake:
	"""Record of a trading mistake for AI learning"""
	timestamp: datetime
	symbol: str
	action: str  # BUY/SELL/HOLD
	price: float
	context: Dict  # Market conditions, signals, etc.
	mistake_type: str  # "false_signal", "wrong_direction", "poor_timing", "risk_management"
	outcome: str  # "loss", "missed_profit", "drawdown"
	severity: float  # 0.0 to 1.0
	lesson: str  # What the AI learned
	features_at_mistake: Dict  # Feature values when mistake occurred


class MistakeLearningSystem:
	"""
	AI system that learns from trading mistakes and adapts behavior.
	Like a software developer learning from bugs and evolving.
	"""
	
	def __init__(self, mistake_db_path: str = "mistakes_db.json"):
		self.mistake_db_path = mistake_db_path
		self.mistakes: List[TradingMistake] = []
		self.learned_patterns: Dict = {}
		self.adaptation_rules: Dict = {}
		self.load_mistakes()
		
		# Learning parameters
		self.min_mistakes_for_pattern = 3
		self.pattern_confidence_threshold = 0.7
		self.adaptation_strength = 0.1
	
	def record_mistake(self, symbol: str, action: str, price: float, 
					  context: Dict, mistake_type: str, outcome: str, 
					  severity: float, features: Dict) -> str:
		"""
		Record a trading mistake and generate a lesson
		"""
		
		# Generate lesson based on mistake type and context
		lesson = self._generate_lesson(mistake_type, outcome, context, features)
		
		# Create mistake record
		mistake = TradingMistake(
			timestamp=datetime.now(),
			symbol=symbol,
			action=action,
			price=price,
			context=context,
			mistake_type=mistake_type,
			outcome=outcome,
			severity=severity,
			lesson=lesson,
			features_at_mistake=features
		)
		
		# Add to database
		self.mistakes.append(mistake)
		self.save_mistakes()
		
		# Learn from this mistake
		self._learn_from_mistake(mistake)
		
		return lesson
	
	def _generate_lesson(self, mistake_type: str, outcome: str, 
						context: Dict, features: Dict) -> str:
		"""Generate a specific lesson from the mistake"""
		
		lessons = {
			"false_signal": {
				"loss": "Avoid trading when ICT signals conflict with higher timeframe bias",
				"missed_profit": "Wait for stronger confirmation before entering trades",
				"drawdown": "Reduce position size when signal confidence is low"
			},
			"wrong_direction": {
				"loss": "Check higher timeframe trend before taking counter-trend positions",
				"missed_profit": "Follow the dominant trend instead of fighting it",
				"drawdown": "Use trend-following strategies in trending markets"
			},
			"poor_timing": {
				"loss": "Wait for better entry points with lower risk",
				"missed_profit": "Enter positions closer to key support/resistance levels",
				"drawdown": "Avoid trading during low liquidity periods"
			},
			"risk_management": {
				"loss": "Always use stop losses and position sizing",
				"missed_profit": "Take partial profits at key levels",
				"drawdown": "Reduce position size during high volatility"
			}
		}
		
		base_lesson = lessons.get(mistake_type, {}).get(outcome, "Review trading decision process")
		
		# Add context-specific advice
		if context.get("volatility", 0) > 0.02:
			base_lesson += ". High volatility detected - use smaller position sizes."
		
		if context.get("news_impact", 0) > 0.7:
			base_lesson += ". High news impact - avoid trading during major events."
		
		if features.get("confidence", 0) < 0.6:
			base_lesson += ". Low confidence signal - wait for stronger confirmation."
		
		return base_lesson
	
	def _learn_from_mistake(self, mistake: TradingMistake):
		"""Learn patterns from mistakes and adapt behavior"""
		
		# Update mistake patterns
		pattern_key = f"{mistake.mistake_type}_{mistake.symbol}"
		if pattern_key not in self.learned_patterns:
			self.learned_patterns[pattern_key] = {
				"count": 0,
				"total_severity": 0.0,
				"contexts": [],
				"features": []
			}
		
		pattern = self.learned_patterns[pattern_key]
		pattern["count"] += 1
		pattern["total_severity"] += mistake.severity
		pattern["contexts"].append(mistake.context)
		pattern["features"].append(mistake.features_at_mistake)
		
		# Generate adaptation rules if enough mistakes
		if pattern["count"] >= self.min_mistakes_for_pattern:
			self._generate_adaptation_rule(pattern_key, pattern)
	
	def _generate_adaptation_rule(self, pattern_key: str, pattern: Dict):
		"""Generate adaptation rules based on mistake patterns"""
		
		avg_severity = pattern["total_severity"] / pattern["count"]
		
		if avg_severity > self.pattern_confidence_threshold:
			# High severity pattern - create strong adaptation rule
			rule = {
				"pattern": pattern_key,
				"action": "avoid",
				"strength": min(avg_severity * 2, 1.0),
				"conditions": self._extract_common_conditions(pattern["contexts"]),
				"created_at": datetime.now().isoformat()
			}
			
			self.adaptation_rules[pattern_key] = rule
			print(f"🤖 AI Learning: Created adaptation rule for {pattern_key} - AVOID with strength {rule['strength']:.2f}")
	
	def _extract_common_conditions(self, contexts: List[Dict]) -> Dict:
		"""Extract common conditions from mistake contexts"""
		
		conditions = {}
		
		# Analyze volatility patterns
		volatilities = [ctx.get("volatility", 0) for ctx in contexts]
		if volatilities:
			conditions["avg_volatility"] = np.mean(volatilities)
			conditions["high_volatility_threshold"] = np.percentile(volatilities, 75)
		
		# Analyze confidence patterns
		confidences = [ctx.get("confidence", 0) for ctx in contexts]
		if confidences:
			conditions["avg_confidence"] = np.mean(confidences)
			conditions["low_confidence_threshold"] = np.percentile(confidences, 25)
		
		# Analyze time patterns
		hours = [ctx.get("hour", 12) for ctx in contexts]
		if hours:
			conditions["common_hours"] = list(set(hours))
		
		return conditions
	
	def apply_adaptations(self, symbol: str, action: str, context: Dict, 
						features: Dict) -> Tuple[bool, str, float]:
		"""
		Apply learned adaptations to trading decisions
		Returns: (should_trade, reason, confidence_adjustment)
		"""
		
		# Check for applicable adaptation rules
		for pattern_key, rule in self.adaptation_rules.items():
			if symbol in pattern_key and self._matches_conditions(context, rule["conditions"]):
				
				if rule["action"] == "avoid":
					confidence_penalty = rule["strength"]
					reason = f"AI learned to avoid this pattern (strength: {confidence_penalty:.2f})"
					return False, reason, -confidence_penalty
				
				elif rule["action"] == "caution":
					confidence_penalty = rule["strength"] * 0.5
					reason = f"AI learned to be cautious with this pattern (penalty: {confidence_penalty:.2f})"
					return True, reason, -confidence_penalty
		
		return True, "No adaptation rules apply", 0.0
	
	def _matches_conditions(self, context: Dict, conditions: Dict) -> bool:
		"""Check if current context matches learned mistake conditions"""
		
		matches = 0
		total_checks = 0
		
		# Check volatility
		if "avg_volatility" in conditions:
			total_checks += 1
			if context.get("volatility", 0) > conditions["high_volatility_threshold"]:
				matches += 1
		
		# Check confidence
		if "avg_confidence" in conditions:
			total_checks += 1
			if context.get("confidence", 0) < conditions["low_confidence_threshold"]:
				matches += 1
		
		# Check time
		if "common_hours" in conditions:
			total_checks += 1
			current_hour = datetime.now().hour
			if current_hour in conditions["common_hours"]:
				matches += 1
		
		# Match if majority of conditions are met
		return matches > total_checks / 2 if total_checks > 0 else False
	
	def get_learning_summary(self) -> Dict:
		"""Get summary of AI learning progress"""
		
		total_mistakes = len(self.mistakes)
		unique_patterns = len(self.learned_patterns)
		adaptation_rules = len(self.adaptation_rules)
		
		# Calculate learning effectiveness
		recent_mistakes = [m for m in self.mistakes if m.timestamp > datetime.now() - timedelta(days=7)]
		old_mistakes = [m for m in self.mistakes if m.timestamp <= datetime.now() - timedelta(days=7)]
		
		recent_avg_severity = np.mean([m.severity for m in recent_mistakes]) if recent_mistakes else 0
		old_avg_severity = np.mean([m.severity for m in old_mistakes]) if old_mistakes else 0
		
		learning_effectiveness = max(0, (old_avg_severity - recent_avg_severity) / max(old_avg_severity, 0.01))
		
		return {
			"total_mistakes": total_mistakes,
			"unique_patterns": unique_patterns,
			"adaptation_rules": adaptation_rules,
			"learning_effectiveness": learning_effectiveness,
			"recent_mistakes": len(recent_mistakes),
			"avg_severity_recent": recent_avg_severity,
			"avg_severity_historical": old_avg_severity,
			"ai_evolution_status": "Learning" if learning_effectiveness > 0.1 else "Adapting" if learning_effectiveness > 0 else "Stable"
		}
	
	def save_mistakes(self):
		"""Save mistakes to JSON file"""
		
		mistakes_data = []
		for mistake in self.mistakes:
			mistakes_data.append({
				"timestamp": mistake.timestamp.isoformat(),
				"symbol": mistake.symbol,
				"action": mistake.action,
				"price": mistake.price,
				"context": mistake.context,
				"mistake_type": mistake.mistake_type,
				"outcome": mistake.outcome,
				"severity": mistake.severity,
				"lesson": mistake.lesson,
				"features_at_mistake": mistake.features_at_mistake
			})
		
		with open(self.mistake_db_path, 'w') as f:
			json.dump({
				"mistakes": mistakes_data,
				"learned_patterns": self.learned_patterns,
				"adaptation_rules": self.adaptation_rules
			}, f, indent=2)
	
	def load_mistakes(self):
		"""Load mistakes from JSON file"""
		
		if os.path.exists(self.mistake_db_path):
			try:
				with open(self.mistake_db_path, 'r') as f:
					data = json.load(f)
				
				# Load mistakes
				for mistake_data in data.get("mistakes", []):
					mistake = TradingMistake(
						timestamp=datetime.fromisoformat(mistake_data["timestamp"]),
						symbol=mistake_data["symbol"],
						action=mistake_data["action"],
						price=mistake_data["price"],
						context=mistake_data["context"],
						mistake_type=mistake_data["mistake_type"],
						outcome=mistake_data["outcome"],
						severity=mistake_data["severity"],
						lesson=mistake_data["lesson"],
						features_at_mistake=mistake_data["features_at_mistake"]
					)
					self.mistakes.append(mistake)
				
				# Load learned patterns and rules
				self.learned_patterns = data.get("learned_patterns", {})
				self.adaptation_rules = data.get("adaptation_rules", {})
				
				print(f"🤖 AI Learning: Loaded {len(self.mistakes)} mistakes and {len(self.adaptation_rules)} adaptation rules")
				
			except Exception as e:
				print(f"Error loading mistakes database: {e}")
				self.mistakes = []
				self.learned_patterns = {}
				self.adaptation_rules = {}


def get_demo_account_balance() -> float:
	"""
	Get actual balance from connected MT5 demo account.
	No need to hardcode 10000 if demo account is connected.
	"""
	try:
		if not mt5.initialize():
			print("MT5 not initialized, using default balance")
			return 10000.0
		
		account_info = mt5.account_info()
		if account_info is None:
			print("No account info available, using default balance")
			return 10000.0
		
		balance = float(account_info.balance)
		print(f"💰 Demo account balance: ${balance:.2f}")
		return balance
		
	except Exception as e:
		print(f"Error getting account balance: {e}, using default")
		return 10000.0


def analyze_trade_outcome(trade_result: Dict, expected_outcome: str, 
						market_context: Dict, features: Dict) -> Optional[TradingMistake]:
	"""
	Analyze trade outcome and identify mistakes for AI learning
	"""
	
	# Determine if this was a mistake
	mistake_type = None
	outcome = None
	severity = 0.0
	
	if trade_result.get("pnl", 0) < 0:
		# Loss occurred
		if expected_outcome == "profit":
			mistake_type = "wrong_direction"
			outcome = "loss"
			severity = abs(trade_result["pnl"]) / 1000  # Normalize by 1000
		elif expected_outcome == "hold":
			mistake_type = "false_signal"
			outcome = "loss"
			severity = abs(trade_result["pnl"]) / 1000
	
	elif trade_result.get("pnl", 0) > 0 and expected_outcome == "hold":
		# Missed profit
		mistake_type = "false_signal"
		outcome = "missed_profit"
		severity = trade_result["pnl"] / 1000
	
	# Check for risk management mistakes
	if trade_result.get("max_drawdown", 0) > 0.05:  # 5% drawdown
		mistake_type = "risk_management"
		outcome = "drawdown"
		severity = trade_result["max_drawdown"]
	
	if mistake_type:
		return TradingMistake(
			timestamp=datetime.now(),
			symbol=trade_result.get("symbol", "UNKNOWN"),
			action=trade_result.get("action", "UNKNOWN"),
			price=trade_result.get("price", 0.0),
			context=market_context,
			mistake_type=mistake_type,
			outcome=outcome,
			severity=min(severity, 1.0),
			lesson="",  # Will be generated by learning system
			features_at_mistake=features
		)
	
	return None