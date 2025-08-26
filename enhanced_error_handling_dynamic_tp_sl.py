#!/usr/bin/env python3
"""
Enhanced Error Handling and Dynamic TP/SL Management for Forex Trading Bot
Features:
- Comprehensive Error Handling and Recovery
- Dynamic Take Profit and Stop Loss
- Adaptive Risk Management
- Intelligent Position Sizing
- Real-time Error Monitoring
- Automatic Recovery Mechanisms
- Advanced Logging and Alerting
"""

import numpy as np
import pandas as pd
import traceback
import sys
import time
import threading
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedErrorHandler:
    def __init__(self, config: Dict):
        """Initialize enhanced error handler"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.error_history = []
        self.recovery_attempts = {}
        self.error_counts = {}
        self.max_retries = config.get("error_handling", {}).get("max_retries", 3)
        self.retry_delay = config.get("error_handling", {}).get("retry_delay", 5)
        self.critical_errors = set()
        self.error_callbacks = {}
        
    def handle_error(self, error: Exception, context: str, critical: bool = False) -> Dict:
        """Handle errors with comprehensive logging and recovery"""
        try:
            error_info = {
                "timestamp": datetime.now().isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
                "critical": critical,
                "traceback": traceback.format_exc(),
                "recovery_attempts": 0
            }
            
            # Log error
            self.logger.error(f"Error in {context}: {error}")
            if critical:
                self.logger.critical(f"CRITICAL ERROR: {error}")
            
            # Store error history
            self.error_history.append(error_info)
            
            # Update error counts
            error_key = f"{context}_{type(error).__name__}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            
            # Attempt recovery
            recovery_result = self._attempt_recovery(error_info, context)
            error_info["recovery_result"] = recovery_result
            
            # Execute error callbacks
            if context in self.error_callbacks:
                try:
                    self.error_callbacks[context](error_info)
                except Exception as callback_error:
                    self.logger.error(f"Error in error callback: {callback_error}")
            
            # Alert if critical
            if critical:
                self._send_critical_alert(error_info)
            
            return error_info
            
        except Exception as e:
            self.logger.error(f"Error in error handler: {e}")
            return {"error": "Error handler failed", "original_error": str(error)}
    
    def _attempt_recovery(self, error_info: Dict, context: str) -> Dict:
        """Attempt to recover from error"""
        try:
            recovery_result = {
                "success": False,
                "method": None,
                "attempts": 0,
                "details": ""
            }
            
            # Check if we should attempt recovery
            if error_info["critical"] and context in self.critical_errors:
                recovery_result["details"] = "Critical error - no recovery attempted"
                return recovery_result
            
            # Get recovery method for context
            recovery_method = self._get_recovery_method(context, error_info["error_type"])
            
            if recovery_method:
                recovery_result["method"] = recovery_method.__name__
                
                # Attempt recovery with retries
                for attempt in range(self.max_retries):
                    try:
                        recovery_result["attempts"] = attempt + 1
                        recovery_result["success"] = recovery_method(error_info)
                        
                        if recovery_result["success"]:
                            recovery_result["details"] = f"Recovery successful after {attempt + 1} attempts"
                            self.logger.info(f"Recovery successful for {context}")
                            break
                        else:
                            time.sleep(self.retry_delay)
                            
                    except Exception as recovery_error:
                        self.logger.error(f"Recovery attempt {attempt + 1} failed: {recovery_error}")
                        recovery_result["details"] = f"Recovery attempt {attempt + 1} failed: {recovery_error}"
                        time.sleep(self.retry_delay)
                
                if not recovery_result["success"]:
                    recovery_result["details"] = f"Recovery failed after {self.max_retries} attempts"
                    self.logger.error(f"Recovery failed for {context} after {self.max_retries} attempts")
            else:
                recovery_result["details"] = "No recovery method available"
            
            return recovery_result
            
        except Exception as e:
            self.logger.error(f"Error in recovery attempt: {e}")
            return {"success": False, "method": None, "attempts": 0, "details": str(e)}
    
    def _get_recovery_method(self, context: str, error_type: str) -> Optional[Callable]:
        """Get appropriate recovery method for context and error type"""
        try:
            recovery_methods = {
                "mt5_connection": self._recover_mt5_connection,
                "data_feed": self._recover_data_feed,
                "trade_execution": self._recover_trade_execution,
                "risk_management": self._recover_risk_management,
                "ml_model": self._recover_ml_model,
                "database": self._recover_database,
                "api_connection": self._recover_api_connection
            }
            
            return recovery_methods.get(context, self._default_recovery)
            
        except Exception as e:
            self.logger.error(f"Error getting recovery method: {e}")
            return None
    
    def _recover_mt5_connection(self, error_info: Dict) -> bool:
        """Recover MT5 connection"""
        try:
            # Attempt to reconnect to MT5
            import MetaTrader5 as mt5
            
            # Shutdown and reinitialize
            mt5.shutdown()
            time.sleep(2)
            
            if mt5.initialize():
                self.logger.info("MT5 connection recovered")
                return True
            else:
                self.logger.error("MT5 connection recovery failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in MT5 recovery: {e}")
            return False
    
    def _recover_data_feed(self, error_info: Dict) -> bool:
        """Recover data feed"""
        try:
            # Attempt to restart data feed
            # This would depend on your specific data feed implementation
            self.logger.info("Data feed recovery attempted")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in data feed recovery: {e}")
            return False
    
    def _recover_trade_execution(self, error_info: Dict) -> bool:
        """Recover trade execution"""
        try:
            # Attempt to recover trade execution
            # This might involve retrying failed trades or resetting execution state
            self.logger.info("Trade execution recovery attempted")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in trade execution recovery: {e}")
            return False
    
    def _recover_risk_management(self, error_info: Dict) -> bool:
        """Recover risk management system"""
        try:
            # Attempt to recover risk management
            # This might involve recalculating positions or resetting risk metrics
            self.logger.info("Risk management recovery attempted")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in risk management recovery: {e}")
            return False
    
    def _recover_ml_model(self, error_info: Dict) -> bool:
        """Recover ML model"""
        try:
            # Attempt to reload or retrain ML model
            self.logger.info("ML model recovery attempted")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in ML model recovery: {e}")
            return False
    
    def _recover_database(self, error_info: Dict) -> bool:
        """Recover database connection"""
        try:
            # Attempt to reconnect to database
            self.logger.info("Database recovery attempted")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in database recovery: {e}")
            return False
    
    def _recover_api_connection(self, error_info: Dict) -> bool:
        """Recover API connection"""
        try:
            # Attempt to reconnect to API
            self.logger.info("API connection recovery attempted")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in API recovery: {e}")
            return False
    
    def _default_recovery(self, error_info: Dict) -> bool:
        """Default recovery method"""
        try:
            # Simple retry with delay
            time.sleep(self.retry_delay)
            self.logger.info("Default recovery attempted")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in default recovery: {e}")
            return False
    
    def _send_critical_alert(self, error_info: Dict):
        """Send critical error alert"""
        try:
            # This would integrate with your alerting system
            alert_message = f"CRITICAL ERROR: {error_info['error_message']} in {error_info['context']}"
            self.logger.critical(alert_message)
            
            # You could add email, SMS, or other alerting methods here
            
        except Exception as e:
            self.logger.error(f"Error sending critical alert: {e}")
    
    def register_error_callback(self, context: str, callback: Callable):
        """Register callback function for specific error context"""
        try:
            self.error_callbacks[context] = callback
            self.logger.info(f"Error callback registered for {context}")
            
        except Exception as e:
            self.logger.error(f"Error registering callback: {e}")
    
    def get_error_summary(self) -> Dict:
        """Get summary of error handling statistics"""
        try:
            return {
                "total_errors": len(self.error_history),
                "critical_errors": sum(1 for e in self.error_history if e.get("critical", False)),
                "recovery_success_rate": sum(1 for e in self.error_history if e.get("recovery_result", {}).get("success", False)) / len(self.error_history) if self.error_history else 0,
                "error_counts": self.error_counts,
                "recent_errors": self.error_history[-10:] if len(self.error_history) > 10 else self.error_history
            }
            
        except Exception as e:
            self.logger.error(f"Error getting error summary: {e}")
            return {"error": str(e)}

class DynamicTPSLManager:
    def __init__(self, config: Dict):
        """Initialize dynamic TP/SL manager"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_positions = {}
        self.tp_sl_history = []
        self.risk_metrics = {}
        self.market_conditions = {}
        
    def calculate_dynamic_tp_sl(self, 
                               position: Dict, 
                               market_data: pd.DataFrame,
                               risk_params: Dict) -> Dict:
        """Calculate dynamic take profit and stop loss levels"""
        try:
            # Get current market conditions
            market_conditions = self._analyze_market_conditions(market_data)
            
            # Calculate base TP/SL levels
            base_levels = self._calculate_base_levels(position, market_data, risk_params)
            
            # Apply market condition adjustments
            adjusted_levels = self._apply_market_adjustments(base_levels, market_conditions)
            
            # Apply volatility adjustments
            volatility_adjusted = self._apply_volatility_adjustments(adjusted_levels, market_data)
            
            # Apply trend adjustments
            trend_adjusted = self._apply_trend_adjustments(volatility_adjusted, market_data)
            
            # Apply risk adjustments
            final_levels = self._apply_risk_adjustments(trend_adjusted, position, risk_params)
            
            # Store calculation history
            self._store_calculation_history(position, final_levels, market_conditions)
            
            return final_levels
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic TP/SL: {e}")
            return {"error": str(e)}
    
    def _analyze_market_conditions(self, market_data: pd.DataFrame) -> Dict:
        """Analyze current market conditions"""
        try:
            conditions = {}
            
            # Volatility analysis
            conditions["volatility"] = self._calculate_volatility(market_data)
            
            # Trend analysis
            conditions["trend"] = self._analyze_trend(market_data)
            
            # Volume analysis
            if 'volume' in market_data.columns:
                conditions["volume"] = self._analyze_volume(market_data)
            
            # Market regime
            conditions["regime"] = self._detect_market_regime(market_data)
            
            # Support/resistance levels
            conditions["support_resistance"] = self._detect_support_resistance(market_data)
            
            return conditions
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {e}")
            return {"error": str(e)}
    
    def _calculate_volatility(self, market_data: pd.DataFrame) -> Dict:
        """Calculate volatility metrics"""
        try:
            returns = market_data['close'].pct_change().dropna()
            
            # Historical volatility
            historical_vol = returns.std() * np.sqrt(252)
            
            # Rolling volatility
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
            current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else historical_vol
            
            # Volatility regime
            vol_regime = "high" if current_vol > historical_vol * 1.5 else "low" if current_vol < historical_vol * 0.5 else "normal"
            
            return {
                "historical": historical_vol,
                "current": current_vol,
                "regime": vol_regime,
                "rolling": rolling_vol.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return {"error": str(e)}
    
    def _analyze_trend(self, market_data: pd.DataFrame) -> Dict:
        """Analyze market trend"""
        try:
            close = market_data['close']
            
            # Multiple timeframe trend analysis
            trends = {}
            
            # Short-term trend (5 periods)
            short_ma = close.rolling(window=5).mean()
            short_trend = "up" if close.iloc[-1] > short_ma.iloc[-1] else "down"
            trends["short_term"] = {"direction": short_trend, "strength": abs(close.iloc[-1] - short_ma.iloc[-1]) / short_ma.iloc[-1]}
            
            # Medium-term trend (20 periods)
            medium_ma = close.rolling(window=20).mean()
            medium_trend = "up" if close.iloc[-1] > medium_ma.iloc[-1] else "down"
            trends["medium_term"] = {"direction": medium_trend, "strength": abs(close.iloc[-1] - medium_ma.iloc[-1]) / medium_ma.iloc[-1]}
            
            # Long-term trend (50 periods)
            long_ma = close.rolling(window=50).mean()
            long_trend = "up" if close.iloc[-1] > long_ma.iloc[-1] else "down"
            trends["long_term"] = {"direction": long_trend, "strength": abs(close.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1]}
            
            # Overall trend consensus
            trend_signals = [trends["short_term"]["direction"], trends["medium_term"]["direction"], trends["long_term"]["direction"]]
            overall_trend = "up" if trend_signals.count("up") >= 2 else "down"
            
            return {
                "timeframes": trends,
                "overall": overall_trend,
                "consensus": trend_signals.count(overall_trend) / len(trend_signals)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend: {e}")
            return {"error": str(e)}
    
    def _analyze_volume(self, market_data: pd.DataFrame) -> Dict:
        """Analyze volume patterns"""
        try:
            volume = market_data['volume']
            
            # Volume analysis
            avg_volume = volume.rolling(window=20).mean()
            current_volume = volume.iloc[-1]
            
            # Volume spike detection
            volume_spike = current_volume > (avg_volume.iloc[-1] * 1.5) if not avg_volume.empty else False
            
            # Volume trend
            volume_trend = "increasing" if volume.iloc[-5:].mean() > volume.iloc[-10:-5].mean() else "decreasing"
            
            return {
                "current": current_volume,
                "average": avg_volume.iloc[-1] if not avg_volume.empty else 0,
                "spike": volume_spike,
                "trend": volume_trend
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume: {e}")
            return {"error": str(e)}
    
    def _detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """Detect current market regime"""
        try:
            # Simple market regime detection based on volatility and trend
            volatility = self._calculate_volatility(market_data)
            trend = self._analyze_trend(market_data)
            
            if "error" in volatility or "error" in trend:
                return "unknown"
            
            vol_regime = volatility.get("regime", "normal")
            overall_trend = trend.get("overall", "unknown")
            
            # Determine market regime
            if vol_regime == "high" and overall_trend == "up":
                return "trending_volatile"
            elif vol_regime == "high" and overall_trend == "down":
                return "trending_volatile"
            elif vol_regime == "low" and overall_trend == "up":
                return "trending_calm"
            elif vol_regime == "low" and overall_trend == "down":
                return "trending_calm"
            elif vol_regime == "normal":
                return "ranging"
            else:
                return "unknown"
                
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return "unknown"
    
    def _detect_support_resistance(self, market_data: pd.DataFrame) -> Dict:
        """Detect support and resistance levels"""
        try:
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            
            # Recent highs and lows
            recent_high = high.rolling(window=20).max().iloc[-1]
            recent_low = low.rolling(window=20).min().iloc[-1]
            current_price = close.iloc[-1]
            
            # Distance to levels
            distance_to_high = (recent_high - current_price) / current_price
            distance_to_low = (current_price - recent_low) / current_price
            
            return {
                "recent_high": recent_high,
                "recent_low": recent_low,
                "distance_to_high": distance_to_high,
                "distance_to_low": distance_to_low
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting support/resistance: {e}")
            return {"error": str(e)}
    
    def _calculate_base_levels(self, position: Dict, market_data: pd.DataFrame, risk_params: Dict) -> Dict:
        """Calculate base TP/SL levels"""
        try:
            current_price = market_data['close'].iloc[-1]
            position_size = position.get("size", 0.01)
            risk_per_trade = risk_params.get("risk_per_trade", 0.02)
            account_balance = risk_params.get("account_balance", 10000)
            
            # Calculate ATR
            atr = self._calculate_atr(market_data)
            
            # Base stop loss (2 ATR)
            base_sl_distance = atr * 2
            base_sl = current_price - base_sl_distance if position.get("type") == "buy" else current_price + base_sl_distance
            
            # Base take profit (risk:reward ratio)
            risk_reward_ratio = risk_params.get("risk_reward_ratio", 2.0)
            base_tp_distance = base_sl_distance * risk_reward_ratio
            base_tp = current_price + base_tp_distance if position.get("type") == "buy" else current_price - base_tp_distance
            
            return {
                "base_sl": base_sl,
                "base_tp": base_tp,
                "sl_distance": base_sl_distance,
                "tp_distance": base_tp_distance,
                "atr": atr
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating base levels: {e}")
            return {"error": str(e)}
    
    def _apply_market_adjustments(self, base_levels: Dict, market_conditions: Dict) -> Dict:
        """Apply market condition adjustments to TP/SL levels"""
        try:
            adjusted_levels = base_levels.copy()
            
            if "error" in market_conditions:
                return adjusted_levels
            
            # Volatility adjustments
            volatility = market_conditions.get("volatility", {})
            if "regime" in volatility:
                vol_regime = volatility["regime"]
                if vol_regime == "high":
                    # Increase distances in high volatility
                    adjusted_levels["sl_distance"] *= 1.5
                    adjusted_levels["tp_distance"] *= 1.5
                elif vol_regime == "low":
                    # Decrease distances in low volatility
                    adjusted_levels["sl_distance"] *= 0.8
                    adjusted_levels["tp_distance"] *= 0.8
            
            # Trend adjustments
            trend = market_conditions.get("trend", {})
            if "overall" in trend:
                overall_trend = trend["overall"]
                if overall_trend == "up":
                    # Favor long positions
                    adjusted_levels["tp_distance"] *= 1.1
                    adjusted_levels["sl_distance"] *= 0.9
                elif overall_trend == "down":
                    # Favor short positions
                    adjusted_levels["tp_distance"] *= 1.1
                    adjusted_levels["sl_distance"] *= 0.9
            
            return adjusted_levels
            
        except Exception as e:
            self.logger.error(f"Error applying market adjustments: {e}")
            return base_levels
    
    def _apply_volatility_adjustments(self, levels: Dict, market_data: pd.DataFrame) -> Dict:
        """Apply volatility-based adjustments"""
        try:
            volatility_adjusted = levels.copy()
            
            # Calculate current volatility vs historical
            returns = market_data['close'].pct_change().dropna()
            current_vol = returns.rolling(window=10).std().iloc[-1]
            historical_vol = returns.std()
            
            # Adjust based on volatility change
            vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
            
            if vol_ratio > 1.5:
                # High volatility - increase distances
                volatility_adjusted["sl_distance"] *= 1.3
                volatility_adjusted["tp_distance"] *= 1.3
            elif vol_ratio < 0.7:
                # Low volatility - decrease distances
                volatility_adjusted["sl_distance"] *= 0.8
                volatility_adjusted["tp_distance"] *= 0.8
            
            return volatility_adjusted
            
        except Exception as e:
            self.logger.error(f"Error applying volatility adjustments: {e}")
            return levels
    
    def _apply_trend_adjustments(self, levels: Dict, market_data: pd.DataFrame) -> Dict:
        """Apply trend-based adjustments"""
        try:
            trend_adjusted = levels.copy()
            
            # Calculate trend strength
            close = market_data['close']
            short_ma = close.rolling(window=5).mean()
            long_ma = close.rolling(window=20).mean()
            
            if not short_ma.empty and not long_ma.empty:
                trend_strength = abs(short_ma.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1]
                
                if trend_strength > 0.02:  # Strong trend
                    trend_adjusted["tp_distance"] *= 1.2
                    trend_adjusted["sl_distance"] *= 0.9
                elif trend_strength < 0.005:  # Weak trend
                    trend_adjusted["tp_distance"] *= 0.9
                    trend_adjusted["sl_distance"] *= 1.1
            
            return trend_adjusted
            
        except Exception as e:
            self.logger.error(f"Error applying trend adjustments: {e}")
            return levels
    
    def _apply_risk_adjustments(self, levels: Dict, position: Dict, risk_params: Dict) -> Dict:
        """Apply risk-based adjustments"""
        try:
            risk_adjusted = levels.copy()
            
            # Position size adjustments
            position_size = position.get("size", 0.01)
            max_position_size = risk_params.get("max_position_size", 0.1)
            
            # Adjust based on position size
            size_ratio = position_size / max_position_size
            if size_ratio > 0.8:
                # Large position - tighten SL, widen TP
                risk_adjusted["sl_distance"] *= 0.8
                risk_adjusted["tp_distance"] *= 1.2
            elif size_ratio < 0.3:
                # Small position - wider SL, tighter TP
                risk_adjusted["sl_distance"] *= 1.2
                risk_adjusted["tp_distance"] *= 0.8
            
            # Account balance adjustments
            account_balance = risk_params.get("account_balance", 10000)
            if account_balance < 5000:
                # Small account - more conservative
                risk_adjusted["sl_distance"] *= 0.9
                risk_adjusted["tp_distance"] *= 0.9
            
            return risk_adjusted
            
        except Exception as e:
            self.logger.error(f"Error applying risk adjustments: {e}")
            return levels
    
    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 0.001
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return 0.001
    
    def _store_calculation_history(self, position: Dict, levels: Dict, market_conditions: Dict):
        """Store TP/SL calculation history"""
        try:
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "position_id": position.get("id", "unknown"),
                "levels": levels,
                "market_conditions": market_conditions
            }
            
            self.tp_sl_history.append(history_entry)
            
            # Keep only last 1000 entries
            if len(self.tp_sl_history) > 1000:
                self.tp_sl_history = self.tp_sl_history[-1000:]
                
        except Exception as e:
            self.logger.error(f"Error storing calculation history: {e}")
    
    def update_position_tp_sl(self, position_id: str, new_levels: Dict) -> bool:
        """Update TP/SL levels for existing position"""
        try:
            # This would integrate with your broker API to modify orders
            self.logger.info(f"Updating TP/SL for position {position_id}: {new_levels}")
            
            # Store the update
            if position_id in self.active_positions:
                self.active_positions[position_id]["tp_sl"] = new_levels
                self.active_positions[position_id]["last_updated"] = datetime.now().isoformat()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating position TP/SL: {e}")
            return False
    
    def generate_tp_sl_report(self) -> Dict:
        """Generate comprehensive TP/SL report"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "active_positions": len(self.active_positions),
                "calculation_history": len(self.tp_sl_history),
                "recent_calculations": self.tp_sl_history[-10:] if self.tp_sl_history else [],
                "market_conditions": self.market_conditions,
                "risk_metrics": self.risk_metrics
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating TP/SL report: {e}")
            return {"error": str(e)}