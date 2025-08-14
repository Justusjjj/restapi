#!/usr/bin/env python3
"""
Advanced Trade Evaluator and Mistake Analysis System
Analyzes trades, identifies mistakes, and provides improvement recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedTradeEvaluator:
    """
    Advanced Trade Evaluator with Mistake Analysis and Improvement Recommendations
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.trade_history = []
        self.mistake_patterns = {}
        self.performance_metrics = {}
        self.improvement_recommendations = []
        
        # Setup logging
        self._setup_logging()
        
    def _default_config(self):
        """Default configuration for trade evaluator"""
        return {
            'evaluation_window': 100,  # Number of trades to analyze
            'mistake_threshold': 0.1,   # Threshold for mistake identification
            'performance_metrics': True,
            'mistake_analysis': True,
            'pattern_recognition': True,
            'improvement_suggestions': True,
            'risk_assessment': True
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trade_evaluator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def add_trade(self, trade_data):
        """
        Add a trade to the evaluation system
        
        Args:
            trade_data: Dictionary containing trade information
        """
        try:
            # Validate trade data
            required_fields = ['symbol', 'side', 'entry_price', 'exit_price', 'entry_time', 'exit_time', 'amount']
            for field in required_fields:
                if field not in trade_data:
                    self.logger.error(f"Missing required field: {field}")
                    return False
            
            # Calculate additional metrics
            trade_data['duration'] = (trade_data['exit_time'] - trade_data['entry_time']).total_seconds() / 3600  # hours
            trade_data['price_change'] = trade_data['exit_price'] - trade_data['entry_price']
            trade_data['price_change_pct'] = (trade_data['price_change'] / trade_data['entry_price']) * 100
            
            # Calculate P&L
            if trade_data['side'].upper() == 'BUY':
                trade_data['pnl'] = trade_data['price_change'] * trade_data['amount']
                trade_data['pnl_pct'] = trade_data['price_change_pct']
            else:  # SELL
                trade_data['pnl'] = -trade_data['price_change'] * trade_data['amount']
                trade_data['pnl_pct'] = -trade_data['price_change_pct']
            
            # Add trade to history
            self.trade_history.append(trade_data)
            self.logger.info(f"Trade added: {trade_data['symbol']} {trade_data['side']} P&L: ${trade_data['pnl']:.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding trade: {e}")
            return False
    
    def evaluate_trade(self, trade_data):
        """
        Evaluate a single trade for mistakes and improvements
        
        Args:
            trade_data: Trade data dictionary
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            evaluation = {
                'trade_id': len(self.trade_history),
                'symbol': trade_data['symbol'],
                'side': trade_data['side'],
                'entry_time': trade_data['entry_time'],
                'exit_time': trade_data['exit_time'],
                'mistakes': [],
                'score': 0.0,
                'recommendations': [],
                'risk_assessment': {},
                'performance_metrics': {}
            }
            
            # 1. Entry Analysis
            entry_analysis = self._analyze_entry(trade_data)
            evaluation['entry_analysis'] = entry_analysis
            
            # 2. Exit Analysis
            exit_analysis = self._analyze_exit(trade_data)
            evaluation['exit_analysis'] = exit_analysis
            
            # 3. Risk Analysis
            risk_analysis = self._analyze_risk(trade_data)
            evaluation['risk_assessment'] = risk_analysis
            
            # 4. Mistake Identification
            mistakes = self._identify_mistakes(trade_data, entry_analysis, exit_analysis, risk_analysis)
            evaluation['mistakes'] = mistakes
            
            # 5. Performance Scoring
            score = self._calculate_trade_score(trade_data, mistakes)
            evaluation['score'] = score
            
            # 6. Improvement Recommendations
            recommendations = self._generate_recommendations(trade_data, mistakes, score)
            evaluation['recommendations'] = recommendations
            
            # 7. Performance Metrics
            performance = self._calculate_performance_metrics(trade_data)
            evaluation['performance_metrics'] = performance
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluating trade: {e}")
            return {}
    
    def _analyze_entry(self, trade_data):
        """Analyze trade entry for potential issues"""
        try:
            entry_analysis = {
                'timing_score': 0.0,
                'price_score': 0.0,
                'volume_score': 0.0,
                'market_condition_score': 0.0,
                'issues': []
            }
            
            # Entry timing analysis
            entry_hour = trade_data['entry_time'].hour
            if 8 <= entry_hour <= 16:  # London/NY overlap
                entry_analysis['timing_score'] = 1.0
            elif 13 <= entry_hour <= 21:  # NY session
                entry_analysis['timing_score'] = 0.8
            elif 0 <= entry_hour < 8:  # Asian session
                entry_analysis['timing_score'] = 0.6
            else:
                entry_analysis['timing_score'] = 0.4
                entry_analysis['issues'].append("Entry during low-liquidity hours")
            
            # Entry price analysis (if market data available)
            if 'market_data' in trade_data:
                market_data = trade_data['market_data']
                current_price = market_data['close'].iloc[-1]
                
                # Check if entry is near support/resistance
                if 'support_level' in market_data.columns and 'resistance_level' in market_data.columns:
                    support = market_data['support_level'].iloc[-1]
                    resistance = market_data['resistance_level'].iloc[-1]
                    
                    if abs(current_price - support) / current_price < 0.001:  # Within 0.1% of support
                        entry_analysis['price_score'] = 1.0
                        entry_analysis['issues'].append("Entry near strong support level")
                    elif abs(current_price - resistance) / current_price < 0.001:  # Within 0.1% of resistance
                        entry_analysis['price_score'] = 0.3
                        entry_analysis['issues'].append("Entry near strong resistance level")
                    else:
                        entry_analysis['price_score'] = 0.7
                else:
                    entry_analysis['price_score'] = 0.5
            
            # Volume analysis
            if 'volume' in trade_data:
                volume = trade_data['volume']
                if volume > 10000:  # High volume
                    entry_analysis['volume_score'] = 1.0
                elif volume > 5000:  # Medium volume
                    entry_analysis['volume_score'] = 0.7
                else:  # Low volume
                    entry_analysis['volume_score'] = 0.3
                    entry_analysis['issues'].append("Low volume entry")
            else:
                entry_analysis['volume_score'] = 0.5
            
            # Market condition analysis
            if 'market_regime' in trade_data:
                regime = trade_data['market_regime']
                if regime == 'trending':
                    entry_analysis['market_condition_score'] = 1.0
                elif regime == 'mean_reverting':
                    entry_analysis['market_condition_score'] = 0.8
                else:
                    entry_analysis['market_condition_score'] = 0.5
            else:
                entry_analysis['market_condition_score'] = 0.5
            
            return entry_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing entry: {e}")
            return {'timing_score': 0.0, 'price_score': 0.0, 'volume_score': 0.0, 'market_condition_score': 0.0, 'issues': []}
    
    def _analyze_exit(self, trade_data):
        """Analyze trade exit for potential issues"""
        try:
            exit_analysis = {
                'timing_score': 0.0,
                'price_score': 0.0,
                'reason_score': 0.0,
                'issues': []
            }
            
            # Exit timing analysis
            duration = trade_data['duration']
            if duration < 1:  # Less than 1 hour
                exit_analysis['timing_score'] = 0.3
                exit_analysis['issues'].append("Very short trade duration - possible overtrading")
            elif duration < 4:  # 1-4 hours
                exit_analysis['timing_score'] = 0.7
            elif duration < 24:  # 4-24 hours
                exit_analysis['timing_score'] = 1.0
            elif duration < 72:  # 1-3 days
                exit_analysis['timing_score'] = 0.8
            else:  # More than 3 days
                exit_analysis['timing_score'] = 0.5
                exit_analysis['issues'].append("Very long trade duration - possible overholding")
            
            # Exit price analysis
            if trade_data['pnl'] > 0:  # Profitable trade
                exit_analysis['price_score'] = 1.0
                
                # Check if profit was taken too early
                if trade_data['pnl_pct'] < 0.5:  # Less than 0.5% profit
                    exit_analysis['issues'].append("Profit taken too early - consider trailing stops")
            else:  # Losing trade
                exit_analysis['price_score'] = 0.3
                
                # Check stop loss usage
                if 'stop_loss' in trade_data and trade_data['stop_loss']:
                    exit_analysis['reason_score'] = 0.8
                    exit_analysis['issues'].append("Stop loss triggered - review risk management")
                else:
                    exit_analysis['reason_score'] = 0.2
                    exit_analysis['issues'].append("No stop loss used - implement risk management")
            
            # Exit reason analysis
            if 'exit_reason' in trade_data:
                reason = trade_data['exit_reason']
                if reason == 'take_profit':
                    exit_analysis['reason_score'] = 1.0
                elif reason == 'stop_loss':
                    exit_analysis['reason_score'] = 0.8
                elif reason == 'manual':
                    exit_analysis['reason_score'] = 0.6
                else:
                    exit_analysis['reason_score'] = 0.5
            
            return exit_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing exit: {e}")
            return {'timing_score': 0.0, 'price_score': 0.0, 'reason_score': 0.0, 'issues': []}
    
    def _analyze_risk(self, trade_data):
        """Analyze trade risk management"""
        try:
            risk_analysis = {
                'position_sizing_score': 0.0,
                'stop_loss_score': 0.0,
                'take_profit_score': 0.0,
                'correlation_score': 0.0,
                'overall_risk_score': 0.0,
                'issues': []
            }
            
            # Position sizing analysis
            if 'account_balance' in trade_data and 'risk_per_trade' in trade_data:
                balance = trade_data['account_balance']
                risk_amount = abs(trade_data['pnl'])
                risk_percentage = (risk_amount / balance) * 100
                
                if risk_percentage <= 1.0:  # 1% or less
                    risk_analysis['position_sizing_score'] = 1.0
                elif risk_percentage <= 2.0:  # 1-2%
                    risk_analysis['position_sizing_score'] = 0.8
                elif risk_percentage <= 5.0:  # 2-5%
                    risk_analysis['position_sizing_score'] = 0.5
                    risk_analysis['issues'].append("Position size too large - risk per trade > 2%")
                else:  # More than 5%
                    risk_analysis['position_sizing_score'] = 0.0
                    risk_analysis['issues'].append("Position size excessive - risk per trade > 5%")
            else:
                risk_analysis['position_sizing_score'] = 0.5
            
            # Stop loss analysis
            if 'stop_loss' in trade_data and trade_data['stop_loss']:
                stop_loss_pips = abs(trade_data['entry_price'] - trade_data['stop_loss']) * 10000
                
                if stop_loss_pips <= 20:  # 20 pips or less
                    risk_analysis['stop_loss_score'] = 0.3
                    risk_analysis['issues'].append("Stop loss too tight - consider wider stops")
                elif stop_loss_pips <= 50:  # 20-50 pips
                    risk_analysis['stop_loss_score'] = 0.8
                elif stop_loss_pips <= 100:  # 50-100 pips
                    risk_analysis['stop_loss_score'] = 1.0
                else:  # More than 100 pips
                    risk_analysis['stop_loss_score'] = 0.6
                    risk_analysis['issues'].append("Stop loss too wide - consider tighter stops")
            else:
                risk_analysis['stop_loss_score'] = 0.0
                risk_analysis['issues'].append("No stop loss used - implement risk management")
            
            # Take profit analysis
            if 'take_profit' in trade_data and trade_data['take_profit']:
                take_profit_pips = abs(trade_data['entry_price'] - trade_data['take_profit']) * 10000
                risk_reward_ratio = take_profit_pips / stop_loss_pips if 'stop_loss' in trade_data and trade_data['stop_loss'] else 0
                
                if risk_reward_ratio >= 2.0:  # 2:1 or better
                    risk_analysis['take_profit_score'] = 1.0
                elif risk_reward_ratio >= 1.5:  # 1.5:1 or better
                    risk_analysis['take_profit_score'] = 0.8
                elif risk_reward_ratio >= 1.0:  # 1:1 or better
                    risk_analysis['take_profit_score'] = 0.6
                else:  # Less than 1:1
                    risk_analysis['take_profit_score'] = 0.3
                    risk_analysis['issues'].append("Poor risk-reward ratio - aim for 2:1 or better")
            else:
                risk_analysis['take_profit_score'] = 0.5
                risk_analysis['issues'].append("No take profit set - consider profit targets")
            
            # Calculate overall risk score
            risk_scores = [
                risk_analysis['position_sizing_score'],
                risk_analysis['stop_loss_score'],
                risk_analysis['take_profit_score']
            ]
            risk_analysis['overall_risk_score'] = np.mean(risk_scores)
            
            return risk_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing risk: {e}")
            return {'position_sizing_score': 0.0, 'stop_loss_score': 0.0, 'take_profit_score': 0.0, 'overall_risk_score': 0.0, 'issues': []}
    
    def _identify_mistakes(self, trade_data, entry_analysis, exit_analysis, risk_analysis):
        """Identify specific mistakes in the trade"""
        try:
            mistakes = []
            
            # Entry mistakes
            if entry_analysis['timing_score'] < 0.6:
                mistakes.append({
                    'type': 'entry_timing',
                    'severity': 'medium',
                    'description': 'Poor entry timing - consider market session analysis',
                    'impact': 'May result in poor fill prices and increased slippage'
                })
            
            if entry_analysis['price_score'] < 0.5:
                mistakes.append({
                    'type': 'entry_price',
                    'severity': 'high',
                    'description': 'Entry near strong resistance/support without confirmation',
                    'impact': 'High probability of immediate reversal'
                })
            
            if entry_analysis['volume_score'] < 0.5:
                mistakes.append({
                    'type': 'entry_volume',
                    'severity': 'medium',
                    'description': 'Low volume entry - lack of market participation',
                    'impact': 'May result in poor liquidity and wide spreads'
                })
            
            # Exit mistakes
            if exit_analysis['timing_score'] < 0.5:
                mistakes.append({
                    'type': 'exit_timing',
                    'severity': 'medium',
                    'description': 'Poor exit timing - consider trend analysis',
                    'impact': 'May result in premature exits or overholding'
                })
            
            if exit_analysis['price_score'] < 0.5:
                mistakes.append({
                    'type': 'exit_price',
                    'severity': 'high',
                    'description': 'Poor exit execution - consider trailing stops',
                    'impact': 'May result in reduced profits or increased losses'
                })
            
            # Risk management mistakes
            if risk_analysis['overall_risk_score'] < 0.5:
                mistakes.append({
                    'type': 'risk_management',
                    'severity': 'high',
                    'description': 'Poor risk management - review position sizing and stops',
                    'impact': 'May result in excessive losses and account drawdown'
                })
            
            if risk_analysis['stop_loss_score'] == 0.0:
                mistakes.append({
                    'type': 'no_stop_loss',
                    'severity': 'critical',
                    'description': 'No stop loss used - implement risk management immediately',
                    'impact': 'Unlimited downside risk - potential for catastrophic losses'
                })
            
            # Market condition mistakes
            if 'market_regime' in trade_data:
                regime = trade_data['market_regime']
                if regime == 'high_volatility' and trade_data['duration'] > 24:
                    mistakes.append({
                        'type': 'market_condition',
                        'severity': 'medium',
                        'description': 'Holding positions too long in high volatility',
                        'impact': 'Increased risk of adverse price movements'
                    })
            
            return mistakes
            
        except Exception as e:
            self.logger.error(f"Error identifying mistakes: {e}")
            return []
    
    def _calculate_trade_score(self, trade_data, mistakes):
        """Calculate overall trade score"""
        try:
            base_score = 100.0
            
            # Deduct points for mistakes
            mistake_deductions = {
                'critical': 30,
                'high': 20,
                'medium': 10,
                'low': 5
            }
            
            for mistake in mistakes:
                severity = mistake['severity']
                if severity in mistake_deductions:
                    base_score -= mistake_deductions[severity]
            
            # Bonus points for good practices
            if trade_data['pnl'] > 0:  # Profitable trade
                base_score += 10
            
            if 'stop_loss' in trade_data and trade_data['stop_loss']:
                base_score += 5
            
            if 'take_profit' in trade_data and trade_data['take_profit']:
                base_score += 5
            
            # Ensure score is within bounds
            return max(0.0, min(100.0, base_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating trade score: {e}")
            return 50.0
    
    def _generate_recommendations(self, trade_data, mistakes, score):
        """Generate improvement recommendations"""
        try:
            recommendations = []
            
            # General recommendations based on score
            if score < 50:
                recommendations.append({
                    'priority': 'high',
                    'category': 'risk_management',
                    'description': 'Implement comprehensive risk management system',
                    'action': 'Set maximum risk per trade to 1-2% of account balance'
                })
            
            if score < 70:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'entry_timing',
                    'description': 'Improve entry timing using market session analysis',
                    'action': 'Focus on London/NY overlap for major currency pairs'
                })
            
            # Specific recommendations based on mistakes
            for mistake in mistakes:
                if mistake['type'] == 'no_stop_loss':
                    recommendations.append({
                        'priority': 'critical',
                        'category': 'risk_management',
                        'description': 'Always use stop losses to limit downside risk',
                        'action': 'Set stop loss at 2-3 times the average true range (ATR)'
                    })
                
                elif mistake['type'] == 'entry_timing':
                    recommendations.append({
                        'priority': 'medium',
                        'category': 'entry_timing',
                        'description': 'Improve entry timing using market session analysis',
                        'action': 'Use economic calendar to avoid high-impact news events'
                    })
                
                elif mistake['type'] == 'position_sizing':
                    recommendations.append({
                        'priority': 'high',
                        'category': 'risk_management',
                        'description': 'Reduce position sizes to manage risk',
                        'action': 'Calculate position size based on stop loss distance and account risk'
                    })
                
                elif mistake['type'] == 'exit_timing':
                    recommendations.append({
                        'priority': 'medium',
                        'category': 'exit_timing',
                        'description': 'Improve exit timing using technical analysis',
                        'action': 'Use trailing stops and multiple take profit levels'
                    })
            
            # Performance-based recommendations
            if trade_data['pnl'] < 0:  # Losing trade
                recommendations.append({
                    'priority': 'medium',
                    'category': 'analysis',
                    'description': 'Review trade analysis and market conditions',
                    'action': 'Keep detailed trade journal and review weekly'
                })
            
            # Duration-based recommendations
            if trade_data['duration'] < 1:  # Very short trade
                recommendations.append({
                    'priority': 'medium',
                    'category': 'trading_style',
                    'description': 'Consider longer-term trades to reduce noise',
                    'action': 'Use higher timeframes (4H, 1D) for trend analysis'
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _calculate_performance_metrics(self, trade_data):
        """Calculate detailed performance metrics"""
        try:
            metrics = {
                'pnl': trade_data['pnl'],
                'pnl_pct': trade_data['pnl_pct'],
                'duration_hours': trade_data['duration'],
                'risk_reward_ratio': 0.0,
                'slippage': 0.0,
                'fill_quality': 0.0
            }
            
            # Calculate risk-reward ratio
            if 'stop_loss' in trade_data and trade_data['stop_loss'] and 'take_profit' in trade_data and trade_data['take_profit']:
                stop_distance = abs(trade_data['entry_price'] - trade_data['stop_loss'])
                profit_distance = abs(trade_data['entry_price'] - trade_data['take_profit'])
                if stop_distance > 0:
                    metrics['risk_reward_ratio'] = profit_distance / stop_distance
            
            # Calculate slippage (if market data available)
            if 'intended_entry' in trade_data and 'intended_exit' in trade_data:
                entry_slippage = abs(trade_data['entry_price'] - trade_data['intended_entry'])
                exit_slippage = abs(trade_data['exit_price'] - trade_data['intended_exit'])
                metrics['slippage'] = entry_slippage + exit_slippage
            
            # Calculate fill quality
            if 'spread' in trade_data:
                spread = trade_data['spread']
                if spread <= 0.0001:  # 1 pip or less
                    metrics['fill_quality'] = 1.0
                elif spread <= 0.0003:  # 1-3 pips
                    metrics['fill_quality'] = 0.8
                elif spread <= 0.0005:  # 3-5 pips
                    metrics['fill_quality'] = 0.6
                else:  # More than 5 pips
                    metrics['fill_quality'] = 0.3
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def analyze_trading_patterns(self, window=50):
        """
        Analyze trading patterns and identify recurring mistakes
        
        Args:
            window: Number of recent trades to analyze
            
        Returns:
            Dictionary with pattern analysis results
        """
        try:
            if len(self.trade_history) < window:
                window = len(self.trade_history)
            
            recent_trades = self.trade_history[-window:]
            
            pattern_analysis = {
                'total_trades': len(recent_trades),
                'winning_trades': len([t for t in recent_trades if t['pnl'] > 0]),
                'losing_trades': len([t for t in recent_trades if t['pnl'] < 0]),
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_duration': 0.0,
                'common_mistakes': {},
                'performance_trends': {},
                'risk_metrics': {}
            }
            
            if len(recent_trades) > 0:
                # Calculate basic metrics
                winning_trades = [t for t in recent_trades if t['pnl'] > 0]
                losing_trades = [t for t in recent_trades if t['pnl'] < 0]
                
                if winning_trades:
                    pattern_analysis['avg_win'] = np.mean([t['pnl'] for t in winning_trades])
                
                if losing_trades:
                    pattern_analysis['avg_loss'] = np.mean([abs(t['pnl']) for t in losing_trades])
                
                pattern_analysis['win_rate'] = len(winning_trades) / len(recent_trades)
                pattern_analysis['avg_duration'] = np.mean([t['duration'] for t in recent_trades])
                
                # Calculate profit factor
                total_wins = sum([t['pnl'] for t in winning_trades])
                total_losses = sum([abs(t['pnl']) for t in losing_trades])
                if total_losses > 0:
                    pattern_analysis['profit_factor'] = total_wins / total_losses
                
                # Analyze common mistakes
                all_mistakes = []
                for trade in recent_trades:
                    evaluation = self.evaluate_trade(trade)
                    all_mistakes.extend(evaluation.get('mistakes', []))
                
                mistake_counts = {}
                for mistake in all_mistakes:
                    mistake_type = mistake['type']
                    if mistake_type in mistake_counts:
                        mistake_counts[mistake_type] += 1
                    else:
                        mistake_counts[mistake_type] = 1
                
                pattern_analysis['common_mistakes'] = mistake_counts
                
                # Performance trends
                if len(recent_trades) >= 10:
                    first_half = recent_trades[:len(recent_trades)//2]
                    second_half = recent_trades[len(recent_trades)//2:]
                    
                    first_half_pnl = sum([t['pnl'] for t in first_half])
                    second_half_pnl = sum([t['pnl'] for t in second_half])
                    
                    if second_half_pnl > first_half_pnl:
                        pattern_analysis['performance_trends']['trend'] = 'improving'
                    else:
                        pattern_analysis['performance_trends']['trend'] = 'declining'
                
                # Risk metrics
                if 'account_balance' in recent_trades[0]:
                    total_risk = sum([abs(t['pnl']) for t in recent_trades])
                    account_balance = recent_trades[0]['account_balance']
                    pattern_analysis['risk_metrics']['total_risk_pct'] = (total_risk / account_balance) * 100
            
            return pattern_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing trading patterns: {e}")
            return {}
    
    def generate_improvement_plan(self, analysis_results):
        """
        Generate comprehensive improvement plan based on analysis
        
        Args:
            analysis_results: Results from pattern analysis
            
        Returns:
            Dictionary with improvement plan
        """
        try:
            improvement_plan = {
                'priority_actions': [],
                'medium_term_goals': [],
                'long_term_goals': [],
                'training_recommendations': [],
                'risk_management_updates': []
            }
            
            # Priority actions based on common mistakes
            common_mistakes = analysis_results.get('common_mistakes', {})
            
            if 'no_stop_loss' in common_mistakes and common_mistakes['no_stop_loss'] > 0:
                improvement_plan['priority_actions'].append({
                    'action': 'Implement mandatory stop losses',
                    'deadline': 'Immediate',
                    'description': 'Set stop loss for all open positions and future trades',
                    'impact': 'Critical for risk management'
                })
            
            if 'position_sizing' in common_mistakes and common_mistakes['position_sizing'] > 0:
                improvement_plan['priority_actions'].append({
                    'action': 'Reduce position sizes',
                    'deadline': 'Next trading session',
                    'description': 'Calculate position size based on 1-2% risk per trade',
                    'impact': 'Reduce account drawdown risk'
                })
            
            if 'entry_timing' in common_mistakes and common_mistakes['entry_timing'] > 0:
                improvement_plan['priority_actions'].append({
                    'action': 'Improve entry timing',
                    'deadline': 'This week',
                    'description': 'Focus on market session analysis and avoid news events',
                    'impact': 'Improve entry prices and reduce slippage'
                })
            
            # Medium-term goals
            win_rate = analysis_results.get('win_rate', 0)
            if win_rate < 0.4:
                improvement_plan['medium_term_goals'].append({
                    'goal': 'Improve win rate to 50%',
                    'timeline': '3 months',
                    'actions': ['Review entry criteria', 'Improve market analysis', 'Reduce overtrading']
                })
            
            profit_factor = analysis_results.get('profit_factor', 0)
            if profit_factor < 1.5:
                improvement_plan['medium_term_goals'].append({
                    'goal': 'Achieve profit factor of 2.0',
                    'timeline': '6 months',
                    'actions': ['Improve risk-reward ratios', 'Better exit timing', 'Reduce average loss']
                })
            
            # Long-term goals
            improvement_plan['long_term_goals'].append({
                'goal': 'Consistent monthly profitability',
                'timeline': '12 months',
                'actions': ['Develop trading system', 'Implement proper risk management', 'Continuous learning']
            })
            
            # Training recommendations
            if 'risk_management' in common_mistakes:
                improvement_plan['training_recommendations'].append({
                    'topic': 'Risk Management Fundamentals',
                    'resources': ['Risk management books', 'Online courses', 'Trading psychology']
                })
            
            if 'entry_timing' in common_mistakes:
                improvement_plan['training_recommendations'].append({
                    'topic': 'Market Analysis and Timing',
                    'resources': ['Technical analysis courses', 'Market session analysis', 'News trading']
                })
            
            # Risk management updates
            improvement_plan['risk_management_updates'].append({
                'rule': 'Maximum 1% risk per trade',
                'implementation': 'Immediate',
                'monitoring': 'Daily review'
            })
            
            improvement_plan['risk_management_updates'].append({
                'rule': 'Maximum 5% daily loss limit',
                'implementation': 'Immediate',
                'monitoring': 'Daily review'
            })
            
            return improvement_plan
            
        except Exception as e:
            self.logger.error(f"Error generating improvement plan: {e}")
            return {}
    
    def get_comprehensive_report(self, trade_id=None):
        """
        Generate comprehensive trading report
        
        Args:
            trade_id: Specific trade ID, or None for overall report
            
        Returns:
            Dictionary with comprehensive report
        """
        try:
            if trade_id is not None and trade_id < len(self.trade_history):
                # Single trade report
                trade_data = self.trade_history[trade_id]
                evaluation = self.evaluate_trade(trade_data)
                
                report = {
                    'report_type': 'single_trade',
                    'trade_data': trade_data,
                    'evaluation': evaluation,
                    'timestamp': datetime.now()
                }
            else:
                # Overall performance report
                pattern_analysis = self.analyze_trading_patterns()
                improvement_plan = self.generate_improvement_plan(pattern_analysis)
                
                report = {
                    'report_type': 'overall_performance',
                    'pattern_analysis': pattern_analysis,
                    'improvement_plan': improvement_plan,
                    'total_trades': len(self.trade_history),
                    'timestamp': datetime.now()
                }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            return {}

if __name__ == "__main__":
    # Example usage
    evaluator = AdvancedTradeEvaluator()
    
    # Sample trade data
    sample_trade = {
        'symbol': 'EUR/USD',
        'side': 'BUY',
        'entry_price': 1.1000,
        'exit_price': 1.1050,
        'entry_time': datetime.now() - timedelta(hours=4),
        'exit_time': datetime.now(),
        'amount': 10000,
        'stop_loss': 1.0950,
        'take_profit': 1.1100,
        'exit_reason': 'take_profit',
        'account_balance': 100000,
        'volume': 8000,
        'spread': 0.0002
    }
    
    # Add and evaluate trade
    evaluator.add_trade(sample_trade)
    evaluation = evaluator.evaluate_trade(sample_trade)
    
    print("Trade Evaluation Results:")
    print(json.dumps(evaluation, indent=2, default=str))
    
    # Analyze patterns
    patterns = evaluator.analyze_trading_patterns()
    print("\nTrading Pattern Analysis:")
    print(json.dumps(patterns, indent=2, default=str))
    
    # Generate improvement plan
    improvement_plan = evaluator.generate_improvement_plan(patterns)
    print("\nImprovement Plan:")
    print(json.dumps(improvement_plan, indent=2, default=str))