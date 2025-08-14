import numpy as np
import pandas as pd
import mplfinance as mpf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ICTPriceActionAnalyzer:
    """
    Advanced ICT (Inner Circle Trader) and Price Action Analysis
    Includes candlestick patterns, order flow, and mathematical formulas
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.pattern_history = []
        self.order_flow_data = []
        self.ict_levels = {}
        
    def _default_config(self):
        """Default configuration for ICT analysis"""
        return {
            'fair_value_gaps': True,
            'liquidity_levels': True,
            'order_blocks': True,
            'breakers': True,
            'mitigation_blocks': True,
            'candlestick_patterns': True,
            'volume_profile': True,
            'market_structure': True,
            'time_analysis': True
        }
    
    def analyze_ict_levels(self, df):
        """
        Analyze ICT (Inner Circle Trader) levels
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with ICT levels identified
        """
        try:
            df = df.copy()
            
            # Fair Value Gaps (FVG)
            df['fvg_bullish'] = self._identify_fair_value_gaps(df, 'bullish')
            df['fvg_bearish'] = self._identify_fair_value_gaps(df, 'bearish')
            
            # Liquidity Levels
            df['liquidity_high'] = self._identify_liquidity_levels(df, 'high')
            df['liquidity_low'] = self._identify_liquidity_levels(df, 'low')
            
            # Order Blocks
            df['order_block_bullish'] = self._identify_order_blocks(df, 'bullish')
            df['order_block_bearish'] = self._identify_order_blocks(df, 'bearish')
            
            # Breakers
            df['breaker_bullish'] = self._identify_breakers(df, 'bullish')
            df['breaker_bearish'] = self._identify_breakers(df, 'bearish')
            
            # Mitigation Blocks
            df['mitigation_block'] = self._identify_mitigation_blocks(df)
            
            # Market Structure
            df['market_structure'] = self._analyze_market_structure(df)
            
            # Time Analysis
            df['time_analysis'] = self._analyze_time_analysis(df)
            
            return df
            
        except Exception as e:
            print(f"Error analyzing ICT levels: {e}")
            return df
    
    def _identify_fair_value_gaps(self, df, gap_type):
        """
        Identify Fair Value Gaps (FVG)
        
        Args:
            df: DataFrame with OHLCV data
            gap_type: 'bullish' or 'bearish'
            
        Returns:
            Series with FVG levels
        """
        try:
            fvg_levels = pd.Series(0.0, index=df.index)
            
            for i in range(2, len(df)):
                if gap_type == 'bullish':
                    # Bullish FVG: Low of current candle > High of 2 candles ago
                    if (df['low'].iloc[i] > df['high'].iloc[i-2]):
                        # Calculate FVG level
                        fvg_top = df['low'].iloc[i]
                        fvg_bottom = df['high'].iloc[i-2]
                        fvg_mid = (fvg_top + fvg_bottom) / 2
                        fvg_levels.iloc[i] = fvg_mid
                        
                elif gap_type == 'bearish':
                    # Bearish FVG: High of current candle < Low of 2 candles ago
                    if (df['high'].iloc[i] < df['low'].iloc[i-2]):
                        # Calculate FVG level
                        fvg_top = df['low'].iloc[i-2]
                        fvg_bottom = df['high'].iloc[i]
                        fvg_mid = (fvg_top + fvg_bottom) / 2
                        fvg_levels.iloc[i] = fvg_mid
            
            return fvg_levels
            
        except Exception as e:
            print(f"Error identifying FVG: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _identify_liquidity_levels(self, df, level_type):
        """
        Identify Liquidity Levels
        
        Args:
            df: DataFrame with OHLCV data
            level_type: 'high' or 'low'
            
        Returns:
            Series with liquidity levels
        """
        try:
            liquidity_levels = pd.Series(0.0, index=df.index)
            
            window = 20
            
            for i in range(window, len(df)):
                if level_type == 'high':
                    # Look for swing highs
                    if (df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max() and
                        df['high'].iloc[i] > df['high'].iloc[i-1] and
                        df['high'].iloc[i] > df['high'].iloc[i+1]):
                        liquidity_levels.iloc[i] = df['high'].iloc[i]
                        
                elif level_type == 'low':
                    # Look for swing lows
                    if (df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min() and
                        df['low'].iloc[i] < df['low'].iloc[i-1] and
                        df['low'].iloc[i] < df['low'].iloc[i+1]):
                        liquidity_levels.iloc[i] = df['low'].iloc[i]
            
            return liquidity_levels
            
        except Exception as e:
            print(f"Error identifying liquidity levels: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _identify_order_blocks(self, df, block_type):
        """
        Identify Order Blocks
        
        Args:
            df: DataFrame with OHLCV data
            block_type: 'bullish' or 'bearish'
            
        Returns:
            Series with order block levels
        """
        try:
            order_blocks = pd.Series(0.0, index=df.index)
            
            for i in range(1, len(df)):
                if block_type == 'bullish':
                    # Bullish order block: Strong bullish candle followed by bearish move
                    if (df['close'].iloc[i] > df['open'].iloc[i] and  # Bullish candle
                        df['close'].iloc[i] - df['open'].iloc[i] > 0.5 * (df['high'].iloc[i] - df['low'].iloc[i]) and  # Strong body
                        i < len(df) - 1 and df['close'].iloc[i+1] < df['open'].iloc[i+1]):  # Followed by bearish
                        order_blocks.iloc[i] = df['low'].iloc[i]
                        
                elif block_type == 'bearish':
                    # Bearish order block: Strong bearish candle followed by bullish move
                    if (df['open'].iloc[i] > df['close'].iloc[i] and  # Bearish candle
                        df['open'].iloc[i] - df['close'].iloc[i] > 0.5 * (df['high'].iloc[i] - df['low'].iloc[i]) and  # Strong body
                        i < len(df) - 1 and df['close'].iloc[i+1] > df['open'].iloc[i+1]):  # Followed by bullish
                        order_blocks.iloc[i] = df['high'].iloc[i]
            
            return order_blocks
            
        except Exception as e:
            print(f"Error identifying order blocks: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _identify_breakers(self, df, breaker_type):
        """
        Identify Breakers (broken order blocks)
        
        Args:
            df: DataFrame with OHLCV data
            breaker_type: 'bullish' or 'bearish'
            
        Returns:
            Series with breaker levels
        """
        try:
            breakers = pd.Series(0.0, index=df.index)
            
            # Find order blocks first
            bullish_blocks = self._identify_order_blocks(df, 'bullish')
            bearish_blocks = self._identify_order_blocks(df, 'bearish')
            
            for i in range(1, len(df)):
                if breaker_type == 'bullish':
                    # Bullish breaker: Price breaks above bearish order block
                    if bearish_blocks.iloc[i-1] > 0:
                        if df['high'].iloc[i] > bearish_blocks.iloc[i-1]:
                            breakers.iloc[i] = bearish_blocks.iloc[i-1]
                            
                elif breaker_type == 'bearish':
                    # Bearish breaker: Price breaks below bullish order block
                    if bullish_blocks.iloc[i-1] > 0:
                        if df['low'].iloc[i] < bullish_blocks.iloc[i-1]:
                            breakers.iloc[i] = bullish_blocks.iloc[i-1]
            
            return breakers
            
        except Exception as e:
            print(f"Error identifying breakers: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _identify_mitigation_blocks(self, df):
        """
        Identify Mitigation Blocks (partial fills of order blocks)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with mitigation block levels
        """
        try:
            mitigation_blocks = pd.Series(0.0, index=df.index)
            
            # Find order blocks
            bullish_blocks = self._identify_order_blocks(df, 'bullish')
            bearish_blocks = self._identify_order_blocks(df, 'bearish')
            
            for i in range(1, len(df)):
                # Check for partial fills
                if bullish_blocks.iloc[i-1] > 0:
                    # Partial fill of bullish order block
                    if (df['low'].iloc[i] <= bullish_blocks.iloc[i-1] and
                        df['close'].iloc[i] > bullish_blocks.iloc[i-1]):
                        mitigation_blocks.iloc[i] = bullish_blocks.iloc[i-1]
                        
                if bearish_blocks.iloc[i-1] > 0:
                    # Partial fill of bearish order block
                    if (df['high'].iloc[i] >= bearish_blocks.iloc[i-1] and
                        df['close'].iloc[i] < bearish_blocks.iloc[i-1]):
                        mitigation_blocks.iloc[i] = bearish_blocks.iloc[i-1]
            
            return mitigation_blocks
            
        except Exception as e:
            print(f"Error identifying mitigation blocks: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _analyze_market_structure(self, df):
        """
        Analyze market structure (higher highs, lower lows, etc.)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with market structure analysis
        """
        try:
            market_structure = pd.Series('', index=df.index)
            
            window = 10
            
            for i in range(window, len(df)):
                # Higher High (HH)
                if (df['high'].iloc[i] > df['high'].iloc[i-window:i].max()):
                    market_structure.iloc[i] = 'HH'
                    
                # Lower Low (LL)
                elif (df['low'].iloc[i] < df['low'].iloc[i-window:i].min()):
                    market_structure.iloc[i] = 'LL'
                    
                # Higher Low (HL)
                elif (df['low'].iloc[i] > df['low'].iloc[i-window:i].min()):
                    market_structure.iloc[i] = 'HL'
                    
                # Lower High (LH)
                elif (df['high'].iloc[i] < df['high'].iloc[i-window:i].max()):
                    market_structure.iloc[i] = 'LH'
                    
                # Sideways
                else:
                    market_structure.iloc[i] = 'SIDE'
            
            return market_structure
            
        except Exception as e:
            print(f"Error analyzing market structure: {e}")
            return pd.Series('', index=df.index)
    
    def _analyze_time_analysis(self, df):
        """
        Analyze time-based patterns (London/NY overlap, etc.)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with time analysis
        """
        try:
            time_analysis = pd.Series('', index=df.index)
            
            for i in range(len(df)):
                hour = df.index[i].hour
                
                # London session (8-16 UTC)
                if 8 <= hour < 16:
                    time_analysis.iloc[i] = 'LONDON'
                    
                # New York session (13-21 UTC)
                elif 13 <= hour < 21:
                    time_analysis.iloc[i] = 'NY'
                    
                # London-NY overlap (13-16 UTC)
                elif 13 <= hour < 16:
                    time_analysis.iloc[i] = 'LONDON-NY'
                    
                # Asian session (0-8 UTC)
                elif hour < 8:
                    time_analysis.iloc[i] = 'ASIA'
                    
                # Late NY (21-24 UTC)
                elif hour >= 21:
                    time_analysis.iloc[i] = 'LATE_NY'
                    
                # Early morning (0-4 UTC)
                elif hour < 4:
                    time_analysis.iloc[i] = 'EARLY_MORNING'
            
            return time_analysis
            
        except Exception as e:
            print(f"Error analyzing time patterns: {e}")
            return pd.Series('', index=df.index)
    
    def analyze_candlestick_patterns(self, df):
        """
        Analyze advanced candlestick patterns
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with candlestick patterns identified
        """
        try:
            df = df.copy()
            
            # Basic patterns
            df['doji'] = self._identify_doji(df)
            df['hammer'] = self._identify_hammer(df)
            df['shooting_star'] = self._identify_shooting_star(df)
            df['engulfing_bullish'] = self._identify_engulfing(df, 'bullish')
            df['engulfing_bearish'] = self._identify_engulfing(df, 'bearish')
            df['morning_star'] = self._identify_morning_star(df)
            df['evening_star'] = self._identify_evening_star(df)
            df['three_white_soldiers'] = self._identify_three_white_soldiers(df)
            df['three_black_crows'] = self._identify_three_black_crows(df)
            
            # Advanced patterns
            df['inside_bar'] = self._identify_inside_bar(df)
            df['outside_bar'] = self._identify_outside_bar(df)
            df['pin_bar'] = self._identify_pin_bar(df)
            df['fakey'] = self._identify_fakey(df)
            df['breakout_bar'] = self._identify_breakout_bar(df)
            
            # Pattern strength
            df['pattern_strength'] = self._calculate_pattern_strength(df)
            
            return df
            
        except Exception as e:
            print(f"Error analyzing candlestick patterns: {e}")
            return df
    
    def _identify_doji(self, df):
        """Identify Doji patterns"""
        try:
            doji = pd.Series(False, index=df.index)
            
            for i in range(len(df)):
                body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
                total_range = df['high'].iloc[i] - df['low'].iloc[i]
                
                # Doji: body is less than 10% of total range
                if body_size <= 0.1 * total_range:
                    doji.iloc[i] = True
            
            return doji
            
        except Exception as e:
            print(f"Error identifying doji: {e}")
            return pd.Series(False, index=df.index)
    
    def _identify_hammer(self, df):
        """Identify Hammer patterns"""
        try:
            hammer = pd.Series(False, index=df.index)
            
            for i in range(len(df)):
                body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
                total_range = df['high'].iloc[i] - df['low'].iloc[i]
                lower_shadow = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
                upper_shadow = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])
                
                # Hammer: small body, long lower shadow, small upper shadow
                if (body_size <= 0.3 * total_range and
                    lower_shadow >= 2 * body_size and
                    upper_shadow <= 0.1 * total_range):
                    hammer.iloc[i] = True
            
            return hammer
            
        except Exception as e:
            print(f"Error identifying hammer: {e}")
            return pd.Series(False, index=df.index)
    
    def _identify_shooting_star(self, df):
        """Identify Shooting Star patterns"""
        try:
            shooting_star = pd.Series(False, index=df.index)
            
            for i in range(len(df)):
                body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
                total_range = df['high'].iloc[i] - df['low'].iloc[i]
                lower_shadow = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
                upper_shadow = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])
                
                # Shooting Star: small body, long upper shadow, small lower shadow
                if (body_size <= 0.3 * total_range and
                    upper_shadow >= 2 * body_size and
                    lower_shadow <= 0.1 * total_range):
                    shooting_star.iloc[i] = True
            
            return shooting_star
            
        except Exception as e:
            print(f"Error identifying shooting star: {e}")
            return pd.Series(False, index=df.index)
    
    def _identify_engulfing(self, df, pattern_type):
        """Identify Engulfing patterns"""
        try:
            engulfing = pd.Series(False, index=df.index)
            
            for i in range(1, len(df)):
                prev_body_size = abs(df['close'].iloc[i-1] - df['open'].iloc[i-1])
                curr_body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
                
                if pattern_type == 'bullish':
                    # Bullish engulfing: current bullish candle engulfs previous bearish candle
                    if (df['close'].iloc[i] > df['open'].iloc[i] and  # Current bullish
                        df['close'].iloc[i-1] < df['open'].iloc[i-1] and  # Previous bearish
                        df['open'].iloc[i] < df['close'].iloc[i-1] and  # Current open below previous close
                        df['close'].iloc[i] > df['open'].iloc[i-1] and  # Current close above previous open
                        curr_body_size > prev_body_size):  # Current body larger
                        engulfing.iloc[i] = True
                        
                elif pattern_type == 'bearish':
                    # Bearish engulfing: current bearish candle engulfs previous bullish candle
                    if (df['close'].iloc[i] < df['open'].iloc[i] and  # Current bearish
                        df['close'].iloc[i-1] > df['open'].iloc[i-1] and  # Previous bullish
                        df['open'].iloc[i] > df['close'].iloc[i-1] and  # Current open above previous close
                        df['close'].iloc[i] < df['open'].iloc[i-1] and  # Current close below previous open
                        curr_body_size > prev_body_size):  # Current body larger
                        engulfing.iloc[i] = True
            
            return engulfing
            
        except Exception as e:
            print(f"Error identifying engulfing: {e}")
            return pd.Series(False, index=df.index)
    
    def _identify_morning_star(self, df):
        """Identify Morning Star pattern"""
        try:
            morning_star = pd.Series(False, index=df.index)
            
            for i in range(2, len(df)):
                # Morning Star: bearish, small body (doji-like), bullish
                if (df['close'].iloc[i-2] < df['open'].iloc[i-2] and  # First bearish
                    abs(df['close'].iloc[i-1] - df['open'].iloc[i-1]) <= 0.1 * (df['high'].iloc[i-1] - df['low'].iloc[i-1]) and  # Second small body
                    df['close'].iloc[i] > df['open'].iloc[i]):  # Third bullish
                    morning_star.iloc[i] = True
            
            return morning_star
            
        except Exception as e:
            print(f"Error identifying morning star: {e}")
            return pd.Series(False, index=df.index)
    
    def _identify_evening_star(self, df):
        """Identify Evening Star pattern"""
        try:
            evening_star = pd.Series(False, index=df.index)
            
            for i in range(2, len(df)):
                # Evening Star: bullish, small body (doji-like), bearish
                if (df['close'].iloc[i-2] > df['open'].iloc[i-2] and  # First bullish
                    abs(df['close'].iloc[i-1] - df['open'].iloc[i-1]) <= 0.1 * (df['high'].iloc[i-1] - df['low'].iloc[i-1]) and  # Second small body
                    df['close'].iloc[i] < df['open'].iloc[i]):  # Third bearish
                    evening_star.iloc[i] = True
            
            return evening_star
            
        except Exception as e:
            print(f"Error identifying evening star: {e}")
            return pd.Series(False, index=df.index)
    
    def _identify_three_white_soldiers(self, df):
        """Identify Three White Soldiers pattern"""
        try:
            three_white_soldiers = pd.Series(False, index=df.index)
            
            for i in range(2, len(df)):
                # Three consecutive bullish candles with higher closes
                if (df['close'].iloc[i] > df['open'].iloc[i] and  # Current bullish
                    df['close'].iloc[i-1] > df['open'].iloc[i-1] and  # Previous bullish
                    df['close'].iloc[i-2] > df['open'].iloc[i-2] and  # Two ago bullish
                    df['close'].iloc[i] > df['close'].iloc[i-1] and  # Higher close
                    df['close'].iloc[i-1] > df['close'].iloc[i-2]):  # Higher close
                    three_white_soldiers.iloc[i] = True
            
            return three_white_soldiers
            
        except Exception as e:
            print(f"Error identifying three white soldiers: {e}")
            return pd.Series(False, index=df.index)
    
    def _identify_three_black_crows(self, df):
        """Identify Three Black Crows pattern"""
        try:
            three_black_crows = pd.Series(False, index=df.index)
            
            for i in range(2, len(df)):
                # Three consecutive bearish candles with lower closes
                if (df['close'].iloc[i] < df['open'].iloc[i] and  # Current bearish
                    df['close'].iloc[i-1] < df['open'].iloc[i-1] and  # Previous bearish
                    df['close'].iloc[i-2] < df['open'].iloc[i-2] and  # Two ago bearish
                    df['close'].iloc[i] < df['close'].iloc[i-1] and  # Lower close
                    df['close'].iloc[i-1] < df['close'].iloc[i-2]):  # Lower close
                    three_black_crows.iloc[i] = True
            
            return three_black_crows
            
        except Exception as e:
            print(f"Error identifying three black crows: {e}")
            return pd.Series(False, index=df.index)
    
    def _identify_inside_bar(self, df):
        """Identify Inside Bar pattern"""
        try:
            inside_bar = pd.Series(False, index=df.index)
            
            for i in range(1, len(df)):
                # Current bar is completely inside previous bar
                if (df['high'].iloc[i] <= df['high'].iloc[i-1] and
                    df['low'].iloc[i] >= df['low'].iloc[i-1]):
                    inside_bar.iloc[i] = True
            
            return inside_bar
            
        except Exception as e:
            print(f"Error identifying inside bar: {e}")
            return pd.Series(False, index=df.index)
    
    def _identify_outside_bar(self, df):
        """Identify Outside Bar pattern"""
        try:
            outside_bar = pd.Series(False, index=df.index)
            
            for i in range(1, len(df)):
                # Current bar completely engulfs previous bar
                if (df['high'].iloc[i] >= df['high'].iloc[i-1] and
                    df['low'].iloc[i] <= df['low'].iloc[i-1]):
                    outside_bar.iloc[i] = True
            
            return outside_bar
            
        except Exception as e:
            print(f"Error identifying outside bar: {e}")
            return pd.Series(False, index=df.index)
    
    def _identify_pin_bar(self, df):
        """Identify Pin Bar pattern"""
        try:
            pin_bar = pd.Series(False, index=df.index)
            
            for i in range(len(df)):
                body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
                total_range = df['high'].iloc[i] - df['low'].iloc[i]
                upper_shadow = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])
                lower_shadow = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
                
                # Pin bar: long shadow, small body
                if (body_size <= 0.3 * total_range and
                    (upper_shadow >= 2 * body_size or lower_shadow >= 2 * body_size)):
                    pin_bar.iloc[i] = True
            
            return pin_bar
            
        except Exception as e:
            print(f"Error identifying pin bar: {e}")
            return pd.Series(False, index=df.index)
    
    def _identify_fakey(self, df):
        """Identify Fakey pattern (false breakout of inside bar)"""
        try:
            fakey = pd.Series(False, index=df.index)
            
            for i in range(2, len(df)):
                # Inside bar followed by breakout and reversal
                if (self._identify_inside_bar(df).iloc[i-1] and  # Previous is inside bar
                    df['high'].iloc[i] > df['high'].iloc[i-2] and  # Breaks above
                    df['close'].iloc[i] < df['open'].iloc[i]):  # Closes bearish
                    fakey.iloc[i] = True
                    
                elif (self._identify_inside_bar(df).iloc[i-1] and  # Previous is inside bar
                      df['low'].iloc[i] < df['low'].iloc[i-2] and  # Breaks below
                      df['close'].iloc[i] > df['open'].iloc[i]):  # Closes bullish
                    fakey.iloc[i] = True
            
            return fakey
            
        except Exception as e:
            print(f"Error identifying fakey: {e}")
            return pd.Series(False, index=df.index)
    
    def _identify_breakout_bar(self, df):
        """Identify Breakout Bar pattern"""
        try:
            breakout_bar = pd.Series(False, index=df.index)
            
            window = 20
            
            for i in range(window, len(df)):
                # Breaks above recent high
                if df['high'].iloc[i] > df['high'].iloc[i-window:i].max():
                    breakout_bar.iloc[i] = True
                    
                # Breaks below recent low
                elif df['low'].iloc[i] < df['low'].iloc[i-window:i].min():
                    breakout_bar.iloc[i] = True
            
            return breakout_bar
            
        except Exception as e:
            print(f"Error identifying breakout bar: {e}")
            return pd.Series(False, index=df.index)
    
    def _calculate_pattern_strength(self, df):
        """Calculate overall pattern strength"""
        try:
            pattern_strength = pd.Series(0.0, index=df.index)
            
            # Sum all pattern signals
            pattern_columns = [col for col in df.columns if any(pattern in col for pattern in 
                                                             ['doji', 'hammer', 'shooting_star', 'engulfing', 
                                                              'morning_star', 'evening_star', 'three_white_soldiers',
                                                              'three_black_crows', 'inside_bar', 'outside_bar',
                                                              'pin_bar', 'fakey', 'breakout_bar'])]
            
            for i in range(len(df)):
                strength = 0
                for col in pattern_columns:
                    if df[col].iloc[i]:
                        strength += 1
                
                # Normalize strength (0-1)
                pattern_strength.iloc[i] = min(strength / len(pattern_columns), 1.0)
            
            return pattern_strength
            
        except Exception as e:
            print(f"Error calculating pattern strength: {e}")
            return pd.Series(0.0, index=df.index)
    
    def analyze_order_flow(self, df):
        """
        Analyze order flow and market microstructure
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with order flow analysis
        """
        try:
            df = df.copy()
            
            # Volume analysis
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['volume_price_trend'] = df['volume'] * df['close'].pct_change()
            
            # Price impact
            df['price_impact'] = df['close'].pct_change() / df['volume_ratio']
            
            # Order flow imbalance
            df['order_flow_imbalance'] = self._calculate_order_flow_imbalance(df)
            
            # Market efficiency
            df['market_efficiency'] = self._calculate_market_efficiency(df)
            
            # Liquidity zones
            df['liquidity_zones'] = self._identify_liquidity_zones(df)
            
            return df
            
        except Exception as e:
            print(f"Error analyzing order flow: {e}")
            return df
    
    def _calculate_order_flow_imbalance(self, df):
        """Calculate order flow imbalance"""
        try:
            imbalance = pd.Series(0.0, index=df.index)
            
            for i in range(1, len(df)):
                # Volume-weighted price change
                vwap = (df['volume'].iloc[i] * df['close'].iloc[i] + 
                       df['volume'].iloc[i-1] * df['close'].iloc[i-1]) / (df['volume'].iloc[i] + df['volume'].iloc[i-1])
                
                imbalance.iloc[i] = (df['close'].iloc[i] - vwap) / vwap
            
            return imbalance
            
        except Exception as e:
            print(f"Error calculating order flow imbalance: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _calculate_market_efficiency(self, df):
        """Calculate market efficiency ratio"""
        try:
            efficiency = pd.Series(0.0, index=df.index)
            
            window = 20
            
            for i in range(window, len(df)):
                # Efficiency = |Price change| / Sum of absolute price changes
                price_change = abs(df['close'].iloc[i] - df['close'].iloc[i-window])
                sum_abs_changes = df['close'].pct_change().iloc[i-window:i].abs().sum()
                
                if sum_abs_changes > 0:
                    efficiency.iloc[i] = price_change / sum_abs_changes
            
            return efficiency
            
        except Exception as e:
            print(f"Error calculating market efficiency: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _identify_liquidity_zones(self, df):
        """Identify liquidity zones based on volume and price"""
        try:
            liquidity_zones = pd.Series(0.0, index=df.index)
            
            window = 20
            
            for i in range(window, len(df)):
                # High volume areas
                if df['volume'].iloc[i] > df['volume'].iloc[i-window:i].quantile(0.8):
                    liquidity_zones.iloc[i] = 1.0
                    
                # Low volume areas
                elif df['volume'].iloc[i] < df['volume'].iloc[i-window:i].quantile(0.2):
                    liquidity_zones.iloc[i] = -1.0
            
            return liquidity_zones
            
        except Exception as e:
            print(f"Error identifying liquidity zones: {e}")
            return pd.Series(0.0, index=df.index)
    
    def get_comprehensive_analysis(self, df):
        """
        Get comprehensive ICT and price action analysis
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with comprehensive analysis
        """
        try:
            # Perform all analyses
            df_ict = self.analyze_ict_levels(df)
            df_patterns = self.analyze_candlestick_patterns(df_ict)
            df_order_flow = self.analyze_order_flow(df_patterns)
            
            # Compile analysis results
            analysis = {
                'ict_levels': {
                    'fair_value_gaps': df_ict[['fvg_bullish', 'fvg_bearish']].sum().to_dict(),
                    'liquidity_levels': df_ict[['liquidity_high', 'liquidity_low']].sum().to_dict(),
                    'order_blocks': df_ict[['order_block_bullish', 'order_block_bearish']].sum().to_dict(),
                    'breakers': df_ict[['breaker_bullish', 'breaker_bearish']].sum().to_dict(),
                    'mitigation_blocks': df_ict['mitigation_block'].sum(),
                    'market_structure': df_ict['market_structure'].value_counts().to_dict()
                },
                'candlestick_patterns': {
                    'total_patterns': df_patterns['pattern_strength'].sum(),
                    'pattern_distribution': df_patterns[['doji', 'hammer', 'shooting_star', 'engulfing_bullish', 
                                                       'engulfing_bearish', 'morning_star', 'evening_star']].sum().to_dict()
                },
                'order_flow': {
                    'volume_trend': df_order_flow['volume_ratio'].iloc[-1],
                    'order_flow_imbalance': df_order_flow['order_flow_imbalance'].iloc[-1],
                    'market_efficiency': df_order_flow['market_efficiency'].iloc[-1]
                },
                'time_analysis': df_ict['time_analysis'].value_counts().to_dict(),
                'recommendations': self._generate_recommendations(df_order_flow)
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error in comprehensive analysis: {e}")
            return {}

if __name__ == "__main__":
    # Example usage
    analyzer = ICTPriceActionAnalyzer()
    print("ICT Price Action Analyzer initialized successfully!")
    print("Features:")
    print("- Fair Value Gaps (FVG)")
    print("- Liquidity Levels")
    print("- Order Blocks")
    print("- Breakers and Mitigation Blocks")
    print("- Advanced Candlestick Patterns")
    print("- Order Flow Analysis")
    print("- Market Structure Analysis")
    print("- Time-based Analysis")