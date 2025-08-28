import pandas as pd
import numpy as np

"""
This module contains functions to identify various ICT (Inner Circle Trader)
and price action concepts from candlestick data.

Each function takes a pandas DataFrame with 'open', 'high', 'low', 'close' columns
and appends new columns with the identified features.
"""

def find_swing_highs_lows(df, n=2):
    """
    Finds swing highs and lows in the price data.
    A swing high is a candle with a high greater than the 'n' candles before and after it.
    A swing low is a candle with a low lower than the 'n' candles before and after it.

    :param df: pandas DataFrame with price data.
    :param n: The number of candles to the left and right to compare against.
    :return: DataFrame with 'swing_high' and 'swing_low' boolean columns.
    """
    df['swing_high'] = df['high'].rolling(window=2*n+1, center=True).apply(lambda x: x.iloc[n] == x.max(), raw=True)
    df['swing_low'] = df['low'].rolling(window=2*n+1, center=True).apply(lambda x: x.iloc[n] == x.min(), raw=True)

    return df

def find_fair_value_gaps(df):
    """
    Identifies Fair Value Gaps (FVGs) or imbalances.
    A bullish FVG occurs when there's a gap between the high of candle (i-1) and the low of candle (i+1).
    A bearish FVG occurs when there's a gap between the low of candle (i-1) and the high of candle (i+1).

    Appends 'fvg_bullish_top', 'fvg_bullish_bottom', 'fvg_bearish_top', 'fvg_bearish_bottom' columns.
    These columns store the price levels of the identified gaps.
    """
    bullish_fvg_top = []
    bullish_fvg_bottom = []
    bearish_fvg_top = []
    bearish_fvg_bottom = []

    for i in range(1, len(df) - 1):
        # Bullish FVG
        if df['low'][i+1] > df['high'][i-1]:
            bullish_fvg_bottom.append(df['high'][i-1])
            bullish_fvg_top.append(df['low'][i+1])
        else:
            bullish_fvg_bottom.append(np.nan)
            bullish_fvg_top.append(np.nan)

        # Bearish FVG
        if df['high'][i+1] < df['low'][i-1]:
            bearish_fvg_bottom.append(df['high'][i+1])
            bearish_fvg_top.append(df['low'][i-1])
        else:
            bearish_fvg_bottom.append(np.nan)
            bearish_fvg_top.append(np.nan)

    # Append results for all but the first and last rows
    df['fvg_bullish_top'] = [np.nan] + bullish_fvg_top + [np.nan]
    df['fvg_bullish_bottom'] = [np.nan] + bullish_fvg_bottom + [np.nan]
    df['fvg_bearish_top'] = [np.nan] + bearish_fvg_top + [np.nan]
    df['fvg_bearish_bottom'] = [np.nan] + bearish_fvg_bottom + [np.nan]

    return df

def find_order_blocks(df):
    """
    Identifies a simplified version of Order Blocks (OBs).
    - A bullish OB is the last down-candle before a strong move up.
    - A bearish OB is the last up-candle before a strong move down.

    This simplified version looks for engulfing candles as a proxy for a "strong move".
    Appends 'ob_bullish_top', 'ob_bullish_bottom', 'ob_bearish_top', 'ob_bearish_bottom'.
    """
    ob_bullish_top = []
    ob_bullish_bottom = []
    ob_bearish_top = []
    ob_bearish_bottom = []

    for i in range(1, len(df)):
        # Bullish Order Block: A down-candle followed by a strong up-candle that engulfs it.
        is_down_candle = df['close'][i-1] < df['open'][i-1]
        is_up_candle_engulfing = df['open'][i] < df['low'][i-1] and df['close'][i] > df['high'][i-1]

        if is_down_candle and is_up_candle_engulfing:
            ob_bullish_top.append(df['high'][i-1])
            ob_bullish_bottom.append(df['low'][i-1])
        else:
            ob_bullish_top.append(np.nan)
            ob_bullish_bottom.append(np.nan)

        # Bearish Order Block: An up-candle followed by a strong down-candle that engulfs it.
        is_up_candle = df['close'][i-1] > df['open'][i-1]
        is_down_candle_engulfing = df['open'][i] > df['high'][i-1] and df['close'][i] < df['low'][i-1]

        if is_up_candle and is_down_candle_engulfing:
            ob_bearish_top.append(df['high'][i-1])
            ob_bearish_bottom.append(df['low'][i-1])
        else:
            ob_bearish_top.append(np.nan)
            ob_bearish_bottom.append(np.nan)

    df['ob_bullish_top'] = [np.nan] + ob_bullish_top
    df['ob_bullish_bottom'] = [np.nan] + ob_bullish_bottom
    df['ob_bearish_top'] = [np.nan] + ob_bearish_top
    df['ob_bearish_bottom'] = [np.nan] + ob_bearish_bottom

    return df
