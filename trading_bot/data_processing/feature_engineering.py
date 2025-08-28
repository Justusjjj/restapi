import pandas as pd
import pandas_ta as ta

def create_features(df, target_shift=-5):
    """
    Engineers features for the ML model. This includes technical indicators
    and the target variable for prediction.

    :param df: DataFrame with historical price data.
    :param target_shift: The number of periods to look into the future for the target.
                         A negative value means looking into the future.
    :return: A DataFrame with engineered features and a clean index.
    """

    # --- 1. Create Technical Indicators ---
    # Use pandas_ta to add a variety of indicators
    # You can customize this strategy with different indicators and parameters
    custom_strategy = ta.Strategy(
        name="ICT_ML_Features",
        description="RSI, MACD, Bollinger Bands for ML",
        ta=[
            {"kind": "rsi"},
            {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
            {"kind": "bbands", "length": 20, "std": 2},
        ]
    )
    df.ta.strategy(custom_strategy)

    # --- 2. Create the Target Variable ---
    # The goal is to predict if the price will be higher or lower in the future.
    # 'target_shift' periods from now.

    # Calculate the future close price
    df['future_close'] = df['close'].shift(target_shift)

    # Create the binary target: 1 if future price is higher, 0 if lower or same
    df['target'] = (df['future_close'] > df['close']).astype(int)

    # --- 3. Clean up the DataFrame ---
    # Indicators and shifting create NaN values. We need to remove them.
    df.dropna(inplace=True)

    # We don't need the future_close column for training
    df.drop(columns=['future_close'], inplace=True)

    # Reset index to be clean
    df.reset_index(drop=True, inplace=True)

    return df
