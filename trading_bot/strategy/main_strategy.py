import logging
import os
import pandas as pd
import pandas_ta as ta
import MetaTrader5 as mt5

# Local application imports
from connectors.mt5_connector import MT5Connector
from connectors.news_connector import NewsConnector
from strategy.ict_concepts import find_swing_highs_lows, find_fair_value_gaps, find_order_blocks
from data_processing.feature_engineering import create_features
from ml_models.model_manager import load_model
from strategy.risk_manager import calculate_lot_size, calculate_stop_loss, calculate_take_profit
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_strategy_check():
    """
    This is the main orchestration function for the trading strategy.
    It brings all components together to make a trading decision.
    """
    logging.info("--- Starting Strategy Check ---")

    # --- 1. Initialize Connectors and Load Model ---
    mt5_connector = MT5Connector()
    news_connector = NewsConnector()
    model_path = f"trading_bot/ml_models/saved_models/random_forest_{config.TARGET_SYMBOL.lower()}_m15.joblib"

    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}. Please train the model first.")
        return

    ml_model = load_model(model_path)
    if not ml_model:
        return

    # --- 2. News Filter ---
    logging.info("Checking for high-impact news...")
    latest_news = news_connector.get_latest_forex_news()
    if news_connector.is_high_impact_news_imminent(latest_news):
        logging.warning("High-impact news detected. Halting trading activity.")
        return

    # --- 3. Fetch and Process Market Data ---
    if not mt5_connector.initialize():
        return

    logging.info(f"Fetching data for {config.TARGET_SYMBOL}...")
    hist_data = mt5_connector.get_historical_data(config.TARGET_SYMBOL, mt5.TIMEFRAME_M15, 2000)

    if hist_data is None or hist_data.empty:
        logging.error("Could not fetch historical data.")
        mt5_connector.shutdown()
        return

    # Apply full data processing pipeline
    # Calculate ATR first for risk management
    hist_data.ta.atr(append=True)
    current_atr = hist_data['ATRr_14'].iloc[-1]

    processed_data = find_swing_highs_lows(hist_data)
    processed_data = find_fair_value_gaps(processed_data)
    processed_data = find_order_blocks(processed_data)
    featured_data = create_features(processed_data.copy())

    # Get the latest data point for decision making
    latest_candle = featured_data.iloc[-1]
    logging.info(f"Latest candle time: {latest_candle['time']}")

    # --- 4. Primary Signal (ICT Concepts) ---
    logging.info("Checking for primary ICT signals...")
    primary_signal = "HOLD"

    # Example: Check if the latest close is inside a Fair Value Gap
    if not pd.isna(latest_candle['fvg_bullish_bottom']) and latest_candle['close'] > latest_candle['fvg_bullish_bottom']:
        primary_signal = "BUY"
        logging.info("Primary Signal: BUY - Price is in a bullish FVG.")
    elif not pd.isna(latest_candle['fvg_bearish_top']) and latest_candle['close'] < latest_candle['fvg_bearish_top']:
        primary_signal = "SELL"
        logging.info("Primary Signal: SELL - Price is in a bearish FVG.")
    else:
        logging.info("No primary ICT signal found.")

    # --- 5. ML Confirmation Signal ---
    if primary_signal != "HOLD":
        logging.info("Primary signal found. Seeking ML confirmation...")

        # Prepare features for prediction
        features_for_prediction = latest_candle.drop(labels=['time', 'open', 'high', 'low', 'close', 'target'])
        features_df = pd.DataFrame([features_for_prediction])
        features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        ml_prediction = ml_model.predict(features_df)[0]
        ml_confidence = ml_model.predict_proba(features_df)[0]

        logging.info(f"ML Model Prediction: {'UP' if ml_prediction == 1 else 'DOWN'} (Confidence: {max(ml_confidence):.2f})")

        # --- 6. Final Trade Decision ---
        trade_executed = False
        if (primary_signal == "BUY" and ml_prediction == 1) or \
           (primary_signal == "SELL" and ml_prediction == 0):

            logging.info("Trade signal confirmed. Proceeding with risk management and execution...")

            # Get account and price info for execution
            account_info = mt5_connector.get_account_info()
            tick_data = mt5_connector.get_tick_data(config.TARGET_SYMBOL)

            if account_info and tick_data:
                entry_price = tick_data['ask'] if primary_signal == "BUY" else tick_data['bid']

                # Calculate SL, TP, and Lot Size
                sl_price = calculate_stop_loss(entry_price, primary_signal, current_atr, config.ATR_MULTIPLIER)
                tp_price = calculate_take_profit(entry_price, sl_price, primary_signal, config.RISK_REWARD_RATIO)
                sl_pips = abs(entry_price - sl_price) / (mt5.symbol_info(config.TARGET_SYMBOL).point * 10)
                lot_size = calculate_lot_size(account_info['balance'], config.RISK_PER_TRADE, sl_pips, config.TARGET_SYMBOL)

                if lot_size:
                    # Execute the trade
                    logging.critical(f"EXECUTING {primary_signal} TRADE on {config.TARGET_SYMBOL} | Lot Size: {lot_size}, SL: {sl_price}, TP: {tp_price}")
                    trade_result = mt5_connector.open_trade(primary_signal, config.TARGET_SYMBOL, lot_size, sl_price, tp_price)
                    if trade_result and trade_result.retcode == mt5.TRADE_RETCODE_DONE:
                        trade_executed = True
                else:
                    logging.error("Failed to calculate lot size. Halting trade execution.")
            else:
                logging.error("Failed to retrieve account or tick data for trade execution.")

        if not trade_executed:
            logging.info("TRADE DECISION: HOLD. Reason: No confirmed signal or execution failed.")
    else:
        logging.info("TRADE DECISION: HOLD. Reason: No primary signal.")

    # --- 7. Shutdown ---
    mt5_connector.shutdown()
    logging.info("--- Strategy Check Complete ---")
