import logging
import os
import MetaTrader5 as mt5

# Local application imports
from connectors.mt5_connector import MT5Connector
from strategy.ict_concepts import find_swing_highs_lows, find_fair_value_gaps, find_order_blocks
from data_processing.feature_engineering import create_features
from ml_models.model_manager import train_model, evaluate_model, save_model
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_training_pipeline():
    """
    Executes the full pipeline to train, evaluate, and save the ML model.
    """
    logging.info("--- Starting ML Model Training Pipeline ---")

    connector = MT5Connector()

    if connector.initialize():
        # --- 1. Data Collection and Processing ---
        symbol = config.TARGET_SYMBOL
        timeframe = mt5.TIMEFRAME_M15
        # Fetch a large dataset for robust training
        count = 10000
        logging.info(f"Fetching {count} candles for {symbol}...")
        hist_data = connector.get_historical_data(symbol, timeframe, count)

        if hist_data is not None and not hist_data.empty:
            logging.info(f"Success! Received {len(hist_data)} rows.")

            logging.info("Processing data and engineering features...")
            hist_data = find_swing_highs_lows(hist_data)
            hist_data = find_fair_value_gaps(hist_data)
            hist_data = find_order_blocks(hist_data)
            featured_data = create_features(hist_data.copy())
            logging.info(f"Processing complete. Final dataset has {len(featured_data)} rows.")

            # --- 2. Model Training and Evaluation ---
            if not featured_data.empty:
                logging.info("Starting ML model training process...")
                trained_model, X_test, y_test = train_model(featured_data)

                # Evaluate the model on the test set
                evaluate_model(trained_model, X_test, y_test)

                # --- 3. Save the Trained Model ---
                logging.info("Saving the trained model...")
                model_dir = "trading_bot/ml_models/saved_models"
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                model_path = os.path.join(model_dir, f"random_forest_{symbol.lower()}_m15.joblib")

                save_model(trained_model, model_path)
            else:
                logging.error("Dataset is empty after processing, cannot train model.")

        else:
            logging.error(f"Failed to get historical data for {symbol}.")

        # --- Shutdown ---
        logging.info("Shutting down the MT5 connection.")
        connector.shutdown()
    else:
        logging.error("Connection to MetaTrader 5 failed. Training pipeline aborted.")

if __name__ == '__main__':
    run_training_pipeline()
    print("\nTraining script finished. Your model is now saved and ready for the main bot.")
