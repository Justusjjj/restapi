from strategy.main_strategy import run_strategy_check
import logging

def main():
    """
    Main entry point for the trading bot application.

    This script runs the main trading strategy. To train the model,
    please run the `train_model_script.py` separately.
    """
    print("--- Starting Trading Bot ---")
    print("To train a new model, run: python3 trading_bot/train_model_script.py")
    print("To run the live strategy, this script will now proceed...")

    try:
        run_strategy_check()
    except Exception as e:
        # Set up logging specifically for critical errors if not already configured
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.critical(f"A critical error occurred in the main application: {e}", exc_info=True)

if __name__ == '__main__':
    main()
