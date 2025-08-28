import MetaTrader5 as mt5
import logging
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MT5Connector:
    """
    A connector to the MetaTrader 5 terminal using the official library.
    """
    def __init__(self):
        self.connected = False

    def initialize(self):
        """
        Initializes the connection to the MetaTrader 5 terminal.
        """
        if not mt5.initialize():
            logging.error(f"initialize() failed, error code = {mt5.last_error()}")
            self.connected = False
            return False

        logging.info("MetaTrader 5 connection initialized successfully.")
        self.connected = True
        return True

    def shutdown(self):
        """
        Shuts down the connection to the MetaTrader 5 terminal.
        """
        mt5.shutdown()
        logging.info("MetaTrader 5 connection shut down.")
        self.connected = False

    def get_account_info(self):
        """
        Retrieves account information.
        """
        if not self.connected:
            logging.error("Not connected to MetaTrader 5.")
            return None

        account_info = mt5.account_info()
        if account_info is None:
            logging.error(f"Failed to get account info, error code = {mt5.last_error()}")
            return None

        return account_info._asdict()

    def get_tick_data(self, symbol):
        """
        Retrieves the latest tick data for a given symbol.
        """
        if not self.connected:
            logging.error("Not connected to MetaTrader 5.")
            return None

        tick_info = mt5.symbol_info_tick(symbol)
        if tick_info is None:
            logging.error(f"Failed to get tick info for {symbol}, error code = {mt5.last_error()}")
            return None

        return tick_info._asdict()

    def get_historical_data(self, symbol, timeframe, count):
        """
        Retrieves historical data for a given symbol and timeframe.
        :param timeframe: Use MT5 ENUM_TIMEFRAMES, e.g., mt5.TIMEFRAME_H1
        """
        if not self.connected:
            logging.error("Not connected to MetaTrader 5.")
            return None

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            logging.error(f"Failed to get historical data for {symbol}, error code = {mt5.last_error()}")
            return None

        # Convert to a pandas DataFrame for easier use
        df = pd.DataFrame(rates)
        # Convert timestamp to a readable datetime format
        df['time'] = pd.to_datetime(df['time'], unit='s')

        return df

    def get_open_positions(self):
        """
        Retrieves all currently open positions.
        """
        if not self.connected:
            logging.error("Not connected to MetaTrader 5.")
            return pd.DataFrame() # Return empty DataFrame

        positions = mt5.positions_get()
        if positions is None:
            logging.error(f"Failed to get open positions, error code = {mt5.last_error()}")
            return pd.DataFrame()

        # Convert to a pandas DataFrame for easier viewing
        return pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())

    def open_trade(self, trade_type, symbol, lot_size, stop_loss, take_profit, magic_number=234567, comment="Python Bot Trade"):
        """
        Opens a new market trade.

        :param trade_type: 'BUY' or 'SELL'
        :param symbol: The symbol to trade (e.g., "EURUSD")
        :param lot_size: The volume of the trade
        :param stop_loss: The absolute price for the stop loss
        :param take_profit: The absolute price for the take profit
        :param magic_number: A unique identifier for trades from this bot
        :param comment: A comment for the trade
        :return: The result of the order send request.
        """
        if not self.connected:
            logging.error("Not connected to MetaTrader 5.")
            return None

        # Determine order type and price
        if trade_type.upper() == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        elif trade_type.upper() == 'SELL':
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            logging.error(f"Invalid trade type specified: {trade_type}")
            return None

        # Ensure SL and TP are floats
        sl = float(stop_loss) if stop_loss is not None else 0.0
        tp = float(take_profit) if take_profit is not None else 0.0

        # Build the request dictionary
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20, # Slippage
            "magic": magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC, # Good till cancelled
            "type_filling": mt5.ORDER_FILLING_IOC, # Immediate or Cancel
        }

        logging.info(f"Sending trade request: {request}")

        # Send the order to MetaTrader 5
        result = mt5.order_send(request)

        if result is None:
            logging.error(f"order_send failed, error code = {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order failed! retcode={result.retcode}, comment={result.comment}")
        else:
            logging.info(f"Order executed successfully! Ticket: {result.order}")

        return result
