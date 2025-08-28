import MetaTrader5 as mt5
import pandas_ta as ta
import logging
import config

def calculate_lot_size(account_balance, risk_percentage, stop_loss_pips, symbol):
    """
    Calculates the appropriate lot size for a trade based on risk percentage.

    :param account_balance: The current account balance.
    :param risk_percentage: The percentage of the account to risk (e.g., 0.01 for 1%).
    :param stop_loss_pips: The stop loss distance in pips.
    :param symbol: The trading symbol.
    :return: The calculated lot size, or None if an error occurs.
    """
    if stop_loss_pips <= 0:
        logging.error("Stop loss in pips must be greater than zero.")
        return None

    # Get symbol information
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        logging.error(f"Could not get symbol info for {symbol}")
        return None

    # Calculate the value of 1 pip
    tick_size = symbol_info.trade_tick_size
    tick_value = symbol_info.trade_tick_value
    pip_value_in_quote_currency = tick_value / tick_size * (10 * tick_size) # for 5-digit brokers

    # Amount to risk in account currency
    risk_amount = account_balance * risk_percentage

    # Calculate lot size
    # Formula: Lot Size = (Amount to Risk) / (Stop Loss in Pips * Pip Value)
    lot_size = risk_amount / (stop_loss_pips * pip_value_in_quote_currency)

    # Normalize the lot size to the allowed step
    volume_step = symbol_info.volume_step
    lot_size = round(lot_size / volume_step) * volume_step

    # Clamp the lot size to the min and max allowed volumes
    min_volume = symbol_info.volume_min
    max_volume = symbol_info.volume_max

    if lot_size < min_volume:
        logging.warning(f"Calculated lot size {lot_size} is less than min volume {min_volume}. Adjusting to min volume.")
        lot_size = min_volume
    if lot_size > max_volume:
        logging.warning(f"Calculated lot size {lot_size} is greater than max volume {max_volume}. Adjusting to max volume.")
        lot_size = max_volume

    logging.info(f"Calculated Lot Size: {lot_size:.2f} for a risk of {risk_amount:.2f} USD")
    return lot_size


def calculate_stop_loss(entry_price, trade_type, atr_value, atr_multiplier=1.5):
    """
    Calculates a dynamic stop loss based on the Average True Range (ATR).

    :param entry_price: The entry price of the trade.
    :param trade_type: 'BUY' or 'SELL'.
    :param atr_value: The current ATR value.
    :param atr_multiplier: The factor to multiply the ATR by.
    :return: The calculated absolute stop loss price.
    """
    if trade_type.upper() == 'BUY':
        stop_loss_price = entry_price - (atr_value * atr_multiplier)
    elif trade_type.upper() == 'SELL':
        stop_loss_price = entry_price + (atr_value * atr_multiplier)
    else:
        logging.error(f"Invalid trade type for SL calculation: {trade_type}")
        return None

    # Normalize to the symbol's digit precision
    digits = mt5.symbol_info(config.TARGET_SYMBOL).digits
    return round(stop_loss_price, digits)


def calculate_take_profit(entry_price, stop_loss_price, trade_type, risk_reward_ratio):
    """
    Calculates the take profit level based on a fixed risk-to-reward ratio.

    :param entry_price: The entry price of the trade.
    :param stop_loss_price: The stop loss price of the trade.
    :param trade_type: 'BUY' or 'SELL'.
    :param risk_reward_ratio: The desired risk-to-reward ratio (e.g., 2 for 1:2).
    :return: The calculated absolute take profit price.
    """
    risk_distance = abs(entry_price - stop_loss_price)

    if trade_type.upper() == 'BUY':
        take_profit_price = entry_price + (risk_distance * risk_reward_ratio)
    elif trade_type.upper() == 'SELL':
        take_profit_price = entry_price - (risk_distance * risk_reward_ratio)
    else:
        logging.error(f"Invalid trade type for TP calculation: {trade_type}")
        return None

    # Normalize to the symbol's digit precision
    digits = mt5.symbol_info(config.TARGET_SYMBOL).digits
    return round(take_profit_price, digits)
