import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from typing import Optional, Dict


def atr(df: pd.DataFrame, period: int = 14) -> float:
	if df.empty or len(df) < period + 1:
		return 0.0
	high_low = df['high'] - df['low']
	high_close = (df['high'] - df['close'].shift()).abs()
	low_close = (df['low'] - df['close'].shift()).abs()
	tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
	return float(tr.rolling(period).mean().iloc[-1])


def size_from_risk(balance: float, risk_pct: float, stop_pips: float, pip_value: float = 10.0) -> float:
	if stop_pips <= 0:
		return 0.01
	risk_amount = balance * risk_pct
	lots = max(risk_amount / (stop_pips * pip_value), 0.01)
	return float(min(lots, 2.0))


def current_balance() -> float:
	info = mt5.account_info()
	return float(info.balance) if info else 0.0


def execute_order(symbol: str, side: str, sl: Optional[float] = None, tp: Optional[float] = None, volume: float = 0.01) -> Dict:
	order_type = mt5.ORDER_TYPE_BUY if side.upper() == 'BUY' else mt5.ORDER_TYPE_SELL
	tick = mt5.symbol_info_tick(symbol)
	if not tick:
		return {"ok": False, "error": "no_tick"}
	price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
	request = {
		"action": mt5.TRADE_ACTION_DEAL,
		"symbol": symbol,
		"volume": volume,
		"type": order_type,
		"price": price,
		"deviation": 20,
		"magic": 987654,
		"comment": "mtf_trade",
		"type_time": mt5.ORDER_TIME_GTC,
		"type_filling": mt5.ORDER_FILLING_IOC,
	}
	if sl is not None:
		request["sl"] = sl
	if tp is not None:
		request["tp"] = tp
	res = mt5.order_send(request)
	ok = res and res.retcode == mt5.TRADE_RETCODE_DONE
	return {"ok": ok, "retcode": getattr(res, 'retcode', None), "price": price, "order": getattr(res, 'order', None)}