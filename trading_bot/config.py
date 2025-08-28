# -----------------
# Configuration File
# -----------------

# --- API Keys ---
# IMPORTANT: Replace "YOUR_API_KEY" with your actual key from newsapi.org
# To get a key, visit: https://newsapi.org/
NEWS_API_KEY = "YOUR_API_KEY"


# --- Trading Parameters ---
TARGET_SYMBOL = "EURUSD"

# --- Risk Management ---
# The percentage of your account balance to risk on a single trade.
# Example: 0.01 means you will risk 1% of your balance.
RISK_PER_TRADE = 0.01

# The desired risk-to-reward ratio for trades.
# Example: 2.0 means your take profit will be set at twice the distance of your stop loss.
RISK_REWARD_RATIO = 2.0

# The multiplier for the Average True Range (ATR) when calculating stop loss.
# A higher value results in a wider stop loss.
ATR_MULTIPLIER = 1.5
