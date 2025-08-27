from connectors.mt5_connector import MT5SocketClient

def main():
    """
    Main function to run the trading bot application.
    For Phase 1, this function runs a test of the connection.
    """
    print("--- Python Algo-Trading Bot ---")
    print("--- Phase 1: Connection Test ---")
    print("This script will attempt to connect to the MQL5 server.")
    print("Please ensure the PythonSocketServer EA is running in your MT5 terminal as per the README.md.")

    # Create and connect the client
    client = MT5SocketClient()
    if client.connect():

        # --- Test 1: Get Account Info ---
        print("\n1. Requesting Account Information...")
        acc_info_response = client.get_account_info()
        if acc_info_response and acc_info_response.get('status') == 'success':
            print("   Success! Received data:")
            print(f"   {acc_info_response.get('data')}")
        else:
            print("   Failed to get account info.")
            print(f"   Response: {acc_info_response}")

        # --- Test 2: Get Tick Data ---
        test_symbol = "EURUSD"
        print(f"\n2. Requesting Tick Data for {test_symbol}...")
        tick_data_response = client.get_tick_data(test_symbol)
        if tick_data_response and tick_data_response.get('status') == 'success':
            print("   Success! Received data:")
            print(f"   {tick_data_response.get('data')}")
        else:
            print(f"   Failed to get tick data for {test_symbol}.")
            print(f"   Response: {tick_data_response}")

        # --- Test 3: Get Historical Data ---
        test_symbol_hist = "GBPUSD"
        test_timeframe = "PERIOD_H1"
        test_count = 10
        print(f"\n3. Requesting {test_count} {test_timeframe} candles for {test_symbol_hist}...")
        hist_data_response = client.get_historical_data(test_symbol_hist, test_timeframe, test_count)
        if hist_data_response and hist_data_response.get('status') == 'success':
            print("   Success! Received data:")
            # Print first and last candle to avoid spamming the console
            data = hist_data_response.get('data')
            if data and isinstance(data, list) and len(data) > 0:
                print(f"   Total candles received: {len(data)}")
                print(f"   First candle: {data[0]}")
                print(f"   Last candle: {data[-1]}")
            else:
                print(f"   Data: {data}")
        else:
            print(f"   Failed to get historical data for {test_symbol_hist}.")
            print(f"   Response: {hist_data_response}")

        # Disconnect the client
        client.disconnect()
    else:
        print("\nConnection failed. Please see README.md for troubleshooting steps.")

if __name__ == '__main__':
    main()
