# Algorithmic Trading Bot - Phase 1 Test Procedure

This document outlines the steps to test the successful completion of Phase 1: The socket-based connection between the Python application and the MetaTrader 5 terminal.

## Prerequisites

Before you begin, please ensure you have completed the following:

1.  **MetaTrader 5 Terminal is installed.**
2.  The MQL5 server code has been copied and compiled successfully in MetaEditor. The file is located at `mql5_server/PythonSocketServer.mq5`.
3.  The Python environment has the necessary packages installed (currently none, as we are using standard libraries).

## Step 1: Run the MQL5 Server Expert Advisor

1.  Open your **MetaTrader 5 terminal**.
2.  In the **Navigator** window, find `PythonSocketServer` under the **"Expert Advisors"** section.
3.  Drag the `PythonSocketServer` EA onto any chart.
4.  A settings window will pop up. In the **"Common"** tab, ensure **"Allow algorithmic trading"** is checked.
5.  In the **"Inputs"** tab, confirm the `InpServerIP` is `127.0.0.1` and `InpServerPort` is `5555`.
6.  Click **OK**.
7.  **Verification:** You should see a smiley face icon in the top-right corner of the chart. In the terminal's **"Experts"** tab below, you should see the message: `Server started on 127.0.0.1:5555`.

**The server is now running and waiting for a connection.**

## Step 2: Run the Python Client Application

1.  Open a terminal or command prompt.
2.  Navigate to the root directory of this project.
3.  Run the main application script using the following command:

    ```bash
    python3 trading_bot/main.py
    ```

## Step 3: Verify the Output

If the connection is successful, you will see output in your terminal similar to the following. The values for balance, equity, prices, and timestamps will differ based on your account and current market conditions.

```
--- Python Socket Client Test ---
This script will attempt to connect to the MQL5 server.
Please ensure the PythonSocketServer EA is running in your MT5 terminal.
2023-10-27 10:30:00,123 - INFO - Successfully connected to MQL5 server at 127.0.0.1:5555

1. Requesting Account Information...
   Success! Received data:
   {'balance': 10000.00, 'equity': 10000.00, 'profit': 0.00, 'leverage': 100}

2. Requesting Tick Data for EURUSD...
   Success! Received data:
   {'symbol': 'EURUSD', 'ask': 1.05550, 'bid': 1.05545, 'time': 1698388200}

3. Requesting 10 PERIOD_H1 candles for GBPUSD...
   Success! Received data:
   Total candles received: 10
   First candle: {'time': ..., 'open': ..., 'high': ..., 'low': ..., 'close': ..., 'volume': ...}
   Last candle: {'time': ..., 'open': ..., 'high': ..., 'low': ..., 'close': ..., 'volume': ...}
2023-10-27 10:30:02,456 - INFO - Disconnected from MQL5 server.
```

## Troubleshooting

*   **`ConnectionRefusedError` in Python:** This is the most common error. It means the Python script could not find the MQL5 server.
    *   **Solution:** Double-check that the `PythonSocketServer` EA is running on a chart in MT5 and that the smiley face icon is present. Also, ensure the IP and Port match.
*   **Script hangs or times out:**
    *   **Solution:** Check your firewall settings to ensure it is not blocking the connection on port 5555. Also, ensure you have enabled "Allow WebRequest for listed URL" in MT5's options and added `http://127.0.0.1`.
*   **"Invalid symbol" error from the server:**
    *   **Solution:** Make sure the symbol you are requesting (e.g., "EURUSD") is available and enabled in your MT5 Market Watch.
