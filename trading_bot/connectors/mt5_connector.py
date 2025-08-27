import socket
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MT5SocketClient:
    """
    A socket client to communicate with the MQL5 Socket Server EA.
    """
    def __init__(self, host='127.0.0.1', port=5555):
        self.host = host
        self.port = port
        self.socket = None
        self.buffer_size = 4096

    def connect(self):
        """
        Establishes a connection to the MQL5 server.
        Returns True if successful, False otherwise.
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            logging.info(f"Successfully connected to MQL5 server at {self.host}:{self.port}")
            return True
        except socket.error as e:
            logging.error(f"Failed to connect to MQL5 server: {e}")
            self.socket = None
            return False

    def disconnect(self):
        """
        Closes the connection to the MQL5 server.
        """
        if self.socket:
            self.socket.close()
            self.socket = None
            logging.info("Disconnected from MQL5 server.")

    def _send_request(self, command, params={}):
        """
        Sends a JSON request to the server and returns the response.
        """
        if not self.socket:
            logging.error("Not connected. Please call connect() first.")
            return None

        try:
            # Construct the request
            request = {
                "command": command,
                "params": params
            }
            request_json = json.dumps(request) + '\n' # Use newline as a delimiter

            # Send the request
            self.socket.sendall(request_json.encode('utf-8'))

            # Receive the response
            response_json = self.socket.recv(self.buffer_size).decode('utf-8')

            if not response_json:
                logging.error("Received an empty response from the server.")
                return None

            response = json.loads(response_json)
            return response

        except socket.error as e:
            logging.error(f"Socket error during request: {e}")
            self.disconnect() # Disconnect on error
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON response: {e}")
            logging.error(f"Raw response was: {response_json}")
            return None

    # --- Public API Methods ---
    def get_account_info(self):
        """
        Requests account information from the MQL5 server.
        """
        return self._send_request("get_account_info")

    def get_tick_data(self, symbol):
        """
        Requests the latest tick data for a symbol.
        """
        if not isinstance(symbol, str) or not symbol:
            logging.error("Invalid symbol provided.")
            return None
        return self._send_request("get_tick_data", {"symbol": symbol.upper()})

    def get_historical_data(self, symbol, timeframe, count):
        """
        Requests historical candle data for a symbol.
        :param symbol: e.g., "EURUSD"
        :param timeframe: e.g., "PERIOD_H1", "PERIOD_D1"
        :param count: number of candles to fetch
        """
        if not all([isinstance(symbol, str), symbol,
                    isinstance(timeframe, str), timeframe,
                    isinstance(count, int), count > 0]):
            logging.error("Invalid parameters for historical data.")
            return None

        params = {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "count": count
        }
        return self._send_request("get_historical_data", params)


# This space is intentionally left blank.
# The test code has been moved to main.py.
