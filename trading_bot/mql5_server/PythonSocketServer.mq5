//+------------------------------------------------------------------+
//|                                       PythonSocketServer.mq5 |
//|                      Copyright 2023, Your Name/Company         |
//|                                      https://www.example.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023"
#property link      "https://www.example.com"
#property version   "1.00"
#property description "A TCP socket server to communicate with a Python client."

//--- Input parameters for the server
input string InpServerIP      = "127.0.0.1"; // Server IP address
input int    InpServerPort    = 5555;        // Server port
input int    InpTimerInterval = 100;         // Timer interval in milliseconds

//--- Global variables
int  g_listen_socket = -1;      // Handle for the listening socket
int  g_client_sockets[];        // Array to store client socket handles
int  g_timer_interval = 100;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    //--- Store timer interval from inputs
    g_timer_interval = InpTimerInterval;

    //--- Create the listening socket
    g_listen_socket = SocketCreate();
    if(g_listen_socket == INVALID_HANDLE)
    {
        Print("Failed to create socket, error ", GetLastError());
        return(INIT_FAILED);
    }

    //--- Bind the socket to the IP and port
    if(!SocketBind(g_listen_socket, InpServerIP, InpServerPort))
    {
        Print("Failed to bind socket, error ", GetLastError());
        SocketClose(g_listen_socket);
        return(INIT_FAILED);
    }

    //--- Start listening for incoming connections
    if(!SocketListen(g_listen_socket))
    {
        Print("Failed to listen on socket, error ", GetLastError());
        SocketClose(g_listen_socket);
        return(INIT_FAILED);
    }

    //--- Set the socket to non-blocking mode
    SocketSetTimeout(g_listen_socket, 1, true);

    //--- Set up the timer to check for connections and data
    EventSetTimer(g_timer_interval);

    Print("Server started on ", InpServerIP, ":", InpServerPort);
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    //--- Kill the timer
    EventKillTimer();

    //--- Close all client sockets
    int total_clients = ArraySize(g_client_sockets);
    for(int i = 0; i < total_clients; i++)
    {
        SocketClose(g_client_sockets[i]);
    }
    ArrayResize(g_client_sockets, 0);

    //--- Close the listening socket
    if(g_listen_socket != INVALID_HANDLE)
    {
        SocketClose(g_listen_socket);
    }

    Print("Server stopped.");
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
    //--- Check for new client connections
    CheckForNewConnections();

    //--- Check for data from existing clients
    CheckForClientData();
}

//+------------------------------------------------------------------+
//| Check for new client connections                               |
//+------------------------------------------------------------------+
void CheckForNewConnections()
{
    int client_socket = SocketAccept(g_listen_socket);
    if(client_socket != INVALID_HANDLE)
    {
        int total_clients = ArraySize(g_client_sockets);
        ArrayResize(g_client_sockets, total_clients + 1);
        g_client_sockets[total_clients] = client_socket;
        Print("New client connected, handle: ", client_socket);
    }
}

//+------------------------------------------------------------------+
//| Check for data from existing clients                             |
//+------------------------------------------------------------------+
void CheckForClientData()
{
    //--- Buffer for incoming data
    char buffer[];
    string request_string = "";

    //--- Iterate through all connected clients
    for(int i = ArraySize(g_client_sockets) - 1; i >= 0; i--)
    {
        int client_socket = g_client_sockets[i];
        int bytes_available = SocketIsReadable(client_socket);

        if(bytes_available > 0)
        {
            int bytes_read = SocketRead(client_socket, buffer, bytes_available);
            if(bytes_read > 0)
            {
                request_string = CharArrayToString(buffer, 0, bytes_read);
                Print("Received from client ", client_socket, ": ", request_string);

                // Process the request and send a response
                string response = ProcessRequest(request_string);
                SocketSend(client_socket, StringToCharArray(response), StringLen(response));
            }
        }
        else if(!SocketIsConnected(client_socket))
        {
            Print("Client ", client_socket, " disconnected.");
            SocketClose(client_socket);
            // Remove from array
            for(int j = i; j < ArraySize(g_client_sockets) - 1; j++)
            {
                g_client_sockets[j] = g_client_sockets[j+1];
            }
            ArrayResize(g_client_sockets, ArraySize(g_client_sockets) - 1);
        }
    }
}

//+------------------------------------------------------------------+
//| Process the client's request                                     |
//+------------------------------------------------------------------+
string ProcessRequest(string request)
{
    //--- Basic JSON parsing
    string command = GetJsonValue(request, "command");

    if(command == "get_account_info")
    {
        return HandleGetAccountInfo();
    }
    else if(command == "get_tick_data")
    {
        string symbol = GetJsonValue(request, "symbol");
        return HandleGetTickData(symbol);
    }
    else if(command == "get_historical_data")
    {
        string symbol = GetJsonValue(request, "symbol");
        string timeframe_str = GetJsonValue(request, "timeframe");
        long count = (long)StringToInteger(GetJsonValue(request, "count"));
        return HandleGetHistoricalData(symbol, timeframe_str, count);
    }
    // Add more command handlers here in the future

    return CreateJsonResponse("error", "\"Unknown command\"");
}

//+------------------------------------------------------------------+
//| Command Handlers                                                 |
//+------------------------------------------------------------------+
string HandleGetAccountInfo()
{
    string data = "{";
    data += "\"balance\":" + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2) + ",";
    data += "\"equity\":" + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2) + ",";
    data += "\"profit\":" + DoubleToString(AccountInfoDouble(ACCOUNT_PROFIT), 2) + ",";
    data += "\"leverage\":" + IntegerToString((long)AccountInfoInteger(ACCOUNT_LEVERAGE));
    data += "}";
    return CreateJsonResponse("success", data);
}

// Helper to convert string timeframe to ENUM_TIMEFRAMES
ENUM_TIMEFRAMES StringToTimeframe(string timeframe_str)
{
    if(timeframe_str == "PERIOD_M1") return(PERIOD_M1);
    if(timeframe_str == "PERIOD_M5") return(PERIOD_M5);
    if(timeframe_str == "PERIOD_M15") return(PERIOD_M15);
    if(timeframe_str == "PERIOD_M30") return(PERIOD_M30);
    if(timeframe_str == "PERIOD_H1") return(PERIOD_H1);
    if(timeframe_str == "PERIOD_H4") return(PERIOD_H4);
    if(timeframe_str == "PERIOD_D1") return(PERIOD_D1);
    if(timeframe_str == "PERIOD_W1") return(PERIOD_W1);
    if(timeframe_str == "PERIOD_MN1") return(PERIOD_MN1);
    return((ENUM_TIMEFRAMES)-1); // Return an invalid value
}

string HandleGetHistoricalData(string symbol, string timeframe_str, ulong count)
{
    if(symbol == "" || timeframe_str == "" || count <= 0)
    {
        return CreateJsonResponse("error", "\"Invalid parameters for historical data\"");
    }

    ENUM_TIMEFRAMES timeframe = StringToTimeframe(timeframe_str);
    if(timeframe == (ENUM_TIMEFRAMES)-1)
    {
        return CreateJsonResponse("error", "\"Invalid timeframe string: " + timeframe_str + "\"");
    }

    MqlRates rates[];
    // Set response array from newest to oldest
    ArraySetAsSeries(rates, true);
    int copied = CopyRates(symbol, timeframe, 0, (int)count, rates);

    if(copied <= 0)
    {
        return CreateJsonResponse("error", "\"Failed to copy rates, error " + (string)GetLastError() + "\"");
    }

    string data = "[";
    for(int i = 0; i < copied; i++)
    {
        data += "{";
        data += "\"time\":" + (string)rates[i].time + ",";
        data += "\"open\":" + DoubleToString(rates[i].open, _Digits) + ",";
        data += "\"high\":" + DoubleToString(rates[i].high, _Digits) + ",";
        data += "\"low\":" + DoubleToString(rates[i].low, _Digits) + ",";
        data += "\"close\":" + DoubleToString(rates[i].close, _Digits) + ",";
        data += "\"volume\":" + (string)rates[i].tick_volume;
        data += "}";
        if(i < copied - 1)
        {
            data += ",";
        }
    }
    data += "]";

    return CreateJsonResponse("success", data);
}

string HandleGetTickData(string symbol)
{
    if(symbol == "")
    {
        return CreateJsonResponse("error", "\"Symbol not provided\"");
    }

    MqlTick tick;
    if(!SymbolInfoTick(symbol, tick))
    {
        return CreateJsonResponse("error", "\"Invalid symbol or no tick data\"");
    }

    string data = "{";
    data += "\"symbol\":\"" + symbol + "\",";
    data += "\"ask\":" + DoubleToString(tick.ask, _Digits) + ",";
    data += "\"bid\":" + DoubleToString(tick.bid, _Digits) + ",";
    data += "\"time\":" + IntegerToString(tick.time);
    data += "}";
    return CreateJsonResponse("success", data);
}


//+------------------------------------------------------------------+
//| JSON Helper Functions (Basic String Manipulation)                |
//+------------------------------------------------------------------+
string CreateJsonResponse(string status, string data)
{
    return "{\"status\":\"" + status + "\",\"data\":" + data + "}\n"; // Add newline as a delimiter
}

string GetJsonValue(string json, string key)
{
    string search_key = "\"" + key + "\":";
    int key_pos = StringFind(json, search_key);
    if(key_pos == -1) return "";

    int value_start = key_pos + StringLen(search_key);

    // Find the start of the value (skip whitespace)
    while(StringGetCharacter(json, value_start) == ' ' || StringGetCharacter(json, value_start) == '\t')
    {
        value_start++;
    }

    char first_char = StringGetCharacter(json, value_start);
    int value_end;

    if(first_char == '"') // It's a string
    {
        value_start++; // Skip the opening quote
        value_end = StringFind(json, "\"", value_start);
        if(value_end == -1) return "";
        return StringSubstr(json, value_start, value_end - value_start);
    }
    else // It's a number or a nested object/array
    {
        int end_comma = StringFind(json, ",", value_start);
        int end_brace = StringFind(json, "}", value_start);

        if(end_comma == -1 && end_brace == -1) value_end = StringLen(json);
        else if(end_comma == -1) value_end = end_brace;
        else if(end_brace == -1) value_end = end_comma;
        else value_end = MathMin(end_comma, end_brace);

        return StringSubstr(json, value_start, value_end - value_start);
    }

    return "";
}
//+------------------------------------------------------------------+
