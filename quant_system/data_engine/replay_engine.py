"""
Replay Engine
=============

Event generator for backtesting and real-time simulation.
Provides efficient data streaming with configurable time steps.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Iterator, Tuple
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MarketEvent:
    """Represents a market data event."""
    timestamp: datetime
    instrument: str
    event_type: str  # 'tick', 'bar', 'trade'
    data: Dict[str, float]
    
    def __post_init__(self):
        """Ensure timestamp is timezone-aware."""
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)


class ReplayEngine:
    """
    Replays historical market data for backtesting and simulation.
    
    Features:
    - Configurable time steps (tick, 1s, 1m, etc.)
    - Multiple instrument support
    - Event-driven architecture
    - Memory-efficient streaming
    """
    
    def __init__(self, data_storage, config: Optional[Dict] = None):
        """
        Initialize ReplayEngine.
        
        Args:
            data_storage: DataStorage instance
            config: Configuration parameters
        """
        self.data_storage = data_storage
        self.config = config or self._default_config()
        self.current_time = None
        self.end_time = None
        self.instruments = []
        self.data_cache = {}
        
    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            'time_step': '1s',  # Default time step
            'start_time': None,  # Will be set when starting replay
            'end_time': None,    # Will be set when starting replay
            'instruments': [],   # Will be set when starting replay
            'cache_size': 1000,  # Number of rows to cache per instrument
        }
        
    def start_replay(self, instruments: List[str], 
                    start_time: Union[str, datetime],
                    end_time: Union[str, datetime],
                    time_step: str = '1s') -> None:
        """
        Start data replay for specified instruments and time range.
        
        Args:
            instruments: List of instrument symbols
            start_time: Start time for replay
            end_time: End time for replay
            time_step: Time step for replay ('tick', '1s', '1m', etc.)
        """
        self.instruments = instruments
        self.current_time = pd.to_datetime(start_time).tz_localize('UTC')
        self.end_time = pd.to_datetime(end_time).tz_localize('UTC')
        self.config['time_step'] = time_step
        
        # Load initial data for each instrument
        self._load_initial_data()
        
        logger.info(f"Started replay: {instruments} from {self.current_time} to {self.end_time}")
        
    def _load_initial_data(self) -> None:
        """Load initial data for all instruments."""
        self.data_cache = {}
        
        for instrument in self.instruments:
            try:
                if self.config['time_step'] == 'tick':
                    # Load tick data for the date range
                    start_date = self.current_time.date()
                    end_date = self.end_time.date()
                    
                    dates = pd.date_range(start_date, end_date, freq='D')
                    all_data = []
                    
                    for date in dates:
                        try:
                            df = self.data_storage.load_tick_data(instrument, date.strftime('%Y-%m-%d'))
                            all_data.append(df)
                        except FileNotFoundError:
                            continue
                            
                    if all_data:
                        self.data_cache[instrument] = pd.concat(all_data).sort_index()
                    else:
                        logger.warning(f"No tick data found for {instrument}")
                        
                else:
                    # Load OHLCV data for the timeframe
                    df = self.data_storage.load_ohlcv(
                        instrument, 
                        self.config['time_step'],
                        self.current_time.strftime('%Y-%m-%d'),
                        self.end_time.strftime('%Y-%m-%d')
                    )
                    self.data_cache[instrument] = df
                    
            except Exception as e:
                logger.error(f"Error loading data for {instrument}: {e}")
                self.data_cache[instrument] = pd.DataFrame()
                
    def get_next_events(self, max_events: int = 100) -> List[MarketEvent]:
        """
        Get next batch of market events.
        
        Args:
            max_events: Maximum number of events to return
            
        Returns:
            List of MarketEvent objects
        """
        if self.current_time >= self.end_time:
            return []
            
        events = []
        events_added = 0
        
        # Determine next time step
        next_time = self._get_next_time_step()
        
        for instrument in self.instruments:
            if instrument not in self.data_cache or self.data_cache[instrument].empty:
                continue
                
            # Get data for this time window
            instrument_events = self._get_events_for_instrument(
                instrument, self.current_time, next_time
            )
            
            events.extend(instrument_events)
            events_added += len(instrument_events)
            
            if events_added >= max_events:
                break
                
        # Sort events by timestamp
        events.sort(key=lambda x: x.timestamp)
        
        # Update current time
        if events:
            self.current_time = events[-1].timestamp
        else:
            self.current_time = next_time
            
        return events[:max_events]
        
    def _get_next_time_step(self) -> datetime:
        """Calculate next time step based on configuration."""
        if self.config['time_step'] == 'tick':
            # For tick data, advance by 1 second
            return self.current_time + timedelta(seconds=1)
        elif self.config['time_step'] == '1s':
            return self.current_time + timedelta(seconds=1)
        elif self.config['time_step'] == '1m':
            return self.current_time + timedelta(minutes=1)
        elif self.config['time_step'] == '5m':
            return self.current_time + timedelta(minutes=5)
        elif self.config['time_step'] == '15m':
            return self.current_time + timedelta(minutes=15)
        elif self.config['time_step'] == '1h':
            return self.current_time + timedelta(hours=1)
        elif self.config['time_step'] == '4h':
            return self.current_time + timedelta(hours=4)
        elif self.config['time_step'] == '1d':
            return self.current_time + timedelta(days=1)
        else:
            # Default to 1 second
            return self.current_time + timedelta(seconds=1)
            
    def _get_events_for_instrument(self, instrument: str, 
                                 start_time: datetime, 
                                 end_time: datetime) -> List[MarketEvent]:
        """Get events for specific instrument in time range."""
        events = []
        
        if instrument not in self.data_cache:
            return events
            
        df = self.data_cache[instrument]
        
        # Filter data for time range
        mask = (df.index >= start_time) & (df.index < end_time)
        period_data = df[mask]
        
        if period_data.empty:
            return events
            
        # Create events based on data type
        if self.config['time_step'] == 'tick':
            # Create tick events
            for timestamp, row in period_data.iterrows():
                event_data = {
                    'bid': row.get('bid', np.nan),
                    'ask': row.get('ask', np.nan),
                    'mid': row.get('mid', np.nan),
                    'spread': row.get('spread', np.nan),
                    'bid_size': row.get('bid_size', np.nan),
                    'ask_size': row.get('ask_size', np.nan)
                }
                
                # Remove NaN values
                event_data = {k: v for k, v in event_data.items() if not pd.isna(v)}
                
                if event_data:  # Only create event if we have valid data
                    event = MarketEvent(
                        timestamp=timestamp,
                        instrument=instrument,
                        event_type='tick',
                        data=event_data
                    )
                    events.append(event)
        else:
            # Create bar events
            for timestamp, row in period_data.iterrows():
                event_data = {
                    'open': row.get('open', np.nan),
                    'high': row.get('high', np.nan),
                    'low': row.get('low', np.nan),
                    'close': row.get('close', np.nan),
                    'volume': row.get('volume', np.nan),
                    'spread': row.get('spread', np.nan)
                }
                
                # Remove NaN values
                event_data = {k: v for k, v in event_data.items() if not pd.isna(v)}
                
                if event_data:  # Only create event if we have valid data
                    event = MarketEvent(
                        timestamp=timestamp,
                        instrument=instrument,
                        event_type='bar',
                        data=event_data
                    )
                    events.append(event)
                    
        return events
        
    def get_current_market_data(self, instrument: str) -> Optional[Dict[str, float]]:
        """
        Get current market data for instrument.
        
        Args:
            instrument: Instrument symbol
            
        Returns:
            Dictionary with current market data or None
        """
        if instrument not in self.data_cache or self.data_cache[instrument].empty:
            return None
            
        df = self.data_cache[instrument]
        
        # Get most recent data before or at current time
        mask = df.index <= self.current_time
        recent_data = df[mask]
        
        if recent_data.empty:
            return None
            
        # Return most recent row as dictionary
        latest_row = recent_data.iloc[-1]
        return latest_row.to_dict()
        
    def is_replay_complete(self) -> bool:
        """Check if replay has reached the end time."""
        return self.current_time >= self.end_time
        
    def get_replay_progress(self) -> Dict[str, Union[float, str]]:
        """Get replay progress information."""
        if not self.end_time or not self.current_time:
            return {'progress': 0.0, 'status': 'Not started'}
            
        total_duration = (self.end_time - self.current_time).total_seconds()
        elapsed = (self.current_time - self.current_time).total_seconds()
        
        if total_duration <= 0:
            return {'progress': 100.0, 'status': 'Complete'}
            
        progress = min(100.0, (elapsed / total_duration) * 100)
        
        return {
            'progress': progress,
            'current_time': self.current_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'status': 'Complete' if self.is_replay_complete() else 'Running'
        }