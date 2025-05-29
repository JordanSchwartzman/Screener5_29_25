import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pytz

# Alpaca API configuration
API_KEY = "PKUWPLGDIK021YJ89E1U"
API_SECRET = "CDJY2cjFagkYxVhiAeDXgyjA4KcCgGz9vEZef6c7"
BASE_URL = 'https://paper-api.alpaca.markets'  # Use paper trading URL

# Validate API credentials
if not API_KEY or not API_SECRET:
    raise ValueError("""
    Alpaca API credentials not found! Please:
    1. Create a .env file in the same directory as this script
    2. Add your API credentials in this format:
       ALPACA_API_KEY=your_api_key_here
       ALPACA_API_SECRET=your_api_secret_here
    3. Get your API credentials from: https://app.alpaca.markets/paper/dashboard/overview
    """)

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# S&P 500 tickers (you can load this from a file or API)
SP500_TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK.B', 'JPM', 'JNJ', 'V',
    # Add more tickers as needed
]

class LiveScanner:
    def __init__(self):
        self.eastern = pytz.timezone('US/Eastern')
        self.ticker_data = {}  # Store data for each ticker
        self.alerted_tickers = set()  # Track tickers that have already triggered alerts
        
    def is_market_open(self):
        """Check if the market is currently open"""
        clock = api.get_clock()
        return clock.is_open
    
    def get_market_time(self):
        """Get current market time in Eastern timezone"""
        return datetime.now(self.eastern)
    
    def should_scan(self):
        """Check if we should be scanning based on market hours"""
        current_time = self.get_market_time()
        market_time = current_time.time()
        
        # Only scan between 11:00 AM and 3:45 PM ET
        return (market_time >= datetime.strptime('11:00', '%H:%M').time() and 
                market_time <= datetime.strptime('15:45', '%H:%M').time())
    
    def get_previous_day_data(self, ticker):
        """Get previous day's data for a ticker"""
        end_date = datetime.now(self.eastern)
        start_date = end_date - timedelta(days=5)  # Get 5 days of data to ensure we have previous day
        
        bars = api.get_bars(
            ticker,
            '1Day',
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            adjustment='raw'
        ).df
        
        if len(bars) >= 2:
            return bars.iloc[-2]  # Return previous day's data
        return None
    
    def get_intraday_data(self, ticker):
        """Get today's intraday data for a ticker"""
        end_date = datetime.now(self.eastern)
        start_date = end_date.replace(hour=9, minute=30, second=0, microsecond=0)
        
        bars = api.get_bars(
            ticker,
            '5Min',
            start=start_date.strftime('%Y-%m-%d %H:%M:%S'),
            end=end_date.strftime('%Y-%m-%d %H:%M:%S'),
            adjustment='raw'
        ).df
        
        return bars
    
    def check_strategy_conditions(self, ticker):
        """Check if all strategy conditions are met for a ticker"""
        if ticker in self.alerted_tickers:
            return False
            
        # Get previous day's data
        prev_day = self.get_previous_day_data(ticker)
        if prev_day is None:
            return False
            
        # Get today's data
        today_data = self.get_intraday_data(ticker)
        if len(today_data) < 18:  # Need at least 18 5-min bars for 90 minutes
            return False
            
        # Calculate previous day return
        prev_day_return = (today_data['open'].iloc[0] - prev_day['close']) / prev_day['close'] * 100
        
        # Check if previous day was strong down (< -1%)
        if prev_day_return >= -1:
            return False
            
        # Calculate gap down
        gap_down = (today_data['open'].iloc[0] - prev_day['close']) / prev_day['close'] * 100
        if gap_down >= -0.3:  # Must be at least 0.3% gap down
            return False
            
        # Calculate 90-minute high/low
        first_90m = today_data.iloc[:18]
        high_90m = first_90m['high'].max()
        low_90m = first_90m['low'].min()
        
        # Get current price
        current_price = today_data['close'].iloc[-1]
        
        # Check for retest (within 0.1% of high/low)
        high_retest = abs(current_price - high_90m) / high_90m * 100 <= 0.1
        low_retest = abs(current_price - low_90m) / low_90m * 100 <= 0.1
        
        if high_retest or low_retest:
            self.alerted_tickers.add(ticker)
            return True
            
        return False
    
    def run(self):
        """Main scanner loop"""
        print("Starting Live Scanner...")
        
        while True:
            try:
                current_time = self.get_market_time()
                
                # Check if market is open and we should be scanning
                if not self.is_market_open() or not self.should_scan():
                    if current_time.time() >= datetime.strptime('15:45', '%H:%M').time():
                        print("Market closing soon. Stopping scanner...")
                        break
                    time.sleep(60)
                    continue
                
                print(f"\nScanning at {current_time.strftime('%Y-%m-%d %H:%M:%S')} ET")
                
                # Scan each ticker
                for ticker in SP500_TICKERS:
                    try:
                        if self.check_strategy_conditions(ticker):
                            print(f"\n✅ [{ticker}] hit 90-min {'high' if high_retest else 'low'} retest after gap down + strong prev down day.")
                            print(f"Previous Day Return: {prev_day_return:.2f}%")
                            print(f"Gap Down: {gap_down:.2f}%")
                            print(f"Current Price: ${current_price:.2f}")
                    except Exception as e:
                        print(f"Error processing {ticker}: {str(e)}")
                        continue
                
                # Wait for 1 minute before next scan
                time.sleep(60)
                
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                time.sleep(60)

if __name__ == "__main__":
    scanner = LiveScanner()
    scanner.run()
