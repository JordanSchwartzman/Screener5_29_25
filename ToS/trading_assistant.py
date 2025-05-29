import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import pytz
import asyncio
import threading
import time
from queue import Queue

# --- Constants ---
API_KEY = "PKUWPLGDIK021YJ89E1U"
API_SECRET = "CDJY2cjFagkYxVhiAeDXgyjA4KcCgGz9vEZef6c7"
BASE_URL = 'https://paper-api.alpaca.markets'

# Load S&P 500 tickers from CSV
try:
    SP500_TICKERS = pd.read_csv('sp500_tickers.csv')['Ticker'].tolist()
    print(f"Loaded {len(SP500_TICKERS)} S&P 500 tickers")
except Exception as e:
    print(f"Error loading S&P 500 tickers: {str(e)}")
    print("Using default ticker list...")
    SP500_TICKERS = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK-B', 'JPM', 'JNJ', 'V',
        # Add more tickers as needed
    ]

# --- Initialize Alpaca API ---
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# --- Initialize FinBERT Model ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

# --- Scanner Class ---
class LiveScanner:
    def __init__(self, queue):
        self.eastern = pytz.timezone('US/Eastern')
        self.ticker_data = {}
        self.alerted_tickers = set()
        self.queue = queue
        self.running = False
        self.last_scan_time = None
        
    def is_market_open(self):
        clock = api.get_clock()
        return clock.is_open
    
    def get_market_time(self):
        return datetime.now(self.eastern)
    
    def should_scan(self):
        current_time = self.get_market_time()
        market_time = current_time.time()
        return (market_time >= datetime.strptime('11:00', '%H:%M').time() and 
                market_time <= datetime.strptime('15:45', '%H:%M').time())
    
    def get_previous_day_data(self, ticker):
        end_date = datetime.now(self.eastern)
        start_date = end_date - timedelta(days=5)
        
        bars = api.get_bars(
            ticker,
            '1Day',
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            adjustment='raw'
        ).df
        
        if len(bars) >= 2:
            return bars.iloc[-2]
        return None
    
    def get_intraday_data(self, ticker):
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
        if ticker in self.alerted_tickers:
            return None
            
        prev_day = self.get_previous_day_data(ticker)
        if prev_day is None:
            return None
            
        today_data = self.get_intraday_data(ticker)
        if len(today_data) < 18:
            return None
            
        prev_day_return = (today_data['open'].iloc[0] - prev_day['close']) / prev_day['close'] * 100
        if prev_day_return >= -1:
            return None
            
        gap_down = (today_data['open'].iloc[0] - prev_day['close']) / prev_day['close'] * 100
        if gap_down >= -0.3:
            return None
            
        first_90m = today_data.iloc[:18]
        high_90m = first_90m['high'].max()
        low_90m = first_90m['low'].min()
        
        current_price = today_data['close'].iloc[-1]
        
        high_retest = abs(current_price - high_90m) / high_90m * 100 <= 0.1
        low_retest = abs(current_price - low_90m) / low_90m * 100 <= 0.1
        
        if high_retest or low_retest:
            self.alerted_tickers.add(ticker)
            return {
                'ticker': ticker,
                'signal_type': 'High Retest' if high_retest else 'Low Retest',
                'time_triggered': datetime.now(self.eastern).strftime('%H:%M:%S'),
                'prev_day_return': prev_day_return,
                'gap_down': gap_down,
                'current_price': current_price
            }
        
        return None
    
    def run(self):
        self.running = True
        while self.running:
            try:
                if not self.is_market_open() or not self.should_scan():
                    time.sleep(60)
                    continue
                
                self.last_scan_time = datetime.now(self.eastern)
                self.queue.put({'type': 'status_update', 'last_scan': self.last_scan_time})
                
                for ticker in SP500_TICKERS:
                    try:
                        result = self.check_strategy_conditions(ticker)
                        if result:
                            self.queue.put({'type': 'signal', 'data': result})
                    except Exception as e:
                        print(f"Error processing {ticker}: {str(e)}")
                        continue
                
                time.sleep(60)
                
            except Exception as e:
                print(f"Error in scanner loop: {str(e)}")
                time.sleep(60)

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Trading Assistant", layout="centered")
    
    # Initialize session state
    if 'scanner_results' not in st.session_state:
        st.session_state.scanner_results = []
    if 'scanner_running' not in st.session_state:
        st.session_state.scanner_running = False
    if 'scanner_queue' not in st.session_state:
        st.session_state.scanner_queue = Queue()
    
    # Title and description
    st.title("🎯 Jordy's Trading Assistant")
    st.write("Combined FinBERT sentiment analysis and real-time S&P 500 screener")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    # --- FinBERT Sentiment Analysis (Left Column) ---
    with col1:
        st.subheader("📰 News Sentiment Analysis")
        
        # Load FinBERT model
        tokenizer, model = load_model()
        
        text_input = st.text_area("Paste financial news here:", height=300)
        
        def analyze_sentiment(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            labels = ["negative", "neutral", "positive"]
            return labels[prediction.item()], confidence.item(), probs[0].tolist()
        
        if st.button("Analyze Sentiment"):
            if text_input.strip() == "":
                st.warning("Please paste a news article or sentence.")
            else:
                sentiment, confidence, scores = analyze_sentiment(text_input)
                
                # Color coding based on sentiment
                color = {
                    "positive": "green",
                    "neutral": "blue",
                    "negative": "red"
                }[sentiment]
                
                st.markdown(f"**Sentiment:** :{color}[{sentiment.title()}]")
                st.markdown(f"**Confidence:** `{confidence:.2f}`")
                
                with st.expander("🔬 Full Score Breakdown"):
                    st.json({
                        "Positive": round(scores[2], 4),
                        "Neutral": round(scores[1], 4),
                        "Negative": round(scores[0], 4)
                    })
    
    # --- Live Scanner (Right Column) ---
    with col2:
        st.subheader("📊 S&P 500 Live Scanner")
        
        # Scanner controls and status
        scanner_col1, scanner_col2 = st.columns(2)
        with scanner_col1:
            if st.button("Start Scanner" if not st.session_state.scanner_running else "Stop Scanner"):
                st.session_state.scanner_running = not st.session_state.scanner_running
                if st.session_state.scanner_running:
                    scanner = LiveScanner(st.session_state.scanner_queue)
                    scanner_thread = threading.Thread(target=scanner.run)
                    scanner_thread.start()
                else:
                    st.session_state.scanner_results = []
                    st.session_state.last_scan_time = None
        
        # Status indicator
        if st.session_state.scanner_running:
            st.markdown("""
                <style>
                .scanner-status {
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                .status-active {
                    background-color: #e6ffe6;
                    border: 1px solid #00cc00;
                    color: black;
                }
                </style>
                <div class="scanner-status status-active">
                    <p style="margin: 0;">🟢 Scanner Active</p>
                </div>
            """, unsafe_allow_html=True)
            
            if 'last_scan_time' in st.session_state and st.session_state.last_scan_time:
                st.caption(f"Last scan: {st.session_state.last_scan_time.strftime('%H:%M:%S')} ET")
        else:
            st.markdown("""
                <style>
                .scanner-status {
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                .status-inactive {
                    background-color: #ffe6e6;
                    border: 1px solid #cc0000;
                    color: black;
                }
                </style>
                <div class="scanner-status status-inactive">
                    <p style="margin: 0;">🔴 Scanner Inactive</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Display scanner results
        if st.session_state.scanner_results:
            df = pd.DataFrame([r['data'] for r in st.session_state.scanner_results if r['type'] == 'signal'])
            st.dataframe(
                df,
                column_config={
                    "ticker": "Ticker",
                    "signal_type": "Signal Type",
                    "time_triggered": "Time Triggered",
                    "prev_day_return": st.column_config.NumberColumn(
                        "Prev Day Return",
                        format="%.2f%%"
                    ),
                    "gap_down": st.column_config.NumberColumn(
                        "Gap Down",
                        format="%.2f%%"
                    ),
                    "current_price": st.column_config.NumberColumn(
                        "Current Price",
                        format="$%.2f"
                    )
                },
                hide_index=True
            )
        else:
            st.info("No scanner results yet. Start the scanner to begin monitoring.")
    
    # Update scanner results and status
    while not st.session_state.scanner_queue.empty():
        result = st.session_state.scanner_queue.get()
        if result['type'] == 'status_update':
            st.session_state.last_scan_time = result['last_scan']
        elif result['type'] == 'signal':
            st.session_state.scanner_results.append(result)
        st.experimental_rerun()

if __name__ == "__main__":
    main() 