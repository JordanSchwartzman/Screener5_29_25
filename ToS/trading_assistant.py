import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import time
import os

# Load S&P 500 tickers from CSV
DEBUG = True  # Debug flag for detailed logging

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

# --- Initialize FinBERT Model ---
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        # Set environment variable to avoid the torch.classes error
        os.environ['PYTORCH_JIT'] = '0'
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
        # Set model to evaluation mode
        model.eval()
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading FinBERT model: {str(e)}")
        return None, None

def analyze_sentiment(text: str, tokenizer, model) -> tuple:
    """Analyze sentiment of text using FinBERT model"""
    if tokenizer is None or model is None:
        return "error", 0.0, [0.0, 0.0, 0.0]
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
        labels = ["negative", "neutral", "positive"]
        return labels[prediction.item()], confidence.item(), probs[0].tolist()
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        return "error", 0.0, [0.0, 0.0, 0.0]

def run_screener():
    """Run a one-time scan across S&P 500 tickers looking for gap downs after down days"""
    results = []
    eastern = pytz.timezone('US/Eastern')
    end_date = datetime.now(eastern)
    start_date = end_date - timedelta(days=10)  # Increased to 10 days to ensure we have enough data
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(SP500_TICKERS):
        try:
            # Update progress
            progress = (i + 1) / len(SP500_TICKERS)
            progress_bar.progress(progress)
            status_text.text(f"Processing {ticker} ({i + 1}/{len(SP500_TICKERS)})")
            
            # Get daily data using yfinance with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    df = yf.download(
                        ticker,
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        progress=False
                    )
                    if not df.empty and len(df) >= 3:
                        break
                    time.sleep(1)  # Wait before retry
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(1)
                    continue
            
            # Check for empty DataFrame
            if df.empty:
                if DEBUG:
                    print(f"\n{ticker}: Empty DataFrame returned")
                continue
                
            # Flatten MultiIndex columns if they exist
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if DEBUG:
                print(f"\n{'='*50}")
                print(f"Processing {ticker}")
                print(f"DataFrame columns: {df.columns.tolist()}")
                print(f"Number of rows: {len(df)}")
                print("\nLast 5 rows of data:")
                print(df.tail())
            
            if len(df) >= 3:  # Need at least 3 days of data
                # Get the last 3 days of data
                day_before_close = df['Close'].iloc[-3]
                yesterday_close = df['Close'].iloc[-2]
                today_open = df['Open'].iloc[-1]
                
                # Check for NaN values
                if pd.isna(day_before_close) or pd.isna(yesterday_close) or pd.isna(today_open):
                    if DEBUG:
                        print(f"\n{ticker}: NaN values detected")
                        print(f"Day-before close: {day_before_close}")
                        print(f"Yesterday close: {yesterday_close}")
                        print(f"Today open: {today_open}")
                    continue
                
                # Calculate yesterday's return
                yesterday_return = (yesterday_close - day_before_close) / day_before_close * 100
                
                # Calculate gap down
                gap_down = (today_open - yesterday_close) / yesterday_close * 100
                
                if DEBUG:
                    print("\nCalculated values:")
                    print(f"Day-before close: ${day_before_close:.2f}")
                    print(f"Yesterday close: ${yesterday_close:.2f}")
                    print(f"Today open: ${today_open:.2f}")
                    print(f"Yesterday Return %: {yesterday_return:.2f}%")
                    print(f"Gap %: {gap_down:.2f}%")
                    print(f"Meets conditions: {yesterday_return <= -1 and gap_down <= -0.3}")
                    print(f"{'='*50}\n")
                
                # Store all results for analysis
                results.append({
                    'Ticker': ticker,
                    'Day-before close': day_before_close,
                    'Yesterday close': yesterday_close,
                    'Today open': today_open,
                    'Yesterday Return %': yesterday_return,
                    'Gap %': gap_down,
                    'Meets Conditions': yesterday_return <= -1 and gap_down <= -0.3
                })
            else:
                if DEBUG:
                    print(f"\n{ticker}: Insufficient data (need at least 3 days, got {len(df)})")
            
            # Small delay to avoid rate limiting
            time.sleep(0.2)  # Increased delay slightly
            
        except Exception as e:
            if DEBUG:
                print(f"\nError processing {ticker}: {str(e)}")
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    if DEBUG:
        if results_df.empty:
            print("\nNo results found for any tickers")
        else:
            print("\nSummary of all results:")
            print(results_df[['Ticker', 'Yesterday Return %', 'Gap %', 'Meets Conditions']].sort_values('Gap %'))
    
    # Return empty DataFrame if no results
    if results_df.empty:
        return pd.DataFrame(columns=['Ticker', 'Day-before close', 'Yesterday close', 'Today open', 
                                   'Yesterday Return %', 'Gap %'])
    
    # Filter results based on conditions
    filtered_results = results_df[results_df['Meets Conditions']].drop('Meets Conditions', axis=1)
    
    return filtered_results

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Trading Assistant", layout="centered")
    
    # Title and description
    st.title("🎯 Trading Assistant")
    st.write("FinBERT sentiment analysis and S&P 500 gap down screener")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    # --- FinBERT Sentiment Analysis (Left Column) ---
    with col1:
        st.subheader("📰 News Sentiment Analysis")
        
        # Load FinBERT model
        tokenizer, model = load_model()
        
        text_input = st.text_area("Paste financial news here:", height=300)
        
        if st.button("Analyze Sentiment"):
            if text_input.strip() == "":
                st.warning("Please paste a news article or sentence.")
            else:
                sentiment, confidence, scores = analyze_sentiment(text_input, tokenizer, model)
                
                if sentiment != "error":
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
    
    # --- Gap Down Screener (Right Column) ---
    with col2:
        st.subheader("📊 S&P 500 Gap Down Screener")
        
        # Add debug mode toggle
        st.session_state.debug = st.checkbox("Show debug information", value=False)
        
        if st.button("Run Screener"):
            with st.spinner("Scanning S&P 500 tickers..."):
                results_df = run_screener()
                
                # Store the scan time
                st.session_state.last_scan_time = datetime.now(pytz.timezone('US/Eastern'))
                
                if len(results_df) > 0:
                    st.dataframe(
                        results_df,
                        column_config={
                            "Ticker": "Ticker",
                            "Day-before close": st.column_config.NumberColumn(
                                "Day-before close",
                                format="$%.2f"
                            ),
                            "Yesterday close": st.column_config.NumberColumn(
                                "Yesterday close",
                                format="$%.2f"
                            ),
                            "Today open": st.column_config.NumberColumn(
                                "Today open",
                                format="$%.2f"
                            ),
                            "Yesterday Return %": st.column_config.NumberColumn(
                                "Yesterday Return %",
                                format="%.2f%%"
                            ),
                            "Gap %": st.column_config.NumberColumn(
                                "Gap %",
                                format="%.2f%%"
                            )
                        },
                        hide_index=True
                    )
                    
                    # Show last scan time
                    if 'last_scan_time' in st.session_state:
                        st.caption(f"Last scan: {st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')} ET")
                else:
                    st.info("No tickers found matching the criteria.")

    # --- Risk-Reward Calculator ---
    st.markdown("---")
    st.subheader("🎯 Risk-Reward Calculator")
    
    # Create two columns for the calculator
    calc_col1, calc_col2 = st.columns(2)
    
    with calc_col1:
        entry_price = st.number_input("Entry Price ($)", min_value=0.0, step=0.01)
        risk_reward_ratio = st.number_input("Risk-Reward Ratio", min_value=0.1, step=0.1, value=1.0)
        
        # Calculate based on stop loss
        stop_loss = st.number_input("Stop Loss Price ($)", min_value=0.0, step=0.01)
        if stop_loss > 0 and entry_price > 0:
            risk = abs(entry_price - stop_loss)
            reward = risk * risk_reward_ratio
            take_profit = entry_price + reward if entry_price > stop_loss else entry_price - reward
            
            st.markdown("### Results (based on Stop Loss)")
            st.markdown(f"**Risk per share:** ${risk:.2f}")
            st.markdown(f"**Reward per share:** ${reward:.2f}")
            st.markdown(f"**Take Profit Price:** ${take_profit:.2f}")
    
    with calc_col2:
        # Calculate based on take profit
        take_profit_input = st.number_input("Take Profit Price ($)", min_value=0.0, step=0.01)
        if take_profit_input > 0 and entry_price > 0:
            reward = abs(take_profit_input - entry_price)
            risk = reward / risk_reward_ratio
            stop_loss_calc = entry_price - risk if take_profit_input > entry_price else entry_price + risk
            
            st.markdown("### Results (based on Take Profit)")
            st.markdown(f"**Risk per share:** ${risk:.2f}")
            st.markdown(f"**Reward per share:** ${reward:.2f}")
            st.markdown(f"**Stop Loss Price:** ${stop_loss_calc:.2f}")

if __name__ == "__main__":
    main() 
