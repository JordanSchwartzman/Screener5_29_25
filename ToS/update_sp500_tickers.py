import pandas as pd
import os

def clean_ticker(ticker):
    """Clean ticker symbol by replacing dots with hyphens"""
    return ticker.replace('.', '-')

def download_sp500_tickers():
    # URL for S&P 500 companies list
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    try:
        # Read the table from Wikipedia
        print("Downloading S&P 500 tickers from Wikipedia...")
        tables = pd.read_html(url)
        
        # The first table contains the current S&P 500 companies
        df = tables[0]
        
        # Extract and clean ticker symbols
        tickers = df['Symbol'].apply(clean_ticker).tolist()
        
        # Save to text file
        with open('sp500_tickers.txt', 'w') as f:
            f.write('\n'.join(tickers))
        
        # Save to CSV
        df_tickers = pd.DataFrame({'Ticker': tickers})
        df_tickers.to_csv('sp500_tickers.csv', index=False)
        
        print(f"\nSaved {len(tickers)} tickers to sp500_tickers.txt and sp500_tickers.csv")
        print("\nFirst 5 tickers:", tickers[:5])
        print("Last 5 tickers:", tickers[-5:])
        
    except Exception as e:
        print(f"Error downloading tickers: {str(e)}")

if __name__ == "__main__":
    download_sp500_tickers() 