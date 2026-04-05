import yfinance as yf
import pandas as pd

class DataNexus:
    """
    The DataNexus is the foundational data layer of our engine. 
    It ensures all other modules get clean, synchronized data.
    """
    def __init__(self, tickers):
        self.tickers = tickers
        self.prices = None
        self.fundamentals = {}

    def fetch_everything(self, period="2y"):
        """Downloads price history and fundamental metrics in one go."""
        print(f"📡 Fetching market data for {self.tickers}...")
        
        # 1. Fetch Prices
        self.prices = yf.download(self.tickers, period=period)['Close'].ffill().dropna()
        
        # 2. Fetch Fundamentals for Valuation
        for t in self.tickers:
            ticker_obj = yf.Ticker(t)
            info = ticker_obj.info
            self.fundamentals[t] = {
                'name': info.get('longName'),
                'pe_ratio': info.get('forwardPE'),
                'market_cap': info.get('marketCap'),
                'dividend_yield': info.get('dividendYield'),
                'sector': info.get('sector')
            }
        return self.prices, self.fundamentals

if __name__ == "__main__":
    # Test Run
    nexus = DataNexus(['AAPL', 'SAP.DE'])
    p, f = nexus.fetch_everything()
    print("\n✅ Nexus ready. Fundamental for AAPL:", f['AAPL']['pe_ratio'])