import yfinance as yf
from datetime import date, timedelta

class StockSearch:
    def __init__(self, ticker) -> None:
        self.ticker = ticker  # Store the ticker symbol
        self.start_dt = None
        self.end_dt = None
        self.price_data = None

    def get_historical_prices(self, start_dt=None, end_dt=None):
        # Set default dates if not provided
        if start_dt is None:
            start_dt = date.today() - timedelta(days=365)
        if end_dt is None:
            end_dt = date.today()

        # Download historical data
        self.price_data = yf.download(self.ticker, start=start_dt, end=end_dt)

        return self.price_data
