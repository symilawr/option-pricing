import yfinance as yf
from datetime import date, timedelta

class StockSearch:
    def __init__(self, ticker_symb) -> None:
        self.ticker_symb = ticker_symb  # Store the ticker symbol
        self.ticker = yf.Ticker(self.ticker_symb)
        self.ticker_info = self.ticker.info
        self.hist = self.ticker.history(period="1y")
        self.earnings_estimate = self.ticker.earnings_estimate
        self.earnings_history = self.ticker.earnings_history
        self.earning_dates = self.ticker.earnings_dates
        self.option_expry_dates = self.ticker.options
        self.option_chain = None
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
    
    def get_option_chain(self, expry_date):
        opt = self.ticker.option_chain(expry_date)
        self.option_chain = opt
        return self.option_chain
