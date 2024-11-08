import yfinance as yf
from datetime import date, timedelta
from analyze import Analyze

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
        self.current_price = self.hist['Close'].iloc[-1]
        self.option_chain = None
        self.start_dt = None
        self.end_dt = None
        self.price_data = None
        self.analyzer = Analyze()  # Initialize the Analyze class for calculations

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
    
    def calculate_implied_volatility(self, K, T, r, market_price, option_type="call"):
        """
        Uses the Analyze class to calculate the implied volatility for the specified option parameters.
        """
        S = self.current_price  # Use the current price of the stock
        sigma = self.analyzer.implied_volatility(S, K, T, r, market_price, option_type)
        return sigma
    
    def calculate_rho(self, K, T, r, sigma, option_type="call"):
        """
        Calculates the rho of an option and returns the rate used.
        """
        S = self.current_price
        rho_value = self.analyzer.rho(S, K, T, r, sigma, option_type)
        return rho_value, r