import math
from stock_search import StockSearch

# Initialize StockSearch with the ticker symbol
stock = StockSearch(ticker_symb="AAPL")

# Retrieve option chain for a specific expiration date
stock.get_option_chain('2024-12-20')

# Parameters for rho calculation
K = 230           # Strike price
T = 0.821         # Time to expiry in years
r = 0.0469       # Risk-free interest rate
market_price = 4.39	
sigma = 0.99      # Assumed volatility (can be an estimated or observed value)

# Calculate rho and return the rate used
# rho_value, rate_used = stock.calculate_rho(K, T, r, sigma, option_type="call")
# Calculate implied volatility
sigma = stock.calculate_implied_volatility(K, T, r, market_price, option_type="call")

print("Implied Volatility (sigma):", sigma)