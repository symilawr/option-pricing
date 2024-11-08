from stock_search import StockSearch
from analyze import Analyze

# Initialize StockSearch with the ticker symbol
stock = StockSearch(ticker_symb="AAPL")
analyze = Analyze()

# Retrieve historical prices and calculate standard deviation
stock.ticker.info
print(stock.ticker.info)
stock.get_option_chain('2024-12-20')
print(stock.option_chain)

# data = stock.get_historical_prices()
# std_dev = analyze.calc_std_dev(data)

print("done")
