from StockSearch import StockSearch
from analyze import Analyze

# Initialize StockSearch with the ticker symbol
stock = StockSearch(ticker="AAPL")
analyze = Analyze()

# Retrieve historical prices and calculate standard deviation
data = stock.get_historical_prices()
std_dev = analyze.calc_std_dev(data)

print(std_dev)
print("done")
