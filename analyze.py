import pandas as pd
import math
from scipy.stats import norm

class Analyze:

    def __init__(self) -> None:
        pass

    def calc_std_dev(self, data):
        data = pd.DataFrame(data=data)
        standard_deviation = data.iloc[:, 2].std()
        return standard_deviation

    def black_scholes_price(self, S, K, T, r, sigma, option_type="call"):
        """
        Calculates the Black-Scholes price for a call or put option.
        """
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        if option_type == "call":
            price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be either 'call' or 'put'")
        
        return price

    def vega(self, S, K, T, r, sigma):
        """
        Calculates the vega of an option, which is the derivative of the price with respect to volatility.
        """
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return S * norm.pdf(d1) * math.sqrt(T)

    def rho(self, S, K, T, r, sigma, option_type="call"):
        """
        Calculates the rho of an option, which is the sensitivity of the price with respect to interest rates.
        """
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        if option_type == "call":
            rho = K * T * math.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)
        else:
            raise ValueError("option_type must be either 'call' or 'put'")
        
        return rho
    
    
    def implied_volatility(self, S, K, T, r, market_price, option_type="call", tol=1e-5, max_iterations=1000):
        """
        Calculates the implied volatility using the Newton-Raphson method.
        
        Parameters:
        - S: Stock price
        - K: Strike price
        - T: Time to expiration (in years)
        - r: Risk-free interest rate
        - market_price: Observed market price of the option
        - option_type: 'call' or 'put'
        - tol: Tolerance for convergence
        - max_iterations: Maximum number of iterations
        
        Returns:
        - Implied volatility as a float
        """
        # Initial guess for implied volatility
        sigma = .99 # Starting guess (can be adjusted)
        
        for i in range(max_iterations):
            # Calculate the option price and vega with the current sigma
            price = self.black_scholes_price(S, K, T, r, sigma, option_type)
            v = self.vega(S, K, T, r, sigma)
            
            # Calculate the price difference
            price_diff = price - market_price
            
            # Check if the difference is within tolerance
            if abs(price_diff) < tol:
                return sigma
            
            # Update sigma using Newton-Raphson method
            sigma -= price_diff / v

        # If we exceed the maximum iterations without converging
        raise ValueError("Implied volatility did not converge")
