import numpy as np

def monte_carlo_price(spot: float, strike: float, time_to_expiry: float, rate: float, sigma: float, sims: int = 10000, option_type: str = "call", seed: int = None):
    """
    Simple Monte Carlo pricing for European options using risk-neutral simulation.
    
    Arguments:
    spot (float): spot price / current market price of the underlying asset
    strike (float): strike price of the option (fixed price to buy or sell)
    time_to_expiry (float): time to expiry in years
    rate (float): risk-free interest rate (annualized)
    sigma (float): volatility of the underlying asset (annualized)
    sims (int): number of Monte Carlo simulations
    option_type (str): "call" or "put"
    seed (int): random seed for reproducibility

    Returns:
    price (float): Monte Carlo price of the option
    """
    # Use seed to generate reproducible random numbers
    rng = np.random.default_rng(seed)

    # Generate random samples from normal distributions
    Z = rng.standard_normal(sims)

    # Generate future prices using Black-Scholes model
    ST = spot * np.exp((rate - 0.5 * sigma ** 2) * time_to_expiry + sigma * np.sqrt(time_to_expiry) * Z)

    # Calculate payoffs based on option type
    if option_type == "call":
        payoffs = np.maximum(ST - strike, 0.0)
    else:
        payoffs = np.maximum(strike - ST, 0.0)

    # Apply discounted cashflow back to current date and return the average price
    price = np.exp(-rate * time_to_expiry) * np.mean(payoffs)
    return price
