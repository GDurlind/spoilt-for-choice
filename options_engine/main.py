from options_engine.pricing.black_scholes import black_scholes_price
from options_engine.pricing.binomial import binomial_price
from options_engine.pricing.monte_carlo import monte_carlo_price

# ==== Function to compare pricing methods in command line ====

def compare_examples():
    spot, strike, time_to_expiry, rate, sigma = 100, 100, 30/365, 0.01, 0.2
    bs_call = black_scholes_price(spot, strike, time_to_expiry, rate, sigma, 'call')
    bin_call = binomial_price(spot, strike, time_to_expiry, rate, sigma, steps=100, option_type='call')
    mc_call = monte_carlo_price(spot, strike, time_to_expiry, rate, sigma, sims=20000, option_type='call', seed=1)
    print('Black-Scholes Call:', bs_call)
    print('Binomial Call:', bin_call)
    print('Monte Carlo Call:', mc_call)

if __name__ == '__main__':
    compare_examples()
