from options_engine.pricing.black_scholes import black_scholes_price
from options_engine.pricing.binomial import binomial_price
from options_engine.pricing.monte_carlo import monte_carlo_price
from options_engine.pricing.barrier import barrier_price_analytic, barrier_price_mc

# ==== Function to compare pricing methods in command line ====

def compare_examples():
    spot, strike, time_to_expiry, rate, sigma = 100, 100, 30/365, 0.01, 0.2
    bs_call = black_scholes_price(spot, strike, time_to_expiry, rate, sigma, 'call')
    bin_call = binomial_price(spot, strike, time_to_expiry, rate, sigma, steps=100, option_type='call')
    mc_call = monte_carlo_price(spot, strike, time_to_expiry, rate, sigma, sims=20000, option_type='call', seed=1)
    print('Black-Scholes Call:', bs_call)
    print('Binomial Call:', bin_call)
    print('Monte Carlo Call:', mc_call)


# ==== Function to compare barrier pricing methods in command line ====

def compare_barrier_examples():
    spot, strike, barrier, time_to_expiry, rate, sigma = 90, 100, 130, 1.0, 0.03, 0.25
    steps = 252

    continuous = barrier_price_analytic(spot, strike, barrier, time_to_expiry, rate, sigma)
    matched = barrier_price_analytic(spot, strike, barrier, time_to_expiry, rate, sigma, monitoring_steps=steps)
    mc = barrier_price_mc(spot, strike, barrier, time_to_expiry, rate, sigma, sims=100000, steps=steps, seed=1)

    print('Up-and-Out Call (continuous monitoring, closed form):', continuous)
    print(f'Up-and-Out Call (analytic, matched to {steps}-step monitoring):', matched)
    print(f'Up-and-Out Call (Monte Carlo, {steps} steps):', mc)


if __name__ == '__main__':
    compare_examples()
    print()
    compare_barrier_examples()
