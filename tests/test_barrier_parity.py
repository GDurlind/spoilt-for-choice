import math

import numpy as np
import pytest

from options_engine.pricing.barrier import (
    barrier_price_analytic,
    barrier_price_mc,
    barrier_price_up_and_in_analytic,
    barrier_price_up_and_in_mc,
    simulate_gbm_paths,
)
from options_engine.pricing.black_scholes import black_scholes_price

SPOT, STRIKE, BARRIER, T, RATE, SIGMA = 90.0, 100.0, 130.0, 1.0, 0.03, 0.25


def test_analytic_in_out_parity_is_exact():
    knock_out = barrier_price_analytic(SPOT, STRIKE, BARRIER, T, RATE, SIGMA)
    knock_in = barrier_price_up_and_in_analytic(SPOT, STRIKE, BARRIER, T, RATE, SIGMA)
    vanilla = black_scholes_price(SPOT, STRIKE, T, RATE, SIGMA, "call")
    assert knock_out + knock_in == pytest.approx(vanilla, abs=1e-9)


def test_mc_in_out_parity_holds_exactly_for_a_shared_seed():
    # In and out payoffs are complementary on every single path when both
    # pricers are given the same seed, so the sum should match a directly
    # computed vanilla price on those same paths to floating-point precision
    # -- this is an algebraic identity, not a statistical convergence claim,
    # so the tolerance here is tight (not scaled to Monte Carlo noise).
    seed, sims, steps = 123, 20_000, 50

    knock_out = barrier_price_mc(SPOT, STRIKE, BARRIER, T, RATE, SIGMA, sims=sims, steps=steps, seed=seed)
    knock_in = barrier_price_up_and_in_mc(SPOT, STRIKE, BARRIER, T, RATE, SIGMA, sims=sims, steps=steps, seed=seed)

    paths = simulate_gbm_paths(SPOT, T, RATE, SIGMA, sims=sims, steps=steps, seed=seed)
    vanilla_payoffs = np.maximum(paths[:, -1] - STRIKE, 0.0)
    vanilla_mc = math.exp(-RATE * T) * np.mean(vanilla_payoffs)

    assert knock_out + knock_in == pytest.approx(vanilla_mc, abs=1e-9)


def test_mc_in_out_parity_breaks_without_a_shared_seed():
    # Sanity check on the claim above: two independent (differently-seeded)
    # simulations should NOT sum exactly to vanilla -- if they did, the
    # "shared seed" explanation for the exact test above would be wrong.
    sims, steps = 20_000, 50
    knock_out = barrier_price_mc(SPOT, STRIKE, BARRIER, T, RATE, SIGMA, sims=sims, steps=steps, seed=1)
    knock_in = barrier_price_up_and_in_mc(SPOT, STRIKE, BARRIER, T, RATE, SIGMA, sims=sims, steps=steps, seed=2)
    vanilla = black_scholes_price(SPOT, STRIKE, T, RATE, SIGMA, "call")
    assert abs((knock_out + knock_in) - vanilla) > 1e-6


def test_up_and_in_worthless_when_barrier_unreachable():
    price = barrier_price_up_and_in_analytic(SPOT, STRIKE, 1e7, T, RATE, SIGMA)
    assert price == pytest.approx(0.0, abs=1e-6)


def test_up_and_in_equals_vanilla_when_already_breached():
    spot_through = BARRIER + 5.0
    knock_in = barrier_price_up_and_in_analytic(spot_through, STRIKE, BARRIER, T, RATE, SIGMA)
    vanilla = black_scholes_price(spot_through, STRIKE, T, RATE, SIGMA, "call")
    assert knock_in == pytest.approx(vanilla, abs=1e-9)


def test_up_and_in_equals_vanilla_when_barrier_at_or_below_strike():
    # Any path finishing in the money must have already crossed a barrier
    # placed at or below the strike, so the "in" condition is never the
    # binding constraint -- the knock-in collapses to the vanilla price.
    knock_in = barrier_price_up_and_in_analytic(SPOT, STRIKE, STRIKE, T, RATE, SIGMA)
    vanilla = black_scholes_price(SPOT, STRIKE, T, RATE, SIGMA, "call")
    assert knock_in == pytest.approx(vanilla, abs=1e-9)
