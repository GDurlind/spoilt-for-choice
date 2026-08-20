import pytest

from options_engine.pricing.barrier import barrier_price_analytic, barrier_price_mc
from options_engine.pricing.black_scholes import black_scholes_price

SPOT, STRIKE, BARRIER, T, RATE, SIGMA = 90.0, 100.0, 130.0, 1.0, 0.03, 0.25


# ---- barrier_price_analytic ----

def test_analytic_matches_vanilla_as_barrier_goes_to_infinity():
    far = barrier_price_analytic(SPOT, STRIKE, 1e7, T, RATE, SIGMA)
    vanilla = black_scholes_price(SPOT, STRIKE, T, RATE, SIGMA, "call")
    assert far == pytest.approx(vanilla, abs=1e-6)


@pytest.mark.parametrize("barrier", [STRIKE, STRIKE - 5])
def test_analytic_worthless_when_barrier_at_or_below_strike(barrier):
    assert barrier_price_analytic(SPOT, STRIKE, barrier, T, RATE, SIGMA) == 0.0


def test_analytic_worthless_when_spot_already_at_or_past_barrier():
    assert barrier_price_analytic(BARRIER, STRIKE, BARRIER, T, RATE, SIGMA) == 0.0
    assert barrier_price_analytic(BARRIER + 5, STRIKE, BARRIER, T, RATE, SIGMA) == 0.0


def test_analytic_returns_intrinsic_at_expiry():
    itm_spot = 105.0
    price = barrier_price_analytic(itm_spot, STRIKE, BARRIER, 0.0, RATE, SIGMA)
    assert price == pytest.approx(itm_spot - STRIKE)


def test_analytic_priced_below_vanilla_but_above_zero():
    price = barrier_price_analytic(SPOT, STRIKE, BARRIER, T, RATE, SIGMA)
    vanilla = black_scholes_price(SPOT, STRIKE, T, RATE, SIGMA, "call")
    assert 0.0 < price < vanilla


# ---- barrier_price_mc ----

def test_mc_worthless_when_spot_already_at_or_past_barrier():
    assert barrier_price_mc(BARRIER, STRIKE, BARRIER, T, RATE, SIGMA, sims=100, seed=1) == 0.0


def test_mc_returns_intrinsic_at_expiry():
    itm_spot = 105.0
    price = barrier_price_mc(itm_spot, STRIKE, BARRIER, 0.0, RATE, SIGMA, option_type="call", sims=100, seed=1)
    assert price == pytest.approx(itm_spot - STRIKE)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mc_matches_vanilla_when_barrier_never_binds(option_type):
    # A barrier placed far out of reach should never trigger, so the barrier
    # price should reduce to the plain Black-Scholes price -- this exercises
    # the put payoff branch too, which the analytic oracle doesn't cover.
    huge_barrier = 1e6
    mc = barrier_price_mc(SPOT, STRIKE, huge_barrier, T, RATE, SIGMA, option_type=option_type, sims=100_000, steps=50, seed=42)
    vanilla = black_scholes_price(SPOT, STRIKE, T, RATE, SIGMA, option_type)
    assert mc == pytest.approx(vanilla, abs=0.25)


def test_mc_converges_to_analytic_at_matched_monitoring_frequency():
    steps = 200
    mc = barrier_price_mc(SPOT, STRIKE, BARRIER, T, RATE, SIGMA, sims=150_000, steps=steps, seed=42)
    matched = barrier_price_analytic(SPOT, STRIKE, BARRIER, T, RATE, SIGMA, monitoring_steps=steps)
    assert mc == pytest.approx(matched, abs=0.04)


def test_discrete_monitoring_correction_is_actually_needed():
    # Without the BGK correction, matching discrete MC to the continuous formula
    # is measurably worse than matching it to the corrected one -- this is the
    # step 6 bias, and the test guards against silently dropping the correction.
    steps = 200
    mc = barrier_price_mc(SPOT, STRIKE, BARRIER, T, RATE, SIGMA, sims=150_000, steps=steps, seed=42)
    continuous = barrier_price_analytic(SPOT, STRIKE, BARRIER, T, RATE, SIGMA)
    matched = barrier_price_analytic(SPOT, STRIKE, BARRIER, T, RATE, SIGMA, monitoring_steps=steps)
    assert abs(mc - matched) < abs(mc - continuous)
