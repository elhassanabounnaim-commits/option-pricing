"""Tests for the European option pricers.

The put-call parity tests are the most valuable ones: the identity follows from
no-arbitrage alone, independently of any model, so it catches errors that a
comparison between our own implementations would miss.
"""

import numpy as np
import pytest

from pricing.binomial import binomial_european
from pricing.black_scholes import black_scholes
from pricing.monte_carlo import monte_carlo_european


# (S0, K, T, r, sigma): at the money, in the money, out of the money,
# short maturity, high volatility, zero rate.
PARAMS = [
    (100.0, 100.0, 1.0, 0.05, 0.20),
    (100.0, 80.0, 1.0, 0.05, 0.20),
    (100.0, 130.0, 1.0, 0.05, 0.20),
    (100.0, 100.0, 0.08, 0.05, 0.20),
    (100.0, 100.0, 2.0, 0.03, 0.60),
    (50.0, 55.0, 1.5, 0.0, 0.25),
]


def parity_gap(S0, K, T, r):
    """Right-hand side of C - P = S0 - K exp(-rT)."""
    return S0 - K * np.exp(-r * T)


@pytest.mark.parametrize("S0,K,T,r,sigma", PARAMS)
def test_put_call_parity_black_scholes(S0, K, T, r, sigma):
    call = black_scholes(S0, K, T, r, sigma, "call")
    put = black_scholes(S0, K, T, r, sigma, "put")
    assert call - put == pytest.approx(parity_gap(S0, K, T, r), abs=1e-10)


@pytest.mark.parametrize("S0,K,T,r,sigma", PARAMS)
def test_put_call_parity_binomial(S0, K, T, r, sigma):
    call = binomial_european(S0, K, T, r, sigma, 200, "call")
    put = binomial_european(S0, K, T, r, sigma, 200, "put")
    assert call - put == pytest.approx(parity_gap(S0, K, T, r), abs=1e-8)


@pytest.mark.parametrize("S0,K,T,r,sigma", PARAMS)
def test_put_call_parity_monte_carlo(S0, K, T, r, sigma):
    """Parity holds pathwise, so the same seed makes the gap purely statistical."""
    call = monte_carlo_european(S0, K, T, r, sigma, 200_000, "call", seed=1)
    put = monte_carlo_european(S0, K, T, r, sigma, 200_000, "put", seed=1)
    tolerance = 5 * (call.std_error + put.std_error)
    assert call.price - put.price == pytest.approx(parity_gap(S0, K, T, r), abs=tolerance)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_binomial_converges_to_black_scholes(option_type):
    """The error must decrease roughly like 1/n."""
    exact = black_scholes(100, 100, 1, 0.05, 0.2, option_type)
    errors = [
        abs(binomial_european(100, 100, 1, 0.05, 0.2, n, option_type) - exact)
        for n in (50, 200, 800)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.01


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_monte_carlo_confidence_interval_contains_exact_price(option_type):
    exact = black_scholes(100, 100, 1, 0.05, 0.2, option_type)
    result = monte_carlo_european(100, 100, 1, 0.05, 0.2, 200_000, option_type, seed=7)
    low, high = result.ci
    assert low <= exact <= high


def test_monte_carlo_is_reproducible():
    kwargs = dict(S0=100, K=100, T=1, r=0.05, sigma=0.2, n_paths=50_000)
    assert monte_carlo_european(**kwargs, seed=42) == monte_carlo_european(**kwargs, seed=42)
    assert monte_carlo_european(**kwargs, seed=42) != monte_carlo_european(**kwargs, seed=43)


@pytest.mark.parametrize("antithetic,control_variate", [(True, False), (False, True)])
def test_variance_reduction_lowers_standard_error(antithetic, control_variate):
    kwargs = dict(S0=100, K=100, T=1, r=0.05, sigma=0.2, n_paths=100_000, seed=42)
    plain = monte_carlo_european(**kwargs)
    reduced = monte_carlo_european(**kwargs, antithetic=antithetic,
                                   control_variate=control_variate)
    assert reduced.std_error < plain.std_error


def test_zero_maturity_returns_intrinsic_value():
    assert black_scholes(120, 100, 0, 0.05, 0.2, "call") == pytest.approx(20.0)
    assert black_scholes(80, 100, 0, 0.05, 0.2, "call") == pytest.approx(0.0)
    assert black_scholes(80, 100, 0, 0.05, 0.2, "put") == pytest.approx(20.0)


@pytest.mark.parametrize("pricer,extra", [
    (black_scholes, {}),
    (binomial_european, {"n": 50}),
    (monte_carlo_european, {"n_paths": 1000}),
])
def test_invalid_option_type_raises(pricer, extra):
    with pytest.raises(ValueError):
        pricer(100, 100, 1, 0.05, 0.2, option_type="straddle", **extra)


def test_binomial_rejects_arbitrage_parameters():
    """q must lie in ]0,1[, which fails when sigma is small relative to r."""
    with pytest.raises(ValueError):
        binomial_european(100, 100, 1, 0.20, 0.05, n=4)