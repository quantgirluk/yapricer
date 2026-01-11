import os
import sys

import numpy as np
import pytest

# Add parent directory to path to import yapricer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from yapricer.bspricer import BSModel


@pytest.fixture
def bs_params():
    return {
        "spot": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.2,
    }


@pytest.fixture
def model(bs_params):
    return BSModel(**bs_params)


def test_initialization(model, bs_params):
    assert model.S0 == bs_params["spot"]
    assert model.K == bs_params["K"]
    assert model.T == bs_params["T"]
    assert model.r == bs_params["r"]
    assert model.sigma == bs_params["sigma"]
    assert model.GBM is not None


def test_d1_calculation(model):
    d1 = model.d1()
    assert isinstance(d1, (float, np.floating))
    assert np.isfinite(d1)


def test_d2_calculation(model, bs_params):
    d1 = model.d1()
    d2 = model.d2()
    expected_d2 = d1 - bs_params["sigma"] * np.sqrt(bs_params["T"])
    assert isinstance(d2, (float, np.floating))
    assert d2 == pytest.approx(expected_d2, rel=0, abs=1e-10)


def test_european_call_price(model, bs_params):
    call_price = model.european_price(option_type="call")
    assert isinstance(call_price, (float, np.floating))
    assert call_price > 0
    assert call_price < bs_params["spot"]


def test_european_put_price(model, bs_params):
    put_price = model.european_price(option_type="put")
    assert isinstance(put_price, (float, np.floating))
    assert put_price > 0
    assert put_price < bs_params["K"]


def test_put_call_parity(model, bs_params):
    call_price = model.european_price(option_type="call")
    put_price = model.european_price(option_type="put")
    lhs = call_price - put_price
    rhs = bs_params["spot"] - bs_params["K"] * np.exp(-bs_params["r"] * bs_params["T"])
    assert lhs == pytest.approx(rhs, rel=0, abs=1e-8)


def test_european_price_invalid_option_type(model):
    with pytest.raises(ValueError) as excinfo:
        model.european_price(option_type="invalid")
    assert "must be 'call' or 'put'" in str(excinfo.value)


def test_call_delta(model):
    delta = model.delta(option_type="call")
    assert isinstance(delta, (float, np.floating))
    assert 0 <= delta <= 1


def test_put_delta(model):
    delta = model.delta(option_type="put")
    assert isinstance(delta, (float, np.floating))
    assert -1 <= delta <= 0


def test_delta_invalid_option_type(model):
    with pytest.raises(ValueError) as excinfo:
        model.delta(option_type="invalid")
    assert "must be 'call' or 'put'" in str(excinfo.value)


def test_gamma(model):
    gamma = model.gamma()
    assert isinstance(gamma, (float, np.floating))
    assert gamma > 0
    assert np.isfinite(gamma)


def test_vega(model):
    vega = model.vega()
    assert isinstance(vega, (float, np.floating))
    assert vega > 0
    assert np.isfinite(vega)


def test_call_theta(model):
    theta = model.theta(option_type="call")
    assert isinstance(theta, (float, np.floating))
    assert theta < 0


def test_put_theta(model):
    theta = model.theta(option_type="put")
    assert isinstance(theta, (float, np.floating))
    assert np.isfinite(theta)


def test_theta_invalid_option_type(model):
    with pytest.raises(ValueError) as excinfo:
        model.theta(option_type="invalid")
    assert "must be 'call' or 'put'" in str(excinfo.value)


def test_call_rho(model):
    rho = model.rho(option_type="call")
    assert isinstance(rho, (float, np.floating))
    assert rho > 0


def test_put_rho(model):
    rho = model.rho(option_type="put")
    assert isinstance(rho, (float, np.floating))
    assert rho < 0


def test_rho_invalid_option_type(model):
    with pytest.raises(ValueError) as excinfo:
        model.rho(option_type="invalid")
    assert "must be 'call' or 'put'" in str(excinfo.value)


def test_simulate_price(model):
    n = 100
    N = 50
    paths = model.simulate_price(n=n, N=N)
    assert paths is not None
    assert hasattr(paths, "__iter__")


def test_itm_call_option():
    model_itm = BSModel(spot=110, K=100, T=1.0, r=0.05, sigma=0.2)
    call_price = model_itm.european_price(option_type="call")
    intrinsic_value = 10
    assert call_price > intrinsic_value


def test_otm_call_option():
    model_otm = BSModel(spot=90, K=100, T=1.0, r=0.05, sigma=0.2)
    call_price = model_otm.european_price(option_type="call")
    assert call_price > 0


def test_itm_put_option():
    model_itm = BSModel(spot=90, K=100, T=1.0, r=0.05, sigma=0.2)
    put_price = model_itm.european_price(option_type="put")
    intrinsic_value = 10
    assert put_price > intrinsic_value


def test_zero_volatility_edge_case():
    model_low_vol = BSModel(spot=100, K=100, T=1.0, r=0.05, sigma=0.01)
    call_price = model_low_vol.european_price(option_type="call")
    assert call_price > 0
    assert np.isfinite(call_price)


def test_long_maturity():
    model_long = BSModel(spot=100, K=100, T=5.0, r=0.05, sigma=0.2)
    call_price = model_long.european_price(option_type="call")
    assert call_price > 0
    assert np.isfinite(call_price)


def test_short_maturity():
    model_short = BSModel(spot=100, K=100, T=0.1, r=0.05, sigma=0.2)
    call_price = model_short.european_price(option_type="call")
    put_price = model_short.european_price(option_type="put")
    assert call_price > 0
    assert put_price > 0


def test_high_volatility(model):
    model_high_vol = BSModel(spot=100, K=100, T=1.0, r=0.05, sigma=0.5)
    call_price_high = model_high_vol.european_price(option_type="call")
    call_price_low = model.european_price(option_type="call")
    assert call_price_high > call_price_low


def test_greeks_consistency(model):
    delta_call = model.delta(option_type="call")
    delta_put = model.delta(option_type="put")
    gamma = model.gamma()
    vega = model.vega()
    assert delta_call - delta_put == pytest.approx(1.0, rel=0, abs=1e-8)
    assert all(np.isfinite(x) for x in [delta_call, delta_put, gamma, vega])


def test_very_deep_itm_call():
    model_deep_itm = BSModel(spot=200, K=100, T=1.0, r=0.05, sigma=0.2)
    call_price = model_deep_itm.european_price(option_type="call")
    delta = model_deep_itm.delta(option_type="call")
    assert delta > 0.9
    assert call_price > 100


def test_very_deep_otm_put():
    model_deep_otm = BSModel(spot=200, K=100, T=1.0, r=0.05, sigma=0.2)
    put_price = model_deep_otm.european_price(option_type="put")
    delta = model_deep_otm.delta(option_type="put")
    assert abs(delta) < 0.1
    assert 0 < put_price < 1


@pytest.mark.parametrize(
    "params",
    [
        {"spot": 50, "K": 55, "T": 0.5, "r": 0.03, "sigma": 0.15},
        {"spot": 150, "K": 140, "T": 2.0, "r": 0.07, "sigma": 0.3},
        {"spot": 100, "K": 100, "T": 0.25, "r": 0.02, "sigma": 0.25},
    ],
)
def test_different_parameters(params):
    model_varied = BSModel(**params)
    call_price = model_varied.european_price(option_type="call")
    put_price = model_varied.european_price(option_type="put")
    assert call_price > 0
    assert put_price > 0
    assert np.isfinite(call_price)
    assert np.isfinite(put_price)
