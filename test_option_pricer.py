import unittest
import numpy as np
from option_pricer import EuropeanOption, MonteCarloEngine
from black_scholes import BlackScholesModel


class TestBlackScholesModel(unittest.TestCase):

    def test_put_call_parity(self):
        """C - P = S - K*e^(-rT) holds exactly in the analytical formula."""
        call = BlackScholesModel(EuropeanOption(100, 100, 1.0, 0.05, 0.20, 'call')).price()
        put  = BlackScholesModel(EuropeanOption(100, 100, 1.0, 0.05, 0.20, 'put')).price()
        expected = 100 - 100 * np.exp(-0.05 * 1.0)
        self.assertAlmostEqual(call - put, expected, places=8)

    def test_atm_call_positive(self):
        """ATM call must carry positive time value."""
        price = BlackScholesModel(
            EuropeanOption(100, 100, 1.0, 0.05, 0.20, 'call')
        ).price()
        self.assertGreater(price, 0)

    def test_deep_itm_call_approaches_intrinsic(self):
        """Deep ITM call price converges to S - K*e^(-rT)."""
        opt = EuropeanOption(S=300, K=50, T=1.0, r=0.05, sigma=0.20, option_type='call')
        intrinsic = 300 - 50 * np.exp(-0.05 * 1.0)
        self.assertAlmostEqual(BlackScholesModel(opt).price(), intrinsic, delta=0.01)


class TestMonteCarloEngine(unittest.TestCase):

    def test_mc_price_within_bs_confidence_interval(self):
        """MC price (100k paths, antithetic) must be within 2σ of the BS price."""
        opt = EuropeanOption(100, 100, 1.0, 0.05, 0.20, 'call')
        result = MonteCarloEngine(opt, n_sims=100_000, seed=0).price()
        bs_price = BlackScholesModel(opt).price()
        self.assertAlmostEqual(result['price'], bs_price, delta=2 * result['std_error'])

    def test_put_call_parity_mc(self):
        """MC call - put ≈ S - K*e^(-rT) when both engines share the same seed."""
        call_price = MonteCarloEngine(
            EuropeanOption(100, 100, 1.0, 0.05, 0.20, 'call'), n_sims=500_000, seed=42
        ).price()['price']
        put_price = MonteCarloEngine(
            EuropeanOption(100, 100, 1.0, 0.05, 0.20, 'put'), n_sims=500_000, seed=42
        ).price()['price']
        expected = 100 - 100 * np.exp(-0.05 * 1.0)
        self.assertAlmostEqual(call_price - put_price, expected, delta=0.05)

    def test_antithetic_reduces_std_error(self):
        """Antithetic variates must produce a strictly lower std error than naive MC."""
        opt = EuropeanOption(100, 100, 1.0, 0.05, 0.20, 'call')
        std_anti  = MonteCarloEngine(opt, n_sims=50_000, seed=1).price(use_antithetic=True)['std_error']
        std_naive = MonteCarloEngine(opt, n_sims=50_000, seed=1).price(use_antithetic=False)['std_error']
        self.assertLess(std_anti, std_naive)

    def test_deep_otm_call_near_zero(self):
        """Deep OTM call should price near zero."""
        opt = EuropeanOption(S=50, K=200, T=0.5, r=0.05, sigma=0.20, option_type='call')
        result = MonteCarloEngine(opt, n_sims=50_000, seed=99).price()
        self.assertAlmostEqual(result['price'], 0.0, delta=0.01)

    def test_invalid_option_type_raises(self):
        """EuropeanOption must reject an invalid option_type."""
        with self.assertRaises(ValueError):
            EuropeanOption(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type='american')

    def test_nonpositive_maturity_raises(self):
        """EuropeanOption must reject T <= 0."""
        with self.assertRaises(ValueError):
            EuropeanOption(S=100, K=100, T=0.0, r=0.05, sigma=0.20, option_type='call')


if __name__ == '__main__':
    print("Running Monte Carlo Simulator tests...")
    unittest.main()
