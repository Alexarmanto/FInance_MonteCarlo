import numpy as np
from scipy.stats import norm
from option_pricer import EuropeanOption


class BlackScholesModel:
    """
    Analytical Black-Scholes pricing for European call and put options.
    Used as the benchmark to validate the Monte Carlo engine.
    """

    def __init__(self, option: EuropeanOption) -> None:
        self.option = option

    def _d1_d2(self) -> tuple[float, float]:
        opt = self.option
        d1 = (
            np.log(opt.S / opt.K) + (opt.r + 0.5 * opt.sigma ** 2) * opt.T
        ) / (opt.sigma * np.sqrt(opt.T))
        d2 = d1 - opt.sigma * np.sqrt(opt.T)
        return float(d1), float(d2)

    def price(self) -> float:
        opt = self.option
        d1, d2 = self._d1_d2()
        discount = np.exp(-opt.r * opt.T)

        if opt.option_type == 'call':
            return float(opt.S * norm.cdf(d1) - opt.K * discount * norm.cdf(d2))
        return float(opt.K * discount * norm.cdf(-d2) - opt.S * norm.cdf(-d1))
