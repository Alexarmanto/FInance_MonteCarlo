from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class EuropeanOption:
    S: float       # spot price
    K: float       # strike price
    T: float       # time to maturity (years)
    r: float       # risk-free rate (annualised, continuous compounding)
    sigma: float   # volatility (annualised)
    option_type: str  # 'call' or 'put'

    def __post_init__(self) -> None:
        if self.option_type not in ('call', 'put'):
            raise ValueError("option_type must be 'call' or 'put'")
        if self.T <= 0:
            raise ValueError("T must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.S <= 0:
            raise ValueError("S must be positive")


class MonteCarloEngine:
    """
    Prices European options by simulating terminal asset prices under
    Geometric Brownian Motion (risk-neutral measure).

    Variance reduction is achieved via antithetic variates: for each
    standard normal draw Z, the paired draw -Z is also evaluated and
    the two payoffs are averaged. This exploits the negative correlation
    between f(Z) and f(-Z) to reduce estimator variance at no extra cost
    in random number generation.
    """

    def __init__(
        self,
        option: EuropeanOption,
        n_sims: int = 100_000,
        seed: Optional[int] = None,
    ) -> None:
        self.option = option
        self.n_sims = n_sims
        self.rng = np.random.default_rng(seed)

    def _payoff(self, S_T: np.ndarray) -> np.ndarray:
        if self.option.option_type == 'call':
            return np.maximum(S_T - self.option.K, 0.0)
        return np.maximum(self.option.K - S_T, 0.0)

    def price(self, use_antithetic: bool = True) -> dict:
        """
        Estimate option price via Monte Carlo simulation.

        Returns a dict with:
          'price'     — discounted expected payoff (MC estimate)
          'std_error' — standard error of the estimate
          'ci_95'     — (lower, upper) bounds of the 95% confidence interval
        """
        opt = self.option
        drift = (opt.r - 0.5 * opt.sigma ** 2) * opt.T
        diffusion = opt.sigma * np.sqrt(opt.T)
        discount = np.exp(-opt.r * opt.T)

        Z = self.rng.standard_normal(self.n_sims)
        S_T = opt.S * np.exp(drift + diffusion * Z)

        if use_antithetic:
            S_T_anti = opt.S * np.exp(drift - diffusion * Z)
            payoffs = (self._payoff(S_T) + self._payoff(S_T_anti)) / 2
        else:
            payoffs = self._payoff(S_T)

        discounted = discount * payoffs
        mc_price = float(discounted.mean())
        std_error = float(discounted.std() / np.sqrt(self.n_sims))

        return {
            'price': mc_price,
            'std_error': std_error,
            'ci_95': (mc_price - 1.96 * std_error, mc_price + 1.96 * std_error),
        }
