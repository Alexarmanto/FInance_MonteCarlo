# Monte Carlo Option Pricer

European option pricing engine using Monte Carlo simulation under Geometric Brownian Motion, with analytical Black-Scholes validation and convergence analysis.

## Description

This module prices European call and put options by simulating terminal asset prices under the risk-neutral measure (GBM). It compares Monte Carlo estimates against the closed-form Black-Scholes formula and produces a convergence chart showing how pricing error decreases as the number of simulations grows.

Part of a broader quantitative finance project series alongside [Bond Calculator](../Bond_calculator/) and [Basket Optimizer](../basket_optimizer.py).

## Methodology

### Geometric Brownian Motion (GBM)

Under the risk-neutral measure, the terminal asset price follows:

```
S_T = S_0 * exp((r - σ²/2)*T + σ*√T*Z),   Z ~ N(0,1)
```

The option price is the discounted expected payoff:

```
V_0 = e^(-rT) * E[ max(S_T - K, 0) ]   (call)
V_0 = e^(-rT) * E[ max(K - S_T, 0) ]   (put)
```

### Variance Reduction — Antithetic Variates

For each draw `Z_i`, the engine also evaluates the paired draw `-Z_i` and averages both payoffs:

```
payoff_i = ( f(Z_i) + f(-Z_i) ) / 2
```

Because `f(Z)` and `f(-Z)` are negatively correlated for call and put payoffs, their average has lower variance than a single draw — reducing the standard error without any additional random number generation.

### Black-Scholes Benchmark

The analytical call price is:

```
C = S*N(d1) - K*e^(-rT)*N(d2)
d1 = ( ln(S/K) + (r + σ²/2)*T ) / (σ*√T)
d2 = d1 - σ*√T
```

Put price follows by put-call parity: `P = C - S + K*e^(-rT)`.

The convergence plot shows the MC estimate and its 95% CI converging to the BS price at a rate of O(1/√N).

## Tech Stack

- Python 3.9+
- NumPy — vectorised GBM path generation
- SciPy — normal CDF for the Black-Scholes formula
- Matplotlib — convergence visualisation
- unittest — automated test suite

## Project Structure

```
Monte_Carlo_Simulator/
├── option_pricer.py       # EuropeanOption dataclass + MonteCarloEngine
├── black_scholes.py       # BlackScholesModel (closed-form formula)
├── main.py                # Entry point: pricing summary + convergence plot
└── test_option_pricer.py  # Unit tests: put-call parity, BS convergence, variance reduction
```

## How to Run

**Install dependencies** (from the project root):

```bash
pip install -r requirements.txt
```

**Run the pricing engine:**

```bash
cd Monte_Carlo_Simulator
python main.py
```

This prints a MC vs Black-Scholes comparison for an ATM European call and put (S=100, K=100, T=1yr, r=5%, σ=20%, N=100,000), and saves the convergence chart to `outputs/mc_option_convergence.png`.

**Run the tests:**

```bash
cd Monte_Carlo_Simulator
python -m unittest test_option_pricer.py -v
```

## Results

Running `main.py` with default parameters outputs a pricing comparison table for call and put, showing the MC estimate, standard error, 95% confidence interval, Black-Scholes analytical price, and the absolute error |MC − BS|.

The convergence chart (`outputs/mc_option_convergence.png`) shows two panels:

- **Left**: MC price converging toward the BS benchmark with a shrinking 95% CI band as N increases (log x-axis)
- **Right**: absolute pricing error |MC − BS| decreasing at rate O(1/√N) on a log-log scale
