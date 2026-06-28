import os
import numpy as np
import matplotlib.pyplot as plt
from option_pricer import EuropeanOption, MonteCarloEngine
from black_scholes import BlackScholesModel


def _print_result(label: str, mc_result: dict, bs_price: float) -> None:
    print(f"\n  {label}")
    print("-" * 54)
    print(f"  MC Price        : {mc_result['price']:.4f}")
    print(f"  Std Error       : {mc_result['std_error']:.6f}")
    print(f"  95% CI          : [{mc_result['ci_95'][0]:.4f}, {mc_result['ci_95'][1]:.4f}]")
    print(f"  BS Price        : {bs_price:.4f}")
    print(f"  |MC − BS|       : {abs(mc_result['price'] - bs_price):.6f}")


def run_convergence_analysis(option: EuropeanOption, n_max: int = 100_000) -> None:
    """
    Generate all n_max paths once with antithetic variates, then compute
    the running average for geometrically-spaced values of N.
    This produces a smooth convergence curve without re-seeding.
    """
    bs_price = BlackScholesModel(option).price()

    rng = np.random.default_rng(42)
    Z = rng.standard_normal(n_max)
    drift = (option.r - 0.5 * option.sigma ** 2) * option.T
    diffusion = option.sigma * np.sqrt(option.T)
    discount = np.exp(-option.r * option.T)

    S_T = option.S * np.exp(drift + diffusion * Z)
    S_T_anti = option.S * np.exp(drift - diffusion * Z)

    if option.option_type == 'call':
        payoffs = discount * (
            np.maximum(S_T - option.K, 0) + np.maximum(S_T_anti - option.K, 0)
        ) / 2
    else:
        payoffs = discount * (
            np.maximum(option.K - S_T, 0) + np.maximum(option.K - S_T_anti, 0)
        ) / 2

    sim_counts = np.unique(np.geomspace(100, n_max, 80).astype(int))
    running_prices = np.array([payoffs[:n].mean() for n in sim_counts])
    running_std = np.array([payoffs[:n].std() / np.sqrt(n) for n in sim_counts])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Monte Carlo vs Black-Scholes — European {option.option_type.capitalize()}   "
        f"(S={option.S}, K={option.K}, T={option.T}yr, "
        f"r={option.r:.0%}, σ={option.sigma:.0%})",
        fontsize=11,
    )

    # Panel 1 — MC price convergence with CI band
    ax1.plot(sim_counts, running_prices, color='steelblue', linewidth=1.5,
             label='MC Price (antithetic)')
    ax1.fill_between(
        sim_counts,
        running_prices - 1.96 * running_std,
        running_prices + 1.96 * running_std,
        alpha=0.25, color='steelblue', label='95% CI',
    )
    ax1.axhline(
        bs_price, color='crimson', linestyle='--', linewidth=1.5,
        label=f'BS Price: {bs_price:.4f}',
    )
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Simulations (log scale)')
    ax1.set_ylabel('Option Price')
    ax1.set_title('MC Price Convergence')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2 — Absolute error |MC - BS| on log-log scale
    errors = np.abs(running_prices - bs_price)
    positive_mask = errors > 0
    if positive_mask.any():
        ax2.plot(sim_counts[positive_mask], errors[positive_mask],
                 color='darkorange', linewidth=1.5)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Number of Simulations (log scale)')
    ax2.set_ylabel('|MC Price − BS Price| (log scale)')
    ax2.set_title('Absolute Pricing Error vs N')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'mc_option_convergence.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nConvergence plot saved → {out_path}")
    plt.show()


def main() -> None:
    option_call = EuropeanOption(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type='call')
    option_put  = EuropeanOption(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type='put')

    mc_call = MonteCarloEngine(option_call, n_sims=100_000, seed=42).price()
    mc_put  = MonteCarloEngine(option_put,  n_sims=100_000, seed=42).price()
    bs_call = BlackScholesModel(option_call).price()
    bs_put  = BlackScholesModel(option_put).price()

    print("=" * 54)
    print("  EUROPEAN OPTION PRICING — MC vs Black-Scholes")
    print(f"  S={option_call.S} | K={option_call.K} | T={option_call.T}yr "
          f"| r={option_call.r:.0%} | σ={option_call.sigma:.0%}")
    print("=" * 54)
    _print_result("CALL", mc_call, bs_call)
    _print_result("PUT",  mc_put,  bs_put)
    print("=" * 54)

    run_convergence_analysis(option_call)


if __name__ == "__main__":
    main()
