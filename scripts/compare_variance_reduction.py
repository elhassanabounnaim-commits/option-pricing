"""Compare l'erreur standard Monte Carlo selon les techniques de réduction de variance."""
from pricing.black_scholes import black_scholes
from pricing.monte_carlo import monte_carlo_european

S0, T, r, sigma = 100, 1.0, 0.05, 0.2
n_paths = 100_000
seed = 42

configs = [
    ("aucune", False, False),
    ("antithétique", True, False),
    ("contrôle", False, True),
    ("les deux", True, True),
]

for option_type in ("call", "put"):
    for K in (80, 100, 120):
        bs = black_scholes(S0, K, T, r, sigma, option_type)
        print(f"\n{option_type.upper()} K={K}  (Black-Scholes = {bs:.4f})")
        print(f"{'méthode':<14}{'prix':>10}{'erreur std':>12}{'écart BS':>10}{'gain var.':>11}")

        base_se = None
        for name, anti, cv in configs:
            res = monte_carlo_european(S0, K, T, r, sigma, n_paths, option_type,
                                       seed=seed, antithetic=anti, control_variate=cv)
            if base_se is None:
                base_se = res.std_error
            gain = (base_se / res.std_error) ** 2
            print(f"{name:<14}{res.price:>10.4f}{res.std_error:>12.5f}"
                  f"{res.price - bs:>+10.4f}{gain:>10.1f}x")
