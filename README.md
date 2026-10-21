# Option Pricing Engine

Pricing of European options under risk-neutral valuation, implemented from scratch in Python.
Three independent methods are implemented and cross-validated against one another, so that a
disagreement between them signals a bug rather than going unnoticed.

## Pricing framework

Under the no-arbitrage assumption, the price of an option is the cost of the portfolio that
replicates its payoff. That cost can be rewritten as a discounted expectation under a specific
probability measure $\mathbb{Q}$, the risk-neutral measure, under which the discounted asset
price is a martingale:

```math
C_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\!\left[\,\text{payoff}(S_T)\,\right]
```

The three methods below are three ways of computing that same expectation. Notably, the real
probability of an upward move never appears: the price is determined by replication, which
must hold scenario by scenario, not on average.

## Implemented methods

| Method | Scope | Error | Role |
| --- | --- | --- | --- |
| Black-Scholes | European vanilla | Exact (closed form) | Reference value |
| Cox-Ross-Rubinstein binomial tree | European | `O(1/n)` in the number of steps | Discrete-time approach |
| Monte Carlo | Any simulable payoff | `O(1/sqrt(N))`, with confidence interval | General method |

### Black-Scholes

Evaluates the expectation analytically under a lognormal terminal distribution:

```math
C_0 = S_0\,N(d_1) - K e^{-rT} N(d_2), \qquad
d_1 = \frac{\ln(S_0/K) + \left(r + \frac{\sigma^2}{2}\right)T}{\sigma\sqrt{T}}, \qquad
d_2 = d_1 - \sigma\sqrt{T}
```

where $N$ is the standard normal CDF. Here $N(d_2)$ is the risk-neutral probability of
exercise, while $N(d_1)$ is the delta of the call, that is the number of shares held in the
replicating portfolio. Read that way, the formula is the replicating portfolio written in
continuous time: a long position $S_0 N(d_1)$ in the asset, financed by borrowing
$K e^{-rT} N(d_2)$.

Exact and constant-time, but restricted to European vanilla payoffs under geometric Brownian
motion.

### Binomial tree

Computes the expectation by backward induction. Starting from the payoffs at maturity, the
value at each node is the discounted risk-neutral expectation of the two nodes that follow:

```math
C_{\text{node}} = e^{-r\Delta t}\left[\,q\,C_u + (1-q)\,C_d\,\right], \qquad
q = \frac{e^{r\Delta t} - d}{u - d}
```

Steps are calibrated to the volatility following Cox-Ross-Rubinstein:

```math
u = e^{\sigma\sqrt{\Delta t}}, \qquad d = \frac{1}{u}, \qquad \Delta t = \frac{T}{n}
```

Taking $d = 1/u$ keeps the tree recombining, which reduces the number of terminal nodes from
$2^n$ to $n+1$ and makes trees with hundreds of steps tractable. The implementation raises an
error if $q \notin\, ]0,1[$, which is the no-arbitrage condition $d < e^{r\Delta t} < u$.

### Monte Carlo

Estimates the expectation by sampling. The terminal price is simulated exactly from the
closed-form solution of the SDE, with a single Gaussian draw per path:

```math
S_T^{(i)} = S_0 \exp\!\left[\left(r - \frac{\sigma^2}{2}\right)T + \sigma\sqrt{T}\,Z_i\right],
\qquad Z_i \stackrel{\text{iid}}{\sim} \mathcal{N}(0,1)
```

No Euler scheme is needed, so there is no discretisation bias and the only error is
statistical. The central limit theorem then gives the precision directly:

```math
\hat{C}_N = \frac{1}{N}\sum_{i=1}^{N} X_i, \qquad
\mathrm{SE} = \frac{\hat{\sigma}_X}{\sqrt{N}}, \qquad
\mathrm{CI}_{95\%} = \hat{C}_N \pm 1.96\,\mathrm{SE}
```

where $X_i$ is the discounted payoff of path $i$. The estimator returns its own standard error
alongside the price: a Monte Carlo result without an error bar gives no indication of how many
digits are significant.

## Variance reduction

Monte Carlo error is $\hat{\sigma}_X/\sqrt{N}$, so dividing it by 10 requires 100 times more
paths. Rather than increasing $N$, both techniques below build a different unbiased estimator
of the same expectation with a smaller variance.

**Antithetic variates** evaluate the payoff at both $Z$ and $-Z$, which are equally valid
Gaussian draws, and average each pair:

```math
X_i^{\text{anti}} = \frac{X(Z_i) + X(-Z_i)}{2}, \qquad
\mathrm{Var}\big(X^{\text{anti}}\big) = \frac{\mathrm{Var}(X)}{2}\,(1 + \rho)
```

The method gains when $\rho$, the correlation between the two payoffs, is negative. That holds
whenever the payoff is monotone in $Z$, as it is for a call or a put, and fails on a
non-monotone payoff such as a straddle.

**A control variate** uses a quantity $Y$ whose expectation is known exactly and that is
correlated with the payoff. Here $Y = S_T$, whose risk-neutral expectation is $S_0 e^{rT}$:

```math
X^{\text{cv}} = X - c\,\big(Y - \mathbb{E}[Y]\big), \qquad
c^\star = \frac{\mathrm{Cov}(X,Y)}{\mathrm{Var}(Y)}, \qquad
\mathrm{Var}\big(X^{\text{cv}}\big) = \mathrm{Var}(X)\,\big(1 - \rho_{X,Y}^2\big)
```

The estimator is unbiased for any $c$. The optimal $c^\star$ is the slope of a linear
regression of $X$ on $Y$, and the residual variance is the familiar $1 - R^2$.

## Results

All figures use $S_0 = K = 100$, $T = 1$, $r = 5\%$, $\sigma = 20\%$, and are reproducible with
the seeds stated in the code.

### Cross-validation

| Method | Call | Put |
| --- | --- | --- |
| Black-Scholes (exact) | 10.4506 | 5.5735 |
| Binomial tree, 100 steps | 10.4306 | 5.5536 |
| Monte Carlo, 100k paths, seed 42 | 10.4205 | — |

The Monte Carlo 95% confidence interval is `[10.3289, 10.5122]`, which contains the exact
value.

Put-call parity is satisfied to `1e-4` by every method:

```math
C - P = S_0 - K e^{-rT}
```

This identity is model-free, following from no-arbitrage alone, which makes it a strong
correctness test: any pricer that violates it has a bug, with no further discussion needed.

### Binomial convergence

Fitting the absolute error against the number of steps on a log-log scale gives slopes of
`-0.998` and `-1.001`, confirming the theoretical `O(1/n)` rate.

The error oscillates with the parity of $n$. Terminal nodes are $S_0 u^{2k-n}$, whose exponent
has the parity of $n$: an even number of steps places a node exactly on the strike, that is on
the kink of the payoff, while an odd number places the strike between two nodes. The error
therefore splits into two interleaved subsequences that bracket the exact value, which is why
they are fitted and plotted separately.

### Variance reduction

Measured at an equal number of payoff evaluations:

| Estimator | Standard error | Variance reduction |
| --- | --- | --- |
| Plain Monte Carlo | 0.04677 | 1.00 |
| Antithetic variates | 0.03308 | 2.00 |
| Control variate | 0.01788 | 6.85 |
| Both combined | 0.02489 | 3.53 |

The control variate factor matches the theoretical $1/(1-\rho^2)$ with a measured correlation
of `0.9241` between the discounted payoff and the terminal spot.

Combining both techniques performs *worse* than the control variate alone. Regressing on the
terminal spot removes the component of the payoff that is linear, and therefore monotone, in
the Gaussian draw, which is exactly what antithetic variates exploit. The correlation between
antithetic pairs moves from `-0.499` on raw payoffs to `+0.948` on control-variate residuals,
so the pairing increases the variance instead of reducing it. Variance reduction techniques do
not compose naively: each one changes the structure the next one relies on.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```python
from pricing.black_scholes import black_scholes
from pricing.binomial import binomial_european
from pricing.monte_carlo import monte_carlo_european

black_scholes(100, 100, 1, 0.05, 0.2, "call")
# 10.450583572185565

binomial_european(100, 100, 1, 0.05, 0.2, n=100, option_type="call")
# 10.430611662249113

monte_carlo_european(100, 100, 1, 0.05, 0.2, n_paths=100_000, seed=42)
# MC(price=10.420541..., std_error=0.046769..., ci=(10.328872..., 10.512210...))
```

## Project structure

```
option-pricing/
├── pricing/            pricing library, one module per method
├── notebooks/          convergence studies and figures
├── tests/              unit tests
├── README.md
└── requirements.txt
```

## Model assumptions and limitations

All three methods implement Black-Scholes dynamics: geometric Brownian motion with constant
volatility, no dividends, frictionless continuous trading. Market data contradicts those
assumptions in at least four ways. Log-returns have far heavier tails than the Gaussian, with
empirical kurtosis well above 3. Their distribution is left-skewed, since equities fall faster
than they rise. Prices jump, which no continuous Brownian path reproduces. And volatility is
neither constant nor deterministic, calm and turbulent periods clustering together.

The model remains the market standard, not as a description of reality but as a common
language: traders quote in implied volatility, and Black-Scholes is the function that maps a
volatility quote to a price.

One implementation caveat: the control variate coefficient $c^\star$ is estimated on the same
sample used for the price, which introduces an `O(1/N)` bias. It is negligible at the sample
sizes used here, and would be removed with a separate pilot run if needed.

## Roadmap

- Greeks: analytic, finite-difference and pathwise estimators
- American options: binomial tree with early exercise, and Longstaff-Schwartz least-squares
  Monte Carlo
- Implied volatility by numerical inversion, and the volatility smile
- Unit test suite covering put-call parity, convergence rates and cross-method consistency

## References

- Hull, *Options, Futures and Other Derivatives*
- Glasserman, *Monte Carlo Methods in Financial Engineering*