# exercise1.py
# Lab 7 — Ex. 1: Bayesian inference for Normal(mu, sigma) with unknown mu, sigma
# Data: ambient noise levels (dB): 56, 60, 58, 55, 57, 59, 61, 56, 58, 60
# Weak prior:    mu ~ Normal(x_bar, 10),  sigma ~ HalfNormal(10)
# Strong prior:  mu ~ Normal(50, 1),     sigma ~ HalfNormal(10)
# Prints 95% HDIs and compares to frequentist estimates.

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    # ---------------------------
    # Data and frequentist stats
    # ---------------------------
    data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60], dtype=float)
    x_bar = data.mean()            # <- For part (a), x = sample mean \bar{x}
    s_hat = data.std(ddof=1)

    print_header("Data & Frequentist Estimates")
    print(f"n = {len(data)}")
    print(f"Sample mean (x_bar) = {x_bar:.3f}")
    print(f"Sample std (unbiased) = {s_hat:.3f}")

    # ---------------------------
    # (a,b) Weak prior model
    # ---------------------------
    # mu ~ Normal(x_bar, 10), sigma ~ HalfNormal(10), y ~ Normal(mu, sigma)
    with pm.Model() as weak_model:
        mu = pm.Normal("mu", mu=x_bar, sigma=10.0)
        sigma = pm.HalfNormal("sigma", sigma=10.0)
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)

        trace_weak = pm.sample(
            draws=2000, tune=2000, chains=4, cores=1, target_accept=0.9, random_seed=67
        )

    summary_weak = az.summary(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95)

    print_header("(b) Posterior summaries — Weak Prior (95% HDI)")
    print(summary_weak)

    # Optional: posterior plots
    az.plot_posterior(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95)
    plt.suptitle("Posterior — Weak Prior", fontsize=13)
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # (c) Compare to frequentist
    # ---------------------------
    print_header("(c) Frequentist vs Bayesian (Weak Prior)")
    print(f"Frequentist mean  : {x_bar:.3f}")
    print(f"Frequentist std   : {s_hat:.3f}")
    print("Note: With a weak, data-centered prior, Bayesian posteriors will be"
          " close to frequentist estimates; small differences come from prior "
          "regularization and finite-sample uncertainty.")

    # ---------------------------
    # (d) Strong prior model
    # ---------------------------
    # mu ~ Normal(50, 1), sigma ~ HalfNormal(10)
    with pm.Model() as strong_model:
        mu = pm.Normal("mu", mu=50.0, sigma=1.0)
        sigma = pm.HalfNormal("sigma", sigma=10.0)
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)

        trace_strong = pm.sample(
            draws=2000, tune=2000, chains=4, cores=1, target_accept=0.9, random_seed=67
        )

    summary_strong = az.summary(trace_strong, var_names=["mu", "sigma"], hdi_prob=0.95)

    print_header("(d) Posterior summaries — Strong Prior (95% HDI)")
    print(summary_strong)

    # Optional: posterior plots
    az.plot_posterior(trace_strong, var_names=["mu", "sigma"], hdi_prob=0.95)
    plt.suptitle("Posterior — Strong Prior", fontsize=13)
    plt.tight_layout()
    plt.show()

    print_header("Discussion (why results differ)")
    print(
        "The strong prior pulls the posterior for mu toward 50 dB, so its HDI is "
        "shifted vs. the weak-prior case and the frequentist mean. With limited data, "
        "a concentrated prior has more influence; sigma remains largely data-driven "
        "because the HalfNormal(10) prior is broad."
    )

if __name__ == "__main__":
    main()
