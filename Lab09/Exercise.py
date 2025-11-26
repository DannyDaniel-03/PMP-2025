import itertools
import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt

PRIOR_MEAN_N = 10.0

Y_VALUES = [0, 5, 10]
THETA_VALUES = [0.2, 0.5]

def fit_model(y_obs: int, theta: float, random_seed: int = 42):
    with pm.Model() as model:
        n = pm.Poisson("n", mu=PRIOR_MEAN_N)
        y = pm.Binomial("y_obs", n=n, p=theta, observed=y_obs)
        y_future = pm.Binomial("y_future", n=n, p=theta)

        trace = pm.sample(
            draws=3000,
            tune=3000,
            chains=2,
            cores=1,
            random_seed=random_seed,
            progressbar=False,
        )

        ppc = pm.sample_posterior_predictive(
            trace,
            var_names=["y_future"],
            random_seed=random_seed,
            progressbar=False,
            return_inferencedata=False,
        )

    y_future_samples = np.ravel(ppc["y_future"])
    return trace, y_future_samples


def main():
    print("PyMC Bayesian model for Lab 8 - Ex. 1 (posterior over n and predictive Y*)")
    print(f"Prior: n ~ Poisson({PRIOR_MEAN_N})")
    print(f"Scenarios: Y in {Y_VALUES}, theta in {THETA_VALUES}\n")

    results = {}

    for idx, (y_obs, theta) in enumerate(itertools.product(Y_VALUES, THETA_VALUES), start=1):
        print(f"Scenario {idx}: Y = {y_obs}, theta = {theta}")
        trace, y_future_samples = fit_model(y_obs, theta, random_seed=2025 + idx)
        results[(y_obs, theta)] = {"trace": trace, "y_future": y_future_samples}

        n_samples = np.ravel(trace.posterior["n"].values)
        print(f"  Posterior mean of n: {float(np.mean(n_samples)):.3f}")
        print(f"  Posterior median of n: {float(np.median(n_samples)):.3f}")
        print(f"  Predictive mean of Y*: {float(np.mean(y_future_samples)):.3f}\n")

    # ---------------- Posterior for n ----------------
    fig_post, axes_post = plt.subplots(
        nrows=len(Y_VALUES),
        ncols=len(THETA_VALUES),
        figsize=(12, 8),
        sharex=False,
        sharey=False,
    )

    for i, y_obs in enumerate(Y_VALUES):
        for j, theta in enumerate(THETA_VALUES):
            ax = axes_post[i, j]
            trace = results[(y_obs, theta)]["trace"]
            az.plot_posterior(
                trace,
                var_names=["n"],
                hdi_prob=0.94,
                point_estimate="mean",
                ax=ax,
            )
            ax.set_title(f"Posterior of n | Y={y_obs}, θ={theta}")

    fig_post.suptitle("Posterior distributions of n for all (Y, θ) scenarios", y=0.98)
    fig_post.tight_layout(rect=[0, 0, 1, 0.96])

    # ------------- Predictive posterior for Y* -------------
    fig_pred, axes_pred = plt.subplots(
        nrows=len(Y_VALUES),
        ncols=len(THETA_VALUES),
        figsize=(12, 8),
        sharex=False,
        sharey=False,
    )

    for i, y_obs in enumerate(Y_VALUES):
        for j, theta in enumerate(THETA_VALUES):
            ax = axes_pred[i, j]
            y_future_samples = results[(y_obs, theta)]["y_future"]

            # Use ArviZ posterior plotting for the predictive Y*
            az.plot_posterior(
                {"y_future": y_future_samples},
                var_names=["y_future"],
                hdi_prob=0.94,
                point_estimate="mean",
                ax=ax,
            )
            ax.set_title(f"Predictive posterior of Y* | Y={y_obs}, θ={theta}")
            ax.set_xlabel("Future number of buyers Y*")

    fig_pred.suptitle("Predictive posterior distributions of Y* for all (Y, θ) scenarios", y=0.98)
    fig_pred.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()


    # ------------------------------------------------------------------
    # b) (Explanation — effect of Y and θ on the posterior for n)
    #
    #   Effect of Y (for fixed θ):
    #     - Larger Y means more buyers were observed.
    #     - To explain a larger Y with the same θ, the model prefers larger
    #       values of n (since Y ≤ n and E[Y | n, θ] = n * θ).
    #     - Therefore, as Y increases (e.g., from 0 to 10), the posterior
    #       distribution of n shifts to the right (towards larger n) and
    #       typically becomes more concentrated above Y (because n must be
    #       at least Y).
    #
    #   Effect of θ (for fixed Y):
    #     - θ is the probability that any given customer buys the product.
    #     - For the same observed Y, a smaller θ means each customer is less
    #       likely to buy, so the model needs more customers n to explain
    #       the same number of buyers Y.
    #       -> Posterior for n shifts to larger values when θ is smaller.
    #     - Conversely, a larger θ makes purchases more likely per customer,
    #       so fewer customers are needed, and the posterior for n shifts
    #       toward smaller values.
    #
    #   In short:
    #     - Increasing Y (same θ) pushes the posterior for n upward.
    #     - Decreasing θ (same Y) also pushes the posterior for n upward.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # d) (Explanation — how predictive posterior differs from posterior of n)
    #
    #   Posterior of n:
    #     - The posterior p(n | Y, θ) describes our uncertainty about the
    #       number of customers n on the observed day, after seeing
    #       Y buyers and knowing θ.
    #     - It is a distribution over the hidden parameter n, not over
    #       future data.
    #
    #   Predictive posterior for Y*:
    #     - The predictive posterior p(Y* | Y, θ) is a distribution over a
    #       future observable quantity Y* (the number of buyers on a new
    #       day), given what we learned from Y.
    #
    #   Key differences:
    #     - Different random variable:
    #         * Posterior of n: distribution over the hidden parameter n.
    #         * Predictive posterior: distribution over the future outcome Y*.
    #     - Posterior of n is about "what n was on this day given Y", while
    #       predictive posterior answers "what range of Y* values do we expect
    #       on a new day, given what we have learned about n from Y?".
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
