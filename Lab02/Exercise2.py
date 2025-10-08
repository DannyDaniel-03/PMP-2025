import numpy as np
import matplotlib.pyplot as plt
import secrets
from collections.abc import Sequence

#1
def simulate_poisson_fixed(seed: int | None = None):
    if seed is None:
        seed = secrets.randbits(64)

    rng = np.random.default_rng(seed)
    n = 1000

    datasets = {
        "Poisson_1":  rng.poisson(lam=1,  size=n),
        "Poisson_2":  rng.poisson(lam=2,  size=n),
        "Poisson_5":  rng.poisson(lam=5,  size=n),
        "Poisson_10": rng.poisson(lam=10, size=n),
    }

    #for name, arr in datasets.items():
    #    print(f"{name}: mean={arr.mean():.3f}, var={arr.var(ddof=1):.3f}, n={arr.size}")

    return datasets

#2
def simulate_poisson_randomized(seed: int | None = None):
    if seed is None:
        seed = secrets.randbits(64)

    n = 1000

    rng = np.random.default_rng(seed)
    choices = np.array([1, 2, 5, 10], dtype=np.float64)

    lambdas = rng.choice(choices, size=n, replace=True)
    samples = rng.poisson(lam=lambdas, size=n)

    #print(f"mean≈{samples.mean():.3f}, unique λ counts:",
    #      {v: int((lambdas == v).sum()) for v in {1, 2, 5, 10}})

    return samples, lambdas

#c
def simulate_poisson_skewed_randomized(seed: int | None = None, p: Sequence[float] | dict[int, float] | None = None):
    if seed is None:
        seed = secrets.randbits(64)

    n = 1000
    rng = np.random.default_rng(seed)
    choices = np.array([1, 2, 5, 10], dtype=np.float64)

    # Build probability vector
    if p is None:
        probs = np.full(4, 0.25, dtype=np.float64)
    elif isinstance(p, dict):
        probs = np.array([p.get(int(l), 0.0) for l in choices], dtype=np.float64)
    else:
        probs = np.asarray(p, dtype=np.float64)
        if probs.shape != (4,):
            raise ValueError("If p is a sequence, it must have length 4 for [1,2,5,10].")

    total = probs.sum()
    if total <= 0:
        raise ValueError("Sum of probabilities must be > 0.")
    probs = probs / total
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError("Probabilities must sum to 1 after normalization.")

    lambdas = rng.choice(choices, size=n, replace=True, p=probs)
    samples = rng.poisson(lam=lambdas, size=n)
    return samples, lambdas

#a
def plot_poisson_histograms(seed: int | None = None):
    p_skew = (0.6, 0.2, 0.1, 0.1)

    fixed = simulate_poisson_fixed(seed)
    randomized_samples, _ = simulate_poisson_randomized(seed)
    randomized_skewed, _ = simulate_poisson_skewed_randomized(seed=seed, p=p_skew)

    all_sets = {
        **fixed,
        "Randomized_{1,2,5,10}": randomized_samples,
        f"Randomized (skew p={p_skew})": randomized_skewed,
    }

    max_val = int(max(arr.max() for arr in all_sets.values()))
    bins = np.arange(-0.5, max_val + 1.5, 1)

    fig, axes = plt.subplots(len(all_sets), 1, figsize=(8, 16), constrained_layout=True)
    if len(all_sets) == 1:
        axes = [axes]

    for ax, (name, arr) in zip(axes, all_sets.items()):
        ax.hist(arr, bins=bins, density=True, alpha=0.85, edgecolor="black")
        ax.set_title(f"{name} (n={arr.size})")
        ax.set_xlabel("k")
        ax.set_ylabel("Empirical P(X = k)")
        ax.set_xlim(-0.5, max_val + 0.5)

        if max_val <= 30:
            ax.set_xticks(range(0, max_val + 1))
        else:
            step = max(1, max_val // 10)
            ax.set_xticks(range(0, max_val + 1, step))

        ax.text(
            0.98, 0.95,
            f"mean={arr.mean():.2f}\nvar={arr.var(ddof=1):.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8")
        )

    plt.show()


plot_poisson_histograms()

#b.1
#The shape is more spred out(more variance), compared to the single Poison distributions

#b.2
#This tells us that if we assume fixed lambdas, we underestimate uncertainty,

#c
#The more uniform the skew the more variance there is.