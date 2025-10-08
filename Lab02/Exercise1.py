import random
from fractions import Fraction

#a
def trial() -> bool:
    r, b, k = 3, 4, 2

    d = random.randint(1, 6)
    if d in (2, 3, 5):
        k += 1
    elif d == 6:
        r += 1
    else:
        b += 1

    total = r + b + k
    pick = random.randint(1, total)
    return pick <= r

#b
def estimate_p_red(n_trials: int, seed: int | None = None) -> float:
    if seed is not None:
        random.seed(seed)
    red = sum(trial() for _ in range(n_trials))
    p_red = red / n_trials
    return p_red

#c
def theoretical_p_red() -> tuple[Fraction, float]:
    r, b, k = 3, 4, 2

    outcomes = [
        (Fraction(3, 6), (0, 0, 1)),  # prime (2,3,5): add black
        (Fraction(1, 6), (1, 0, 0)),  # 6: add red
        (Fraction(2, 6), (0, 1, 0)),  # other (1,4): add blue
    ]

    p_red = Fraction(0, 1)
    for p_outcome, (dR, dB, dK) in outcomes:
        r2, b2, k2 = r + dR, b + dB, k + dK
        p_red_given = Fraction(r2, r2 + b2 + k2)
        p_red += p_outcome * p_red_given

    return p_red, float(p_red)

p = estimate_p_red(n_trials=100_000)
print(f"Estimated P(red) ≈ {p:.6f}")

exact, floatApprox = theoretical_p_red()
print(f"Exact P(red) = {exact} ≈ {floatApprox:.6f}")