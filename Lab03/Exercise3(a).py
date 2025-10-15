import random

def coin_flip(p_heads: float = 0.5) -> str:
    return 'H' if random.random() < p_heads else 'T'

def roll_die(sides: int = 6) -> int:
    return random.randint(1, sides)

def simulate_round() -> int:
    starter = 0 if coin_flip(0.5) == 'H' else 1
    other = 1 - starter

    n = roll_die(6)

    p_heads_other = 0.5 if other == 0 else (4/7)
    m = sum(1 for _ in range(2 * n) if coin_flip(p_heads_other) == 'H')

    return starter if n >= m else other

N = 10_000

wins = [0, 0]
for _ in range(N):
    winner = simulate_round()
    wins[winner] += 1

p0_win_rate = wins[0] / N
p1_win_rate = wins[1] / N

print(f"P0 win rate ≈ {p0_win_rate:.3f}")
print(f"P1 win rate ≈ {p1_win_rate:.3f}")

if p0_win_rate > p1_win_rate:
    print("→ Player 0 has the higher chance of winning.")
else:
    print("→ Player 1 has the higher chance of winning.")
