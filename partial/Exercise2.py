import numpy as np
from hmmlearn import hmm
import random

#a
states = ["W", "R", "S"]
obs_symbols = ["L", "M", "H"]

pi = np.array([0.4, 0.3, 0.3])    # P(W), P(R), P(S)

A = np.array([
    [0.6, 0.3, 0.1],  # W to [W, R, S]
    [0.2, 0.7, 0.1],  # R to [W, R, S]
    [0.3, 0.2, 0.5]   # S to [W, R, S]
])

B = np.array([
    [0.10, 0.70, 0.20],  # W emits [L, M, H]
    [0.05, 0.25, 0.70],  # R emits [L, M, H]
    [0.80, 0.15, 0.05]   # S emits [L, M, H]
])

obs_seq = [1, 2, 0]
X = np.array(obs_seq).reshape(-1, 1)

model = hmm.CategoricalHMM(n_components=3, init_params="")
model.startprob_ = pi
model.transmat_ = A
model.emissionprob_ = B

#b
log_prob = model.score(X)
P_MHL = np.exp(log_prob)
print("P(M,H,L) =", P_MHL)

#c
log_p_star, path_idx = model.decode(X, algorithm="viterbi")
path_states = [states[i] for i in path_idx]

print("Most likely hidden state sequence:", path_states)
print("Log-prob of this sequence:", log_p_star)
print("Prob of this sequence:", np.exp(log_p_star))

# Why Viterbi instead of brute force for long sequences?
# T = number of observations, N = number of states
# Brute force checks all N^T state sequences (exponential in T).
# Viterbi uses dynamic programming and runs in O(N^2 * T),
# which is much more efficient for long sequences.


#d
#random.seed(42)

def sample_categorical(probs):
    r = random.random()
    s = 0.0
    for i, p in enumerate(probs):
        s += p
        if r <= s:
            return i
    return len(probs) - 1   # in case of rounding error


def generate_sequence(pi, A, B, T=3):
    path = [sample_categorical(pi)]
    obs_seq_gen = [sample_categorical(B[path[0]])]
    for _ in range(1, T):
        path.append(sample_categorical(A[path[-1]]))
        obs_seq_gen.append(sample_categorical(B[path[-1]]))
    return path, obs_seq_gen


def estimate_probability(num_sequences=10000):
    target = obs_seq
    count = 0
    for _ in range(num_sequences):
        _, obs_gen = generate_sequence(pi, A, B, T=len(target))
        if obs_gen == target:
            count += 1
    return count / num_sequences


empirical_P = estimate_probability(10000)
print("Empirical P(M, H, L) from 10,000 samples =", empirical_P)


