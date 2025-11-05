# Part (a) — Define the HMM (params from Lab 5) and draw the state diagram
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import CategoricalHMM
from matplotlib.patches import FancyArrowPatch

# -----------------------------
# HMM definition
# -----------------------------
states = ["Difficult", "Medium", "Easy"]
observations = ["FB", "B", "S", "NS"]

startprob = np.array([1/3, 1/3, 1/3], dtype=float)

transmat = np.array([
    [0.00, 0.50, 0.50],  # Difficult -> {M,E}
    [0.50, 0.25, 0.25],  # Medium    -> {D,M,E}
    [0.50, 0.25, 0.25],  # Easy      -> {D,M,E}
], dtype=float)

emissionprob = np.array([
    [0.10, 0.20, 0.40, 0.30],  # Difficult: FB, B, S, NS
    [0.15, 0.25, 0.50, 0.10],  # Medium
    [0.20, 0.30, 0.40, 0.10],  # Easy
], dtype=float)

pos = {
    "Difficult": (0.10, 0.50),
    "Medium":    (0.62, 0.82),
    "Easy":      (0.62, 0.18),
}
state_idx = {s:i for i,s in enumerate(states)}
node_r = 0.05

fig, ax = plt.subplots(figsize=(7, 4))
ax.set_axis_off()

# Draw nodes
for s, (x, y) in pos.items():
    circ = plt.Circle((x, y), node_r, fill=False, lw=1.5)
    ax.add_patch(circ)
    ax.text(x, y, s, ha="center", va="center", fontsize=10)

def draw_edge(p_from, p_to, p, self_loop=False):
    if p <= 0:
        return
    (x1, y1) = pos[p_from]
    (x2, y2) = pos[p_to]

    if self_loop:
        # draw a small loop on the right side of the node
        start = (x1 + node_r*0.7, y1 + node_r*0.2)
        end   = (x1 + node_r*0.7, y1 - node_r*0.2)
        patch = FancyArrowPatch(start, end,
                                connectionstyle="arc3,rad=1.2",
                                arrowstyle="->", lw=1.2, mutation_scale=10)
        ax.add_patch(patch)
        ax.text(x1 + node_r*1.15, y1, f"{p:.2f}", ha="left", va="center", fontsize=9)
        return

    # offset so arrows don't touch node boundaries
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy)
    if L == 0:
        return
    ux, uy = dx / L, dy / L
    start = (x1 + ux*node_r, y1 + uy*node_r)
    end   = (x2 - ux*node_r, y2 - uy*node_r)

    patch = FancyArrowPatch(start, end, arrowstyle="->", lw=1.2,
                            mutation_scale=10, connectionstyle="arc3,rad=0.1")
    ax.add_patch(patch)
    mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    ax.text(mx, my, f"{p:.2f}", ha="center", va="center", fontsize=9)

# Draw edges (including self-loops if any > 0)
for i, s_from in enumerate(states):
    for j, s_to in enumerate(states):
        p = transmat[i, j]
        draw_edge(s_from, s_to, p, self_loop=(i == j and p > 0))

ax.set_title("HMM State Diagram — Lab 5 (a)", fontsize=12)
plt.tight_layout()
plt.show()

# Quick printout of parameters
print("States:", states)
print("Observations:", observations)
print("Start probabilities:", startprob)
print("Transition matrix:\n", transmat)
print("Emission probabilities (rows=state, cols=FB,B,S,NS):\n", emissionprob)


seq_labels = ["FB","FB","S","B","B","S","B","B","NS","B","B"]
obs_index = {o:i for i,o in enumerate(observations)}
O = np.array([obs_index[x] for x in seq_labels], dtype=int).reshape(-1, 1)

model = CategoricalHMM(n_components=3, init_params="")
model.startprob_   = startprob
model.transmat_    = transmat
model.emissionprob_= emissionprob

logP = model.score(O)
P = float(np.exp(logP))

print("\nSequence:", seq_labels)
print("log P(O|λ) =", logP)
print("P(O|λ)     =", P)

v_logP, path_idx = model.decode(O, algorithm="viterbi")  # log P*(O, X*)
v_prob = float(np.exp(v_logP))
path_labels = [states[i] for i in path_idx]

print("\nViterbi log-probability:", v_logP)
print("Viterbi probability:    ", v_prob)
print("Path (indices):         ", path_idx.tolist())
print("Path (labels):          ", path_labels)
