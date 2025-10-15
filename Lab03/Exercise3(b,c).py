from math import comb
from itertools import product
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([("S","M"), ("N","M"), ("S","W"), ("N","W"), ("M","W")])

# State labels
S_states = ["P0_starts","P1_starts"]     # S=0,1
N_states = [1,2,3,4,5,6]                 # N=1..6
M_states = list(range(13))               # M=0..12
W_states = ["P0","P1"]                   # W=0,1

# S ~ fair; N ~ fair die
cpd_S = TabularCPD("S", 2, [[0.5],[0.5]], state_names={"S": S_states})
cpd_N = TabularCPD("N", 6, [[1/6]]*6,     state_names={"N": N_states})

def binom_col(s_idx, n):
    p = (4/7) if s_idx == 0 else 0.5
    t = 2*n
    return [comb(t, m)*p**m*(1-p)**(t-m) if m <= t else 0.0 for m in M_states]

cols_M = [binom_col(s_idx, n) for s_idx, n in product(range(2), N_states)]
values_M = list(map(list, zip(*cols_M)))

print(values_M)

cpd_M = TabularCPD(
    "M", 13, values_M,
    evidence=["S","N"], evidence_card=[2,6],
    state_names={"M": M_states, "S": S_states, "N": N_states}
)

def winner_col(s_idx, n, m):
    w = s_idx if n >= m else 1 - s_idx
    return [1.0, 0.0] if w == 0 else [0.0, 1.0]

cols_W = [winner_col(s_idx, n, m) for s_idx, n, m in product(range(2), N_states, M_states)]
values_W = list(map(list, zip(*cols_W)))  # rows=W, cols=cartesian(S,N,M)

cpd_W = TabularCPD(
    "W", 2, values_W,
    evidence=["S","N","M"], evidence_card=[2,6,13],
    state_names={"W": W_states, "S": S_states, "N": N_states, "M": M_states}
)

print(cpd_S, cpd_N, cpd_M, cpd_W)
model.add_cpds(cpd_S, cpd_N, cpd_M, cpd_W)
assert model.check_model()

infer = VariableElimination(model)
post = infer.query(variables=["S"], evidence={"M": 1})
print(post)