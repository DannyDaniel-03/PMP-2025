from itertools import product
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1) Structura: S→O, S→L, S→M, L→M
model = DiscreteBayesianNetwork([("S","O"), ("S","L"), ("S","M"), ("L","M")])

# 2) CPD-uri
cpd_S = TabularCPD("S", 2, [[0.6],[0.4]])  # P(S)
cpd_O = TabularCPD("O", 2,                        # P(O|S)
                   values=[[0.9, 0.3], [0.1, 0.7]],
                   evidence=["S"], evidence_card=[2])
cpd_L = TabularCPD("L", 2,                 # P(L|S)
                   values=[[0.7, 0.2], [0.3, 0.8]],
                   evidence=["S"], evidence_card=[2])
p_M1 = [0.2, 0.6, 0.5, 0.9]                # P(M=1 | S,L) în ordinea (S=0,L=0), (0,1),(1,0),(1,1)
cpd_M = TabularCPD("M", 2,
                   values=[[1-p for p in p_M1], p_M1],
                   evidence=["S","L"], evidence_card=[2,2])

model.add_cpds(cpd_S, cpd_O, cpd_L, cpd_M)
assert model.check_model(), "Model invalid (dimensiuni/normalizare CPD)."

# 3) Independențe (poți folosi oricare din cele două)
print("Local independencies:")
print(model.local_independencies(["S","O","L","M"]))
print("\nAll independencies (structurale):")
print(model.get_independencies())

# 4) Inferență: P(S=1 | O,L,M) + verdict
infer = VariableElimination(model)

def classify(o, l, m, thr=0.5):
    post = infer.query(variables=["S"], evidence={"O": o, "L": l, "M": m})
    p_s1 = float(post.values[1])  # [P(S=0|...), P(S=1|...)]
    return p_s1, ("SPAM" if p_s1 >= thr else "non-spam")

print("\nTabel clasificare (threshold 0.5):")
print(" O  L  M   P(S=1|O,L,M)   Verdict")
for o, l, m in product([0,1],[0,1],[0,1]):
    p, v = classify(o,l,m)
    print(f" {o}  {l}  {m}     {p:0.3f}         {v}")
