from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([("D","C")])

cpd_D = TabularCPD("D", 3, [[0.5],[1/6],[1/3]])  # [prime, six, other]
cpd_C = TabularCPD("C", 3,
    values=[
        [0.3, 0.4, 0.3],  # C=Red   | D=[prime, six, other]
        [0.4, 0.4, 0.5],  # C=Blue
        [0.3, 0.2, 0.2],  # C=Black
    ],
    evidence=["D"], evidence_card=[3]
)

model.add_cpds(cpd_D, cpd_C)
infer = VariableElimination(model)
p_red = float(infer.query(["C"]).values[0])
print(p_red)
