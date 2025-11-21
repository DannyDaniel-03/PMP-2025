from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# O -> H, W
# H, W -> R
# H -> E
# R -> C
model = DiscreteBayesianNetwork([
    ('O', 'H'),
    ('O', 'W'),
    ('H', 'R'),
    ('W', 'R'),
    ('H', 'E'),
    ('R', 'C')
])

# O
cpd_O = TabularCPD(
    variable='O',
    variable_card=2,
    values=[[0.3],      # O = cold
            [0.7]],     # O = mild
    state_names={'O': ['cold', 'mild']}
)

# H|O
cpd_H = TabularCPD(
    variable='H',
    variable_card=2,
    values=[
        [0.9, 0.2],     # H = yes | O = cold, mild
        [0.1, 0.8]      # H = no  | O = cold, mild
    ],
    evidence=['O'],
    evidence_card=[2],
    state_names={
        'H': ['yes', 'no'],
        'O': ['cold', 'mild']
    }
)

# W | O
cpd_W = TabularCPD(
    variable='W',
    variable_card=2,
    values=[
        [0.1, 0.6],     # W = yes | O = cold, mild
        [0.9, 0.4]      # W = no  | O = cold, mild
    ],
    evidence=['O'],
    evidence_card=[2],
    state_names={
        'W': ['yes', 'no'],
        'O': ['cold', 'mild']
    }
)

# R | H, W
cpd_R = TabularCPD(
    variable='R',
    variable_card=2,
    values=[
        [0.6, 0.9, 0.3, 0.5],   # R = warm | H = yes, yes, no, no, W = yes, no, yes, no
        [0.4, 0.1, 0.7, 0.5]    # R = cool | H = yes, yes, no, no, W = yes, no, yes, no
    ],
    evidence=['H', 'W'],
    evidence_card=[2, 2],
    state_names={
        'R': ['warm', 'cool'],
        'H': ['yes', 'no'],
        'W': ['yes', 'no']
    }
)

# E | H
cpd_E = TabularCPD(
    variable='E',
    variable_card=2,
    values=[
        [0.8, 0.2],     # E = high | H = yes, no
        [0.2, 0.8]      # E = low  | H = yes, no
    ],
    evidence=['H'],
    evidence_card=[2],
    state_names={
        'E': ['high', 'low'],
        'H': ['yes', 'no']
    }
)

# C | R
cpd_C = TabularCPD(
    variable='C',
    variable_card=2,
    values=[
        [0.85, 0.40],   # C = comfortable | R = warm, cool
        [0.15, 0.60]    # C = uncomfortable | R = warm, cool
    ],
    evidence=['R'],
    evidence_card=[2],
    state_names={
        'C': ['comfortable', 'uncomfortable'],
        'R': ['warm', 'cool']
    }
)

model.add_cpds(cpd_O, cpd_H, cpd_W, cpd_R, cpd_E, cpd_C)

assert model.check_model()

print("Nodes:", model.nodes())
print("Edges:", model.edges())

infer = VariableElimination(model)

inf_H = infer.query(
    variables=['H'],
    evidence={'C': 'comfortable'}
)
print(inf_H)

inf_E = infer.query(
    variables=['E'],
    evidence={'C': 'comfortable'}
)
print(inf_E)

map_HW = infer.map_query(
    variables=['H', 'W'],
    evidence={'C': 'comfortable'}
)
print(map_HW)


