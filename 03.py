from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = BayesianNetwork([('Exercise', 'HeartDisease'), ('Diet', 'HeartDisease')])

cpd_exercise = TabularCPD(variable='Exercise', variable_card=2, values=[[0.7], [0.3]])
cpd_diet = TabularCPD(variable='Diet', variable_card=2, values=[[0.8], [0.2]])

cpd_heart = TabularCPD(variable='HeartDisease', variable_card=2,
                       values=[[0.9, 0.7, 0.6, 0.3],
                               [0.1, 0.3, 0.4, 0.7]],
                       evidence=['Exercise', 'Diet'],
                       evidence_card=[2, 2])

model.add_cpds(cpd_exercise, cpd_diet, cpd_heart)

inference = VariableElimination(model)
result = inference.query(variables=['HeartDisease'], evidence={'Exercise': 1, 'Diet': 0})

print(result)
