from grader_contracts.pandas_tasks import TitanicInput
from grader_contracts.correlation_tasks import BrainDataInput
from pandas_tasks import analyze_titanic
from correlation_tasks import analyze_brain_correlations

titanic_result = analyze_titanic(TitanicInput("titanic.csv"))
brain_result = analyze_brain_correlations(BrainDataInput("brainsize.txt"))

print(titanic_result)
print(brain_result)