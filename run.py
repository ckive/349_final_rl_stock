from mundane import SingleStock, pathify
from stable_baselines import A2C

model_types = {
    "A2C": lambda env: A2C('MlpLstmPolicy', env, verbose=0),
    #"DQN": lambda env:
}

models = ["A2C"]

a = SingleStock(ticker='amd',
                start=10,
                end=300,
                window_size=10,
                models=models,
                total_steps=100,
                model_types=model_types)
