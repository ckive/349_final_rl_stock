import numpy as np
import yfinance as yf
import pandas as pd
from typing import List
import gym
import gym_anytrading
import matplotlib.pyplot as plt
import stable_baselines as sb
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
import quantstats as qs
import pyfolio as pf

model_types = {
    "A2C": lambda env: A2C('MlpLstmPolicy', env, verbose=0),
    #"DQN": lambda env:
}


def pathify(stock, model, type):
    #generate
    fname = str(stock) + "_" + str(model) + "_" + str(type)
    return fname


class SingleStock:
    def __init__(self, ticker: str, start: int, end: int, window_size: int,
                 models: List, total_steps, model_types):
        #get data
        self.ticker = ticker
        self.data = yf.download(self.ticker, "2017-01-01", "2019-12-31")

        #build env
        self.env = self._build_env(window_size, start, end)

        #train env
        self.models = {}
        #for doing multiple models + transactions,  stats paths
        for m in models:
            self.models[m] = {
                "model": None,
                "transactions": None,
                "analysis": None
            }
            self.models[m]["model"] = model_types[m](
                self.env
            )  #<-- () makes it an instance rather than function object
            self.models[m]["model"].learn(total_timesteps=total_steps)

            #save imgs + pathify
            self.models[m]["transactions"] = pathify(self.ticker, m,
                                                     "transactions")
            self.models[m]["analysis"] = pathify(self.ticker, m, "analysis")

            if m not in model_types:
                break

        #testing env
        for m_name in self.models:
            model = self.models[m_name]
            #blank slate
            observation = self.env.reset()
            #can do this simultaneously for different models? should be able to...
            while True:
                observation = observation[np.newaxis, ...]
                action, _states = self.models[m_name]["model"].predict(
                    observation)
                observation, reward, done, info = self.env.step(action)
                if done:
                    #1st plot buys&sells
                    self.env.save_rendering(self.models[m]["transactions"])

                    #2: analysis --> get returns --> qs.report
                    qs.extend_pandas()  # <-- redundant?/move to top?
                    net_worth = pd.Series(self.env.history['total_profit'],
                                          index=self.data.index[start + 1:end])
                    returns = net_worth.pct_change().iloc[1:]

                    qs.reports.html(returns, output=self.models[m]
                                    ["analysis"])  #dunno behavior here...

    def _build_env(self, window_size, start_idx, end_idx):
        env_maker = lambda: gym.make('stocks-v0',
                                     df=self.data,
                                     frame_bound=(start_idx, end_idx),
                                     window_size=window_size)
        return DummyVecEnv([env_maker])

    def get_transactions(self, model):
        self.models[model]["transactions"]

    def get_analysis(self, model):
        with open(self.models[model]["analysis"]):
            pass
