import pickle
import numpy as np
import pandas as pd
import cvxpy as cp
from pathlib import Path
from collections import defaultdict
from itertools import cycle
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from scipy.stats import norm

from Utils.PlotFunctions import histogram, boxplot
from Utils.Config import configurations
dict_config, config = configurations()

from typing import Tuple, Dict, List, Optional



class Training:

    def __init__(self, model:LinearRegression, inputs:pd.DataFrame, outputs:pd.DataFrame, test_size:Optional[float]=0.25, random_state:Optional[int]=None) -> None:

        self.model = model
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            inputs, outputs,
            test_size=test_size,
            random_state=self.random_state
        )

    def train(self) -> LinearRegression:
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred = pd.DataFrame(data=self.y_pred, columns=self.y_test.columns)
        self.model.residues = (self.y_pred.to_numpy() - self.y_test.to_numpy())
        self.fig_histogram = histogram(self.model.residues[:, 0]*3) #FIXME
        plt.close()
        self.fig_boxplot = boxplot(self.model.residues)
        plt.close()
        return self.model
    
    def evaluate_model(self) -> Tuple[float, float]:
        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        return mse, r2



class Detector:

    def __init__(self, model:LinearRegression, inputs:pd.DataFrame, outputs:pd.DataFrame) -> None:

        self.model = model
        self.X = inputs
        self.y = outputs

        self.A = self.model.coef_
        self.B = self.model.intercept_
        self.std = np.std(self.model.residues, axis=0) * 1
        self.residues_list = []
        self.deltaW = defaultdict(float)

        # R2-scores
        self.scores = []

        # Inputs
        self.inputs_ = {
            "updt": [],
            "r2" : [],
            "r2_after": [],
            "inputs": [],
            "optimized": []
        }

        # Outputs
        self.outputs_ = {
            "r2":[],
            "r2_after": [],
            "updt": [],
            "targets": [],
            "outputs": [],
            "optimized": []
        }

    def execute(self) -> pd.DataFrame:
        
        X, y = self.X.reset_index(drop=True), self.y.reset_index(drop=True)
        for t, row in tqdm(X.iterrows(), total=X.shape[0], desc='LinearRegression'):

            x_array = row.values
            original_x = row.values
            y_true = y.iloc[t].values
            x_df = pd.DataFrame(x_array.reshape(1, -1), columns=self.X.columns)
            y_pred = self.model.predict(x_df).flatten()
            original_pred = self.model.predict(x_df).flatten()
            delta_y = y_pred - y_true
            resn = delta_y / self.std
            self.residues_list.append(resn)

            flag = 0
            counter, patience, threshold = 0, 5, 0.90
            coeficiente_determinacao = r2_score(y_true, y_pred)
            optimized_x = x_array
            while max(resn) > 6 and r2_score(y_true, y_pred) <= threshold:
                
                flag = 1
                threshold = 0.99
                residues = dict(zip(self.y.columns, resn))
                outliers = [max(residues, key=residues.get).split('.')[2]]
                updt_nodes = [i for i, x in enumerate(self.X.columns) if x.split('.')[2] in outliers]
                fixed_nodes = [i for i, x in enumerate(self.X.columns) if x.split('.')[2] not in outliers]

                Xvar = cp.Variable((x_array.shape[0], 1))
                W = np.eye(self.y.shape[1])
                objective = cp.Minimize(cp.sum_squares(cp.sqrt(W) @ (self.A @ Xvar + self.B - y_true)))
                constraints = [
                    Xvar[updt_nodes, :] >= x_array[updt_nodes],
                    Xvar[fixed_nodes, :] >= x_array[fixed_nodes] * (1 - 0.05),
                    Xvar[fixed_nodes, :] <= x_array[fixed_nodes] * (1 + 0.05),
                ]
                
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.SCS, max_iters=50)
                if problem.status in ["optimal", "optimal_inaccurate"]:

                    optimized_x = Xvar.value.flatten()
                    x_df_optim = pd.DataFrame(optimized_x.reshape(1, -1), columns=self.X.columns)
                    y_pred = self.model.predict(x_df_optim).flatten()
                    resn = (y_pred - y_true) / self.std

                    delta_x = optimized_x - x_array
                    delta_x = {key: delta_x[i] for i, key in enumerate(row.index) if delta_x[i] > 0.01}
                    x_array = optimized_x.copy()

                    for key, val in delta_x.items():
                        self.deltaW[key] += val  
                else:
                    continue

                counter += 1
                if counter > patience:
                    break
            
            # Store Inputs
            r2 = r2_score(y_true, y_pred)
            self.inputs_["r2"].append(coeficiente_determinacao)
            self.inputs_["r2_after"].append(r2)
            self.inputs_["updt"].append(flag)
            self.inputs_["inputs"].append(original_x)
            self.inputs_["optimized"].append(optimized_x)

            # Store Outputs
            self.outputs_["r2"].append(coeficiente_determinacao)
            self.outputs_["r2_after"].append(r2)
            self.outputs_["updt"].append(flag)
            self.outputs_["targets"].append(y_true)
            self.outputs_["outputs"].append(original_pred)
            self.outputs_["optimized"].append(y_pred)

        # Convert to DataFrame
        self.inputs_ = pd.DataFrame(self.inputs_)
        self.outputs_ = pd.DataFrame(self.outputs_)
            
        deltaWb_acc: Dict[str, float] = defaultdict(float)
        for key, val in self.deltaW.items():
            deltaWb_acc[key.split('.')[2]] += val * 0.25
        deltaWb_acc = dict(sorted(deltaWb_acc.items(), key=lambda item: item[1], reverse=True))
        deltaWb_acc = {key: round(val, 2) for key, val in deltaWb_acc.items()}

        if len(deltaWb_acc)>0:
            std_val = np.std(list(deltaWb_acc.values()))
            q3 = np.percentile(list(deltaWb_acc.values()), 95)
            filtered_deltaW = {
                k: v for k, v in dict(sorted(deltaWb_acc.items(), key=lambda item: item[1], reverse=True)).items()
                if v >= q3
            }
        else:
            filtered_deltaW = {}

        output = pd.DataFrame(data={"bus_id": list(filtered_deltaW.keys()), "deltaW": list(filtered_deltaW.values())})\
                    .sort_values(by="deltaW", ascending=False)\
                    .reset_index(drop=True)
        return output


