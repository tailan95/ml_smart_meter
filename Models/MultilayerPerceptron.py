import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Sklearn
from sklearn.metrics import r2_score

# MLflow
import mlflow
import mlflow.pytorch

# Utils
from Utils.DataLoader import DataReader
from Utils.PlotFunctions import histogram, boxplot
from Utils.DataLoader import training_dataloaders

# Configurações
from Utils.Config import configurations
dict_config, config = configurations()

# Typing
from typing import Any, Dict, List, Optional

# Define activation functions
FUNCTIONS = {
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(negative_slope=0.01),
    'Softplus': nn.Softplus(),
    'Tanh': nn.Tanh(),
    'Sigmoid': nn.Sigmoid(),
    'Softmax': nn.Softmax(),
    'ELU': nn.ELU(alpha=1.0),
    'SiLU': nn.SiLU(),
    'SELU': nn.SELU(),
    'GELU': nn.GELU(),
    }

# Dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Função de perda
LOSSES = {
    "MSE": nn.MSELoss(),
    "L1Loss": nn.L1Loss(),
    "SmoothL1Loss": nn.SmoothL1Loss(),
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
}

# Neural network
class MLP(nn.Module):
    functions = FUNCTIONS
    
    """
    Multilayer perceptron neural network structure.
    """
    def __init__(
            self, 
            input_size: int,
            hidden_layers: List[int],
            output_size: int,
            activation_function:Optional[str]='ELU',
            **kwargs,
        ) -> None:
        self.activation_function = activation_function
        self.dropout = kwargs.get('dropout', None)
        self.batch_norm = kwargs.get('batch_norm', False)
        self.he_init = kwargs.get('he_init', False)
        self.hidden_layers = hidden_layers

        # Initialize class
        super(MLP, self).__init__()
        
        # Attributes
        self.errors = None
        
        # Build neural structure
        layers = []
        
        # First hidden layer (from input)
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        if self.batch_norm:
            layers.append(nn.BatchNorm1d(hidden_layers[0]))
        layers.append(self.functions[self.activation_function])
        if self.dropout is not None:
            layers.append(nn.Dropout(self.dropout))

        # Additional hidden layers (if any)
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_layers[i]))
            layers.append(self.functions[self.activation_function])
            if self.dropout is not None:
                layers.append(nn.Dropout(self.dropout))

        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        self.network = nn.Sequential(*layers)

        # He Initialization for all Linear layers
        if self.he_init:
            for layer in self.network:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.zeros_(layer.bias)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Send data through the neural network.
        """
        return self.network(data) 


# Treinamento
class Training:

    def __init__(self, model: nn.Module, inputs:pd.DataFrame, outputs:pd.DataFrame, random_state=None) -> None:

        # Params
        self.loss_function = config.mlp.loss_function
        self.learning_rate = config.mlp.learning_rate
        self.weight_decay = config.mlp.weight_decay
        self.num_epochs = config.mlp.num_epochs
        self.early_stop = config.mlp.early_stop
        self.random_state = random_state
        
        # model
        self.model = model.to(DEVICE)

        # Função de perda
        self.criterion = LOSSES[self.loss_function]

        # Otimizador
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.training_dataloader, self.test_dataloader, self.validation_dataloader = training_dataloaders(
            inputs=inputs,
            outputs=outputs,
            random_state = random_state,
        )

        self.fig_boxplot = None
        self.fig_hist = None

    def _train(self, dataloader: DataLoader) -> float:
        self.model.train()
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def _validate(self, dataloader: DataLoader) -> torch.Tensor:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(dataloader)
    
    def _test(self, dataloader: DataLoader) -> List[float]:
        err = []
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = self.model(inputs)
                errors = outputs - targets
                err.extend(errors.cpu().numpy())
        return torch.tensor(np.asarray(err))

    def train(self, use_mlflow:Optional[bool]=False, log_params:Optional[Dict[str, Any]]={}):

        # Looping de treinamento
        best_val_loss = float('inf')
        best_model_state = None
        current_patience = 0
        progress_bar = tqdm(range(self.num_epochs), desc='Training MLP model')
        if use_mlflow:

            # Set Experiment
            mlflow.set_experiment("/Users/tailan_@hotmail.com/experiments/model_free_ntl_detection")

            # Start running
            run = mlflow.start_run()

            # Log params
            inputs, targets = next(iter(test_dataloader))
            params = dict(
                model = "MLP",
                input_layer = inputs.shape[-1],
                hidden_layers = self.model.hidden_layers,
                output_layer = targets.shape[-1],
                activation_function_hidden_layer = self.model.activation_function,
                num_epochs = self.mlp.num_epochs,
                early_stop = self.mlp.early_stop,
            )
            if log_params:
                params.update(log_params)
            mlflow.log_params(params)

        # Training loop
        for epoch in progress_bar:

            # Iteração de treinamento
            loss = self._train(self.training_dataloader)
            val_loss = self._validate(self.validation_dataloader)
            progress_bar.set_description(f'Training MLP model (val_loss = {val_loss:.6f})')

            # Early stopping
            if self.early_stop:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = deepcopy(self.model.state_dict())
                    current_patience = 0
                else:
                    current_patience += 1
                    if self.early_stop and current_patience >= self.early_stop:
                        self.model.load_state_dict(best_model_state)
                        progress_bar.set_description(f'Training MLP model (val_loss = {val_loss:.6f})')
                        progress_bar.update(self.num_epochs)  # Refresh display
                        break
            else:
                continue
                
            # Log epoch
            if use_mlflow:
                mlflow.log_metric("val_loss", np.round(val_loss, 3), step=epoch)

        # Testa modelo
        err = self._test(self.test_dataloader)
        self.model.errors = err

        # Set params to MLFlow
        self.fig_histogram = histogram(self.model.errors[:, 1].detach().numpy())
        plt.close(self.fig_histogram)
        self.fig_boxplot = boxplot(self.model.errors.detach().numpy())
        plt.close(self.fig_boxplot)
        if use_mlflow: 

            # Log model     
            inputs, targets = next(iter(self.test_dataloader))
            mlflow.pytorch.log_model(
                self.model,
                artifact_path="model",
                input_example = torch.tensor(inputs[0]).unsqueeze(0).cpu().numpy() #TODO o imputs está dentro do dataloader
            )

            # Histogram artifact
            for frmt in ["png", "svg"]:
                path = f"/Volumes/le41/bronze/files/temp//histogram.{frmt}"
                self.fig_histogram.savefig(path, dpi=800)
                mlflow.log_artifact(path)
            

            # Boxplot artifact
            for frmt in ["png", "svg"]:
                path = f"/Volumes/le41/bronze/files/temp/histogram.{frmt}"
                self.fig_boxplot.savefig(path, dpi=800)
                mlflow.log_artifact(path)
            

            # Configs
            artifact_path = f"/Volumes/le41/bronze/files/temp/configs.json"
            with open(artifact_path, "w") as f:
                json.dump(dict_config, f, indent=4)
            mlflow.log_artifact(artifact_path)
            os.remove(artifact_path)
                
            # Run id
            print(f"Modelo salvo em: {run.info.run_id}")

            # Stop mlflow
            mlflow.end_run()
        return self.model


# Detecção da PNT
class NTLDetection:

    def __init__(self, model:nn.Module, inputs:pd.DataFrame, outputs:pd.DataFrame) -> None:

        self.model = model
        self.inputs = inputs
        self.outputs = outputs
        
        self.deltaW = defaultdict(float)
        self.dataloader = DataLoader(DataReader(inputs, outputs), batch_size=1, shuffle=False)
        self.residues = []

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

    def execute_(self, rn_threshold:Optional[float]=6, r2_threshold:Optional[float]=0.90):
        """
        deprecated
        """
        progress_bar = tqdm(
            enumerate(self.dataloader), 
            desc="MLP Predictor", 
            unit="step", 
            total=len(self.dataloader)
        )
        for t, (x_input, y_target) in progress_bar:
            
            # Original input
            x_original = x_input.clone()

            # True output
            y_true = y_target.detach().cpu().numpy().flatten()

            # Predicted outputs
            with torch.no_grad():
                y_pred = self.model(x_input).detach().cpu().numpy().flatten()
            
            # Residues
            delta_y = y_pred - y_true
            resn = delta_y/self.model.errors.std(axis=0).detach().cpu().numpy()
            self.residues.append(resn)
            self.scores.append(np.round(r2_score(y_true, y_pred), 2))
            
            # Clone da entrada
            x_val = x_input.clone()
            best_x = x_val.clone().detach()
            flag = 0 # indica atualização

            # Check inconsistency
            residuo_normalizado = max(resn)
            coeficiente_determinacao = np.round(r2_score(y_true, y_pred), 2)
            if max(resn) > rn_threshold and self.scores[-1] < r2_threshold:
                outer_counter = 0
                last_updated = []
                flag = 1 # indica atualização
                while self.scores[-1] < 0.98:
                    
                    # Check residues
                    resid = dict(zip(self.outputs.columns, resn))
                    resid = {k: v for k, v in resid.items() if k.split('.')[2] not in last_updated}
                    if not self.residues:
                        break
                    
                    # Check outliers
                    outliers = [max(resid, key=resid.get).split('.')[2]]
                    outliers_nodes = {
                        k: v for k, v in resid.items()
                        if k.split('.')[-2] in outliers # and abs(v) > 2
                    }
                    if not outliers_nodes:
                        last_updated = outliers
                        break

                    # Get nodes to update
                    sorted_nodes = sorted([node_id.split('.')[-1] for node_id in outliers_nodes.keys()])
                    updt_indices = [
                        i for i, col in enumerate(self.inputs.columns)
                        if ('kW' in col or 'kvar' in col)
                        and col.split('.')[2] in outliers
                        and col.split('.')[-1] in sorted_nodes
                            ]
                    if not updt_indices:
                        last_updated = outliers
                        break

                    # Get mask
                    mask_np = np.ones(len(self.inputs.columns), dtype=np.float32)
                    mask_np[updt_indices] = 0.0
                    mask = torch.tensor(mask_np, device=x_val.device).unsqueeze(0)

                    # Optimize inputs
                    x_opt = x_val.clone().detach().requires_grad_(True)
                    for param in self.model.parameters():
                        param.requires_grad = False
                    optimizer = torch.optim.Adam([x_opt], lr=1e-2)
                    criterion = torch.nn.MSELoss()
                    lambda_deviation = 1e3
                    max_iters = 100
                    counter = 0
                    best_r2 = self.scores[-1]
                    best_x = x_val.clone().detach()
                    r2_scores, x_values = [best_r2], [best_x]
                    stop = 0
                    for _ in range(max_iters):
                        optimizer.zero_grad()
                        y_pred_opt = self.model(x_opt)
                        loss_pred = criterion(y_pred_opt, y_target)
                        loss_dev = torch.sum(mask*(x_opt - x_val.detach())**2)
                        loss = loss_pred + lambda_deviation * loss_dev
                        loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                            y_pred_np = y_pred_opt.detach().cpu().numpy().flatten()
                        r2 = r2_score(y_true, y_pred_np)
                        if np.round(r2, 4) > np.round(max(r2_scores), 4):
                            r2_scores.append(np.round(r2, 4))
                            best_x = x_opt.clone().detach()
                            x_values.append(best_x)
                            counter = 0
                        else:
                            counter += 1
                        if counter >= 20:
                            idx = r2_scores.index(np.round(max(r2_scores), 4))
                            best_x = x_values[idx]
                            stop+=1
                            if stop>=3:
                                break
                            else:
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] /= 10

                    # Check residues
                    delta_y = self.model(best_x)[0].detach().numpy() - y_true
                    resn = delta_y/self.model.errors.std(axis=0).detach().cpu().numpy()
                    self.scores[-1] = max(r2_scores)
                    last_updated = outliers
                    x_val = best_x.clone().detach()
                    outer_counter += 1
                    if outer_counter >= 10:
                        break
                
                # Atualização de deltaW 
                delta_x = best_x.flatten().cpu().numpy() - x_input.detach().cpu().numpy().flatten()
                delta_x_dict = dict(zip(self.inputs.columns, delta_x))
                for key, val in delta_x_dict.items():
                    if 'kW' in key.split('.'):
                        self.deltaW[key] += val
            
            # Store Inputs
            r2 = r2_score(y_target.detach().cpu().numpy().flatten(), self.model(best_x)[0].detach().numpy())
            self.inputs_["r2"].append(coeficiente_determinacao)
            self.inputs_["r2_after"].append(r2)
            self.inputs_["updt"].append(flag)
            self.inputs_["inputs"].append(x_input.detach().cpu().numpy().flatten())
            self.inputs_["optimized"].append(best_x.detach().cpu().numpy().flatten())

            # Store Outputs
            self.outputs_["r2"].append(coeficiente_determinacao)
            self.outputs_["r2_after"].append(r2)
            self.outputs_["updt"].append(flag)
            self.outputs_["targets"].append(y_target.detach().cpu().numpy().flatten())
            self.outputs_["outputs"].append(self.model(x_input)[0].detach().numpy())
            self.outputs_["optimized"].append(self.model(best_x)[0].detach().numpy())

        # Convert to DataFrame
        self.inputs_ = pd.DataFrame(self.inputs_)
        self.outputs_ = pd.DataFrame(self.outputs_)

        # Aggregate deltaW by bus id (3rd element in split key)
        deltaWb_acc = defaultdict(float)
        for key, val in self.deltaW.items():
            deltaWb_acc[key.split('.')[2]] += val
        return pd.DataFrame(data={"bus_id": list(deltaWb_acc.keys()), "deltaW": list(deltaWb_acc.values())})\
            .sort_values(by="deltaW", ascending=False)\
            .reset_index(drop=True)
    
    def execute(self, rn_threshold:Optional[float]=6, r2_threshold:Optional[float]=0.90):
        """
        deprecated
        """
        progress_bar = tqdm(
            enumerate(self.dataloader), 
            desc="MLP Predictor", 
            unit="step", 
            total=len(self.dataloader)
        )
        for t, (x_input, y_target) in progress_bar:
            
            # Original input
            x_original = x_input.clone()

            # True output
            y_true = y_target.detach().cpu().numpy().flatten()

            # Predicted outputs
            with torch.no_grad():
                y_pred = self.model(x_input).detach().cpu().numpy().flatten()
            
            # Residues
            delta_y = y_pred - y_true
            resn = delta_y/self.model.errors.std(axis=0).detach().cpu().numpy()
            self.residues.append(resn)
            self.scores.append(np.round(r2_score(y_true, y_pred), 2))
            
            # Clone da entrada
            x_val = x_input.clone()
            best_x = x_val.clone().detach()
            flag = 0 # indica atualização

            # Check inconsistency
            residuo_normalizado = max(resn)
            coeficiente_determinacao = np.round(r2_score(y_true, y_pred), 2)
            if max(resn) > rn_threshold and self.scores[-1] < r2_threshold:
                outer_counter = 0
                last_updated = []
                flag = 1 # indica atualização
                while self.scores[-1] < 0.99:
                    
                    # Check residues
                    resid = dict(zip(self.outputs.columns, resn))
                    resid = {k: v for k, v in resid.items() if k.split('.')[2] not in last_updated}
                    if not self.residues:
                        break
                    
                    # Check outliers
                    outliers = [max(resid, key=resid.get).split('.')[2]]
                    outliers_nodes = {
                        k: v for k, v in resid.items()
                        if k.split('.')[-2] in outliers # and abs(v) > 2
                    }
                    if not outliers_nodes:
                        last_updated = outliers
                        break

                    # Get nodes to update
                    sorted_nodes = sorted([node_id.split('.')[-1] for node_id in outliers_nodes.keys()])
                    updt_indices = [
                        i for i, col in enumerate(self.inputs.columns)
                        if ('kW' in col or 'kvar' in col)
                        and col.split('.')[2] in outliers
                        and col.split('.')[-1] in sorted_nodes
                            ]
                    
                    if not updt_indices:
                        last_updated = outliers
                        break

                    # Get mask
                    mask_np = np.ones(len(self.inputs.columns), dtype=np.float32)
                    mask_np[updt_indices] = 0.0
                    mask = torch.tensor(mask_np, device=x_val.device).unsqueeze(0)

                    # Optimize inputs
                    x_opt = x_val.clone().detach().requires_grad_(True)
                    for param in self.model.parameters():
                        param.requires_grad = False
                    optimizer = torch.optim.Adam([x_opt], lr=1e-2)
                    criterion = torch.nn.MSELoss()
                    lambda_deviation = 1e2
                    max_iters = 100
                    counter = 0
                    best_r2 = self.scores[-1]
                    best_x = x_val.clone().detach()
                    r2_scores, x_values = [best_r2], [best_x]
                    stop = 0
                    for _ in range(max_iters):
                        optimizer.zero_grad()
                        y_pred_opt = self.model(x_opt)
                        loss_pred = criterion(y_pred_opt, y_target)
                        loss_dev = torch.sum(mask*(x_opt - x_val.detach())**2)
                        loss = loss_pred + lambda_deviation * loss_dev
                        loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                            x_opt.data = torch.clamp(x_opt.data, min=x_val)
                            y_pred_np = self.model(x_opt).detach().cpu().numpy().flatten()
                        r2 = r2_score(y_true, y_pred_np)
                        if np.round(r2, 4) > np.round(max(r2_scores), 4):
                            r2_scores.append(np.round(r2, 4))
                            best_x = x_opt.clone().detach()
                            x_values.append(best_x)
                            counter = 0
                        else:
                            counter += 1
                        if counter >= 20:
                            idx = r2_scores.index(np.round(max(r2_scores), 4))
                            best_x = x_values[idx]
                    
                    # Check residues
                    delta_y = self.model(best_x)[0].detach().numpy() - y_true
                    resn = delta_y/self.model.errors.std(axis=0).detach().cpu().numpy()
                    self.scores[-1] = max(r2_scores)
                    last_updated = outliers
                    x_val = best_x.clone().detach()
                    outer_counter += 1
                    if outer_counter >= 5:
                        break
                
                # Atualização de deltaW 
                delta_x = best_x.flatten().cpu().numpy() - x_input.detach().cpu().numpy().flatten()
                delta_x_dict = dict(zip(self.inputs.columns, delta_x))
                for key, val in delta_x_dict.items():
                    if 'kW' in key.split('.'):
                        self.deltaW[key] += val

            # Store Inputs
            r2 = r2_score(y_target.detach().cpu().numpy().flatten(), self.model(best_x)[0].detach().numpy())
            self.inputs_["r2"].append(coeficiente_determinacao)
            self.inputs_["r2_after"].append(r2)
            self.inputs_["updt"].append(flag)
            self.inputs_["inputs"].append(x_input.detach().cpu().numpy().flatten())
            self.inputs_["optimized"].append(best_x.detach().cpu().numpy().flatten())

            # Store Outputs
            self.outputs_["r2"].append(coeficiente_determinacao)
            self.outputs_["r2_after"].append(r2)
            self.outputs_["updt"].append(flag)
            self.outputs_["targets"].append(y_target.detach().cpu().numpy().flatten())
            self.outputs_["outputs"].append(self.model(x_input)[0].detach().numpy())
            self.outputs_["optimized"].append(self.model(best_x)[0].detach().numpy())

        # Convert to DataFrame
        self.inputs_ = pd.DataFrame(self.inputs_)
        self.outputs_ = pd.DataFrame(self.outputs_)

        # Aggregate deltaW by bus id (3rd element in split key)
        deltaWb_acc = defaultdict(float)
        for key, val in self.deltaW.items():
            deltaWb_acc[key.split('.')[2]] += val
        
        # Post processing
        if len(deltaWb_acc)>0:
            std_val = np.std(list(deltaWb_acc.values()))
            q3 = np.percentile(list(deltaWb_acc.values()), 95)
            filtered_deltaW = {
                k: v for k, v in dict(sorted(deltaWb_acc.items(), key=lambda item: item[1], reverse=True)).items()
                if v >= q3
            }
        else:
            filtered_deltaW = {}

        return pd.DataFrame(data={"bus_id": list(filtered_deltaW.keys()), "deltaW": list(filtered_deltaW.values())})\
            .sort_values(by="deltaW", ascending=False)\
            .reset_index(drop=True)




