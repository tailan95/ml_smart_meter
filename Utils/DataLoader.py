import torch
import pandas as pd
from typing import Union

# Scikit-learn
from sklearn.model_selection import train_test_split

# Pytorch
from torch.utils.data import DataLoader

# Configurações
from Utils.Config import configurations
_, config = configurations()

# Typing
from typing import Optional, Union

class DataReader:

    def __init__(self, inputs:pd.DataFrame, outputs:pd.DataFrame) -> None:
        self.inputs = inputs.reset_index(drop=True)
        self.outputs = outputs.reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, i:int) -> Union[torch.Tensor, torch.Tensor]:
        inputs = self.inputs.iloc[i].to_numpy()
        targets = self.outputs.iloc[i].to_numpy()
        return torch.Tensor(inputs), torch.Tensor(targets)



def training_dataloaders(
    inputs:pd.DataFrame, 
    outputs:pd.DataFrame,
    random_state:Optional[int] = None, 
    ) -> Union[DataLoader, DataLoader, DataLoader]:

    trainig_inputs, test_inputs, training_outputs, test_outputs = train_test_split(
        inputs, outputs,
        test_size=config.mlp.split_sizes.get("test_size"),
        random_state=random_state
    )
    test_inputs, val_inputs, test_outputs, val_outputs = train_test_split(
        test_inputs, test_outputs,
        test_size=config.mlp.split_sizes.get("val_size"),
        random_state=random_state
    )
    training_dataloader = DataLoader(
        DataReader(trainig_inputs, training_outputs), 
        batch_size=config.mlp.batch_sizes.get("train"), 
        shuffle=config.mlp.shuffle.get("train")
    )
    test_dataloader = DataLoader(
        DataReader(test_inputs, test_outputs), 
        batch_size=config.mlp.batch_sizes.get("test"), 
        shuffle=config.mlp.shuffle.get("test")
    )
    validation_dataloader = DataLoader(
        DataReader(val_inputs, val_outputs), 
        batch_size=config.mlp.batch_sizes.get("val"), 
        shuffle=config.mlp.shuffle.get("test")
    )
    return training_dataloader, test_dataloader, validation_dataloader