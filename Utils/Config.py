import toml
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import os
import sys

main_folder = Path(os.getcwd()).parent
sys.path.insert(1, str(main_folder))


@dataclass
class GeneralConfig:
    dss_path: str


@dataclass
class LoadFlowConfig:
    sub: str
    feeder: str
    uni_tr_mt: Optional[str] = None
    vsource: Optional[float] = 1.035
    stepsize: Optional[float] = 0.25
    kva_base: Optional[float] = None
    power_factor: Optional[List[float]] = field(default_factory=lambda: [0.85, 0.95])

    @property
    def npts(self):
        return int(24 * 60 / self.stepsize)


@dataclass
class IrregConfig:
    num_irregs: Optional[int] = None
    kw_range: Optional[List[float]] = None
    loadshape: Optional[str] = "fixed"
    theft_on_training: Optional[bool] = False
    power_factor: Optional[List[float]] = field(default_factory=lambda: [0.85, 1.00])
    ntl_period: Optional[List[int]] = field(default_factory=lambda: [17, 21])


@dataclass
class SimulationConfig:
    trafo_meas: Optional[bool] = False
    se_meas: Optional[bool] = False
    noise: Optional[Dict[str, float]] = field(default_factory=dict)
    num_days: Optional[int] = 31
    days_for_training: Optional[int] = 14
    minimum_ntl_days: Optional[int] = 10
    public_light: Optional[bool] = True
    input_data: Optional[List[str]] = field(default_factory=lambda: ['active_power', 'reactive_power'])
    batch_sizes: Optional[Dict[str, int]] = None
    shuffle: Optional[Dict[str, bool]] = None


@dataclass
class MLP:
    hidden_layers: List[int]
    activation_function: Optional[str] = "ELU"
    num_epochs: Optional[int] = 500
    dropout: Optional[float] = None
    early_stop: Optional[int] = 10
    weight_decay: Optional[float] = 0.00
    batch_norm: Optional[bool] = False
    he_init: Optional[bool] = False
    learning_rate: Optional[float] = 1e-3
    loss_function: Optional[str] = "MSE"
    split_sizes: Optional[Dict[str, float]] = None
    shuffle: Optional[Dict[str, float]] = None
    batch_sizes: Optional[Dict[str, int]] = None


@dataclass
class Settings:
    general: GeneralConfig
    loadflow: LoadFlowConfig
    irregularity: IrregConfig
    simulation: SimulationConfig
    mlp: MLP


def ReadToml(file: Optional[str] = "input") -> Dict:
    path = Path(os.getcwd()).parent / 'Input' / f'{file}.toml'
    with open(path, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    return config


def configurations() -> Tuple[Dict, Settings]:
    text = ReadToml()

    Config = Settings(
        general=GeneralConfig(**text['general']),
        loadflow=LoadFlowConfig(**text['loadflow']),
        irregularity=IrregConfig(**text['irregs']),
        simulation=SimulationConfig(**text['simulation']),
    mlp=MLP(**text['MLP']),
    )
    return text, Config
