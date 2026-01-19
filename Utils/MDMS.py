import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict

# Matplotlib
import matplotlib.pyplot as plt

# Gzip
import Utils.GZipReader as gzipreader

# Altdss
from altdss import altdss, enums

# Configurações
from Utils.Config import configurations
_, config = configurations()

# Datetime
from datetime import datetime
from zoneinfo import ZoneInfo

# Typing
from typing import Any, Dict, List, Optional, Tuple

# Spark Session
from pyspark.sql import SparkSession
spark = SparkSession.builder\
    .appName("DataQuality")\
    .getOrCreate()

# DBUTILS
from pyspark.dbutils import DBUtils
dbutils = DBUtils(spark)

# Resolve path
dss_path = "/Volumes/le41/bronze/files/models/low-voltage-systems/202012/63/dss.zip"
altdss.ZIP.Open(str(Path(dss_path).resolve()))

# ZipFile
from zipfile import ZipFile

def Substations() -> List[str]:
     with ZipFile(dss_path, "r") as dss_zip:
        return tuple(x.split("/")[0] for x in  dss_zip.namelist() if x.count("/") == 1)
    
def Feeders(sub:str) -> List[str]:
    with ZipFile(dss_path) as dss_zip:
        return tuple([x.split("/")[-2]  for x in  dss_zip.namelist() if ((x.count("/") == 2) and (x.startswith(sub) and (x.endswith("BusVoltageBases.dss"))))])

def LVSystems(sub:str, feeder:str) -> List[str]:
    with ZipFile(dss_path) as dss_zip:
        return tuple([x.split("/")[-2] for x in dss_zip.namelist() if ((x.count("/") == 3) and (x.startswith(sub) and feeder in x.split("/") and (x.endswith("BusVoltageBases.dss"))))])


# Get loadshapes
ldshapes = gzipreader.acquire()


class MDMS:

    ldshapes = ldshapes
    altdss = altdss

    def __init__(self, sub:str, feeder:str, uni_tr_mt:Optional[str]=None, random_state:Optional[int]=None):
        
        # Params
        self.random_state = random_state
        self.irregs = pd.DataFrame()
        self.meas = pd.DataFrame()
        self.inputs, self.outputs = pd.DataFrame(), pd.DataFrame()
        self.bus_distances = {}
        self.training = {}
        self.inference = {}
        self.sub=sub
        self.feeder=feeder
        self.uni_tr_mt=uni_tr_mt
        
        # Graph
        self.nxgraph = None
        self._compile()

        # Public light
        if config.simulation.public_light:
            for x in self.altdss.Load:
                if x.Class%10 == 2:
                    x.kW=0.0
                
    def _compile(self) -> altdss:
        self.altdss.ClearAll()
        if self.uni_tr_mt is None: 
            path = Path(f"{self.sub}/{self.feeder}/Master_aneel.dss")
        else: 
            path = Path(f"{self.sub}/{self.feeder}/{self.uni_tr_mt}/Master.dss")
        self.altdss.ZIP.Redirect(str(path))
        return self.altdss
    
    def _set_solution_mode(self) -> altdss:
        self.altdss.Solution.Mode = enums.SolutionMode.Daily
        self.altdss.Solution.StepsizeHr = config.loadflow.stepsize
        self.altdss.Solution.Tolerance = 1e-6
        self.altdss.Solution.Number = int(24/config.loadflow.stepsize)
        return self.altdss

    def _set_params(self, **kwargs) -> altdss:
        if "vsource" in kwargs:
            self.altdss.Vsource.pu = kwargs["vsource"]
        self.altdss.Load.Model = enums.LoadModel.ConstantPQ
        return self.altdss

    def _update_loadshapes(self) -> altdss:
        np.random.seed(self.random_state)
        customers = tuple([x.Name for x in self.altdss.Load if x.Class%10 == 1])
        houses = np.random.choice(list(self.ldshapes.keys()), len(customers), replace=True)
        mapping = dict(zip(customers, houses))
        loadshapes = {}
        for customer, house in mapping.items():
            daily_profile = []
            for i in range(config.simulation.num_days):
                daily_profile += self.ldshapes[house]['curves'].loc[f'dia_{i}'].tolist()
            curve = np.array(daily_profile)
            loadshapes[customer] = curve / np.mean(curve)
        for load in tuple([x for x in self.altdss.Load if x.Class%10 == 1]):
            load.Daily = self.altdss.LoadShape.new(
                name = load.Name.split('_')[-1],
                NPts = loadshapes[load.Name].__len__(),
                MInterval = 15,
                PMult = loadshapes[load.Name],
                Hour = np.arange(0, 24*len(loadshapes[load.Name])/int(24/config.loadflow.stepsize), config.loadflow.stepsize)
            )
        return self.altdss
    
    def _install_monitors(self) -> altdss:
        objects = ()
        if config.simulation.se_meas:
            objects += tuple(self.altdss.Vsource())
        if config.simulation.trafo_meas:
            objects += tuple(self.altdss.Transformer())
        objects += tuple([x for x in self.altdss.Load if x.Class%10 == 1])
        for obj in objects:
            namelist = obj.Name.split('_')
            namelist = list(filter(lambda x: x != '', namelist))
            self.altdss.Monitor.new(
                name = f"{namelist[-1]}_v",
                Element = obj,
                Terminal = obj.NumTerminals(),
                Mode = 0,
                PPolar = 1,
            )
            self.altdss.Monitor.new(
                name = f"{namelist[-1]}_pq",
                Element = obj,
                Terminal = obj.NumTerminals(),
                Mode = 1,
                PPolar = 0,
            )
        return self.altdss

    def _create_ntl_loadshape(
            self, 
            curve:Optional[str]="fixed", 
            period:Optional[List[str]]=[0, 24],
        ) -> np.ndarray:
        step = int(1/config.loadflow.stepsize)
        npts = step*24
        if (curve.lower() == "synthetic") or (curve.lower() == "lvns"):
            house_id = np.random.choice(list(self.ldshapes.keys()))
            df = self.ldshapes[house_id]["curves"]
            daily_list = []
            for day in range(config.simulation.num_days):
                daily = np.array(df.loc[f"dia_{day}"].tolist())
                daily /= max(daily)
                daily[:period[0]*step] = 0
                daily[period[1]*step:] = 0
                daily_list.extend(daily)
        elif curve.lower() == "fixed":
            base = np.ones(npts)
            base[:period[0]*step] = 0
            base[period[1]*step:] = 0
            daily_list = list(base) * config.simulation.num_days
        else:
            raise ValueError("Invalid NTL curve type.")
        return np.array(daily_list) / max(daily_list)
    
    def _insert_ntl(self, customers: Dict[str,List[Any]]) -> pd.DataFrame:
        customers_df = pd.DataFrame(customers)
        customers.update({
            "load_id": [],
            "bus_id": [],
            "num_phases": [],
            "kwh_measured": [],
            "kwh_theft": [],
            "theft_kw": [],
            "measured_kw":[],
        })
        for _, row in customers_df.iterrows():

            # Load obj
            load = self.altdss.Load[row.get("name")]

            # Days without theft
            null_days = 0 if config.irregularity.theft_on_training else np.random.choice([config.simulation.days_for_training+1, config.simulation.num_days - config.simulation.minimum_ntl_days])
            
            # Pmult
            pmult = self._create_ntl_loadshape(curve = row.get("shape_type"), period = row.get("working_period"))
            pmult[:null_days*int(24/config.loadflow.stepsize)] = 0
            
            # Daily loadshape
            daily = self.altdss.LoadShape.new(
                name = f"ntl_{load.Name.split('_')[-1]}",
                NPts = pmult.__len__(),
                MInterval = 15,
                PMult = pmult,
                Hour = np.arange(0, 24*pmult.__len__()/96, 0.25)
            )

            # Create new load
            ntl = self.altdss.Load.new(
                name=f"ntl_{load.Name.split('_')[-1]}",
                Bus1=load.Bus1,
                Phases=load.NumPhases(),
                kV=load.kV,
                kW=row.get("kw_by_phase")*load.NumPhases(),
                PF=row.get("pf"),
                Conn=load.Conn,
                Model=enums.LoadModel.ConstantPQ,
                Daily=daily,
                Status=enums.LoadStatus.Exempt,
                Class=999
                )
            
            # Measured
            measured_kw = np.array(load.Daily.PMult) * load.kW
            kwh_measured = np.round(np.trapz(measured_kw)*config.loadflow.stepsize, 2)

            # Energy theft
            kwh = np.trapz(pmult, dx=config.loadflow.stepsize)*row.get("kw_by_phase")*load.NumPhases()
            
            # Update info
            customers["load_id"].append(f"ntl-{load.Name.split('_')[-1]}")
            customers["bus_id"].append(load.Bus1.split(".")[0].split('_')[-1])
            customers["num_phases"].append(load.NumPhases())
            customers["kwh_measured"].append(kwh_measured)
            customers["kwh_theft"].append(kwh)
            customers["theft_kw"].append(pmult*row.get("kw_by_phase")*load.NumPhases())
            customers["measured_kw"].append(measured_kw)

        # Final df
        self.irregs = pd.DataFrame(customers)
        return self.irregs
    
    def _solve(self) -> altdss:
        self.altdss.Solution.Number = int(24/config.loadflow.stepsize)*config.simulation.num_days      
        self.altdss.Solution.Hour = -config.loadflow.stepsize
        (x.Reset() for x in self.altdss.Monitor())
        self.altdss.Solution.Solve()
        return self.altdss

    def _read_monitors(self) -> pd.DataFrame:
        vmeas, pmeas, qmeas = {}, {}, {}
        sbase = 1.00 if config.loadflow.kva_base is None else config.loadflow.kva_base
        for monitor in self.altdss.Monitor():
            if "Load" in monitor.Element.FullName().split("."): 
                bus_id, label = monitor.Element.Bus1.split(".")[0], "load"
            elif "Transformer" in monitor.Element.FullName().split("."): 
                bus_id, label = monitor.Element.Buses[-1].split(".")[0], "trafo"
            else: 
                continue
            node_order = monitor.Element.NodeOrder() if monitor.Element.NumTerminals() == 1 else np.split(monitor.Element.NodeOrder(), 2)[-1]
            node_order = np.asarray([x for x in node_order if x!=4])
            if monitor.PPolar:
                vbase = 1.00 if config.loadflow.kva_base is None else 1000*self.altdss.Bus[bus_id].kVBase
                vmags = np.split(monitor.AsMatrix()[:, 2:], 2, axis=1)[0][:, 0:-1:2]
                vmags = vmags[:, 0:len(node_order)]
                for i, node in enumerate(node_order):
                    if f"{bus_id}.{node}" not in vmeas:
                        vmeas[f"Vmag.{label}.{bus_id.split('_')[-1]}.{node}"] = vmags[:,i]/vbase
                    else:
                        vmeas[f"Vmag.{label}.{bus_id.split('_')[-1]}.{node}"] += vmags[:,i]/vbase
                        vmeas[f"Vmag.{label}.{bus_id.split('_')[-1]}.{node}"] /= 2
            else:
                pq = monitor.AsMatrix()[:,2:]
                p, q = pq[:, 0:2*len(node_order):2], pq[:, 1:2*len(node_order):2]
                for i, node in enumerate(node_order):
                    if f"{bus_id}.{node}" not in vmeas:
                        pmeas[f"kW.{label}.{bus_id.split('_')[-1]}.{node}"] = p[:, i]/sbase
                        qmeas[f"kvar.{label}.{bus_id.split('_')[-1]}.{node}"] = q[:, i]/sbase
                    else:
                        pmeas[f"kW.{label}.{bus_id.split('_')[-1]}.{node}"] += p[:, i]/sbase
                        qmeas[f"kvar.{label}.{bus_id.split('_')[-1]}.{node}"] += q[:, i]/sbase
        vmeas, pmeas, qmeas = pd.DataFrame(vmeas), pd.DataFrame(pmeas), pd.DataFrame(qmeas)
        if config.simulation.noise.get('verror') is not None:
            np.random.seed(self.random_state)
            vmeas *= np.random.normal(loc=1, scale=config.simulation.noise.get('verror')/300, size=vmeas.shape)
        if config.simulation.noise.get('perror') is not None:
            np.random.seed(self.random_state)
            pmeas *= np.random.normal(loc=1, scale=config.simulation.noise.get('perror')/300, size=vmeas.shape)
        if config.simulation.noise.get('qerror') is not None:
            np.random.seed(self.random_state)
            qmeas *= np.random.normal(loc=1, scale=config.simulation.noise.get('qerror')/300, size=vmeas.shape)
        return pd.concat((vmeas, pmeas, qmeas), axis=1)

    def _reorder_columns(self, keys: List[str], sequence: List[str]) -> List[str]:
        grouped = defaultdict(list)
        for key in keys:
            parts = key.split('.')
            if len(parts) >= 3:
                grouped[parts[2]].append(key)
        ordered = []
        for group in sequence:
            if group in grouped:
                sorted_keys = sorted(grouped[group], key=lambda k: int(k.split('.')[-1]))
                ordered.extend(sorted_keys)
        return ordered

    def _get_io_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        input_cols, output_cols = [], []
        mapping = {
            "voltage_magnitudes": [x for x in self.meas.columns if x.startswith("Vmag")],
            "active_power": [x for x in self.meas.columns if x.startswith("kW")],
            "reactive_power": [x for x in self.meas.columns if x.startswith("kvar")],
        }
        if config.simulation.trafo_meas:
            input_cols += [col for key in config.simulation.input_data for col in mapping.get(key, []) if 'trafo' in col]
        dist = {x.split('_')[-1]: y for x, y in self.bus_distances.items()}
        order = [k for k, _ in sorted(dist.items(), key=lambda item: item[1])]
        cols = []
        for key in config.simulation.input_data:
            cols += [x for x in mapping.get(key, []) if 'load' in x and 'Vmag' not in x]
        input_cols += self._reorder_columns(cols, order)
        for key in ["voltage_magnitudes"]:
            cols = [x for x in mapping.get(key, []) if 'load' in x and 'Vmag' in x]
        output_cols += self._reorder_columns(cols, order)
        output_cols.sort(key=lambda s: int(s.split('.')[-1]))
        input_cols.sort(key=lambda s: int(s.split('.')[-1]))
        self.inputs = self.meas[input_cols]
        self.outputs = self.meas[output_cols]
        return self.inputs, self.outputs

    def _bus_distances(self) -> Dict[str, float]:
        self.bus_distances = {}
        nx.set_node_attributes(self.nxgraph, np.nan, 'dist_km')
        root_bus = [u for u, v in self.nxgraph.in_degree() if v == 0][0]
        self.nxgraph.nodes[root_bus]['dist_km'] = 0.0
        self.bus_distances[root_bus] = 0.0
        for bus in tuple(nx.dfs_tree(self.nxgraph, root_bus))[1:]:
            upBus = tuple(self.nxgraph.predecessors(bus))[0]
            distance = np.round(self.nxgraph.nodes[upBus]['dist_km']+self.nxgraph.edges[upBus, bus]['length'], 3)
            self.nxgraph.nodes[bus]['dist_km'] = distance
            self.bus_distances[bus] = distance
        return self.bus_distances

    def _assemble_topology(self) -> nx.DiGraph:
        self.nxgraph = nx.DiGraph()
        DSS = self.altdss.to_dss_python()
        for element in DSS.ActiveCircuit.PDElements:
            DSS.ActiveCircuit.ActiveCktElement(element.Name)
            bus1, bus2 = tuple(map(lambda x: x.split(".")[0], DSS.ActiveCircuit.CktElements.BusNames[:2]))
            if bus1 != bus2:
                if element == "Line":
                    DSS.ActiveCircuit.Lines.Name = element.Name
                    length = 1e-3 if DSS.ActiveCircuit.Lines.IsSwitch else DSS.ActiveCircuit.Lines.Length
                else:
                    length = 1e-3
                self.nxgraph.add_edge(bus1, bus2, name=element.Name, length=length)
        self._bus_distances()
        return self.nxgraph

    def acquire(
            self, 
            customers: Optional[Dict[str, Dict[str, Any]]] = None
        ) -> pd.DataFrame:
        
        # Compile system
        self._compile()

        # Set solution mode to daily
        self._set_solution_mode()

        # Set loadflow config params
        self._set_params(vsource=config.loadflow.vsource)

        # Update loadshapes
        self._update_loadshapes()

        # Install monitors (smart meters)
        self._install_monitors()

        # Insert NTL
        if customers is None:
            if config.irregularity.num_irregs>0:
                customers = {
                    "name": np.random.choice(tuple([x.Name for x in self.altdss.Load if x.Class%10 == 1]), config.irregularity.num_irregs, replace=False),
                    "kw_by_phase": np.random.uniform(1.5, 3.5, config.irregularity.num_irregs),
                    "pf": np.random.uniform(0.88, 1.00, config.irregularity.num_irregs),
                    "shape_type": [config.irregularity.loadshape]*config.irregularity.num_irregs,
                    "working_period": [config.irregularity.ntl_period]*config.irregularity.num_irregs,
                    "theft_on_training": [config.irregularity.theft_on_training]*config.irregularity.num_irregs,
                }
            else:
                customers = {}

        # Insert NTL
        self._insert_ntl(customers)

        # Assemble topology
        self._assemble_topology()

        # Solve loadflows
        self._solve()

        # Read moonitors
        self.meas = self._read_monitors()

        # Get I/O data
        self._get_io_data()

        # Training dataset
        #training_idx = np.random.choice(self.inputs.index, size=int(24/config.loadflow.stepsize)*config.simulation.days_for_training, replace=False)
        self.training = {
            "inputs": self.inputs.iloc[:int(24/config.loadflow.stepsize)*config.simulation.days_for_training], 
            "outputs": self.outputs.iloc[:int(24/config.loadflow.stepsize)*config.simulation.days_for_training]
        }

        # Inference dataset
        #prediction_idx = np.setdiff1d(self.inputs.index, training_idx)
        self.inference = {"inputs": self.inputs.iloc[int(24/config.loadflow.stepsize)*config.simulation.days_for_training:], "outputs": self.outputs.iloc[int(24/config.loadflow.stepsize)*config.simulation.days_for_training:]}

        return self.meas

    def plot_ntl(self) -> List[plt.figure]:
        timeseries = {}
        figures = []
        for i, row in self.irregs.iterrows():
            match = [name for name in self.altdss.Load.Name if name.split('_')[-1] in row.load_id.split('-')[-1] and 'ntl' not in name]
            if match:
                fig = plt.figure(figsize=(3.8, 2.2))
                load = self.altdss.Load[match[0]]
                measured_kw = np.array(load.Daily.PMult) * load.kW
                timeseries[match[0]] = np.round(np.trapz(measured_kw) / 4, 2)  # kWh assuming 15 min resolution
                step = config.loadflow.stepsize
                t = np.arange(0, 24, step)
                num_pts = int(24 / step)
                plt.plot(t, row.theft_kw[-num_pts:], color='red', label='NTL')
                plt.plot(t, measured_kw[-num_pts:], label='Measurement')
                plt.ylabel('Power (kW)')
                plt.xlabel('Time (h)')
                plt.legend()
                plt.grid()
                plt.tight_layout()
                figures.append(fig)
        return figures
    
    def plot_topology(
        self, 
        irreg_buses = [], 
        suspect_buses = []
        ) -> plt.figure:
        load_buses = [x.Bus1.split(".")[0].split('_')[-1] for x in self.altdss.Load if x.Class%10 == 1]
        if not self.nxgraph:
            self._assemble_topology()
        root_bus = [u for u, v in self.nxgraph.in_degree() if v == 0][0]
        coords = dict(zip(self.altdss.Bus.Name(), zip(self.altdss.Bus.X(), self.altdss.Bus.Y())))
        sizes, nodecolor = zip(*[
            (100, 'black') if node == root_bus else
            (25, 'red') if node.split('_')[-1] in irreg_buses else 
            (15, 'orange') if node.split('_')[-1] in suspect_buses else
            (25, 'black') if node.split('_')[-1] in load_buses else (25, 'gray')
            for node in self.nxgraph
        ])
        labels = {node: node[-3:] for node in self.nxgraph}
        fig = plt.figure(figsize=(3.8, 5.2))
        plt.title(self.uni_tr_mt)
        nx.draw_networkx(
            self.nxgraph, 
            pos=coords, 
            labels={}, # labels
            node_size=sizes, 
            node_shape='o', 
            arrows=False, 
            width=3, 
            node_color=nodecolor, 
            edge_color='silver', 
            alpha=0.70,
            font_size=9,
            font_color='black'
        )
        plt.tight_layout()
        return fig
