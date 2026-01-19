import time
import json
import requests
import pandas as pd
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

try: from Utils.MDMS import MDMS, LVSystems
except: from MDMS import MDMS, LVSystems

# Spark Session
from pyspark.sql import SparkSession
spark = SparkSession.builder\
    .appName("DataQuality")\
    .getOrCreate()

class ApiHandler:

    databricks_instance = "" # Insert your databricks instance
    token = "" # Insert your token

    def __init__(self, name:str) -> None:
        
        # Job name
        self.name = name
    
    def job_id(self) -> str:
        url = f"https://{self.databricks_instance}/api/2.2/jobs/list"
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(url, headers=headers)
        jobs = response.json().get("jobs", [])
        job = next((j for j in jobs if j["settings"]["name"] == self.name), None)
        if job:
            return job["job_id"]
        else:
            return None
    
    def cluster_id(self, cluster_name:str) -> str:
        url = f"https://{self.databricks_instance}/api/2.0/clusters/list"
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(url, headers=headers)
        clusters = response.json()
        for c in clusters["clusters"]:
            if c["cluster_name"]==cluster_name:
                return c["cluster_id"]
        return ""
    
    def task(self, uni_tr_mt:str, **kwargs) -> Dict[str, Any]:
        return dict(
            task_key = uni_tr_mt,
            notebook_task = dict(
                notebook_path = "/Workspace/Users/tailan_@hotmail.com/git/ml_smart_meter/Notebooks/Monte Carlo - Single NTL",
                base_parameters = dict(
                    uni_tr_mt = uni_tr_mt,
                ),
            ),
            depends_on = [{"task_key": x} for x in kwargs.get("depends_on", [])],
            max_retries = kwargs.get("task_retries", 3),
            min_retry_interval_millis = kwargs.get("min_retry_interval_millis", int(300*1e3)),
            retry_on_timeout = kwargs.get("retry_on_timeout", True),
            environment_key = "tgarcia",
        )
    
    def acquire_payload(self, sub:str, feeder:str, max_concurrent_tasks:int=10) -> Dict[str, Any]:
    
        # Job name
        payload = dict(name = self.name)

        # Global parameters
        payload.update({
            "parameters": [
                {"name": "substation", "default": sub},
                {"name": "feeder", "default": feeder}
            ]
        })

        # Acquire systems
        available_systems = []
        for uni_tr_mt in LVSystems(sub=sub, feeder=feeder):
            mdms = MDMS(sub=sub, feeder=feeder, uni_tr_mt=uni_tr_mt, random_state=None)
            buses = set([x.Bus1.split(".")[0] for x in mdms.altdss.Load if x.Class%10 == 1])
            if len(buses)>10:
                available_systems.append(mdms.uni_tr_mt)

        # Separete tasks in batches
        batches = [
            available_systems[i:i + max_concurrent_tasks] 
            for i in range(0, len(available_systems), max_concurrent_tasks)
        ]

        # Iterate over the batches
        tasks = []
        previous_batch_task_keys = []  # tarefas do lote anterior
        counter = 0
        for batch in batches:

            current_batch_task_keys = []

            # Create dummie task
            dummie_task_key = f"dummie_batch_{counter}"

            # Create dummie task
            dummie_task = dict(
                task_key = dummie_task_key,
                notebook_task = dict(
                    notebook_path = "/Workspace/Users/tailan_@hotmail.com/git/ml_smart_meter/MonteCarlo/dummy",
                ),
                depends_on = [{"task_key": x} for x in previous_batch_task_keys],
            )

            # Append dummie task
            tasks.append(dummie_task)

            # Create the batch of tasks
            for uni_tr_mt in batch:
                task_config = self.task(
                    uni_tr_mt=uni_tr_mt,
                    depends_on=[dummie_task_key],  # depende do dummie
                    task_retries=5,
                    min_retry_interval_millis=300000,
                    retry_on_timeout=True,
                )
                tasks.append(task_config)
                current_batch_task_keys.append(uni_tr_mt)
            
            #Update list of tasks
            previous_batch_task_keys = current_batch_task_keys #+ [dummie_task_key]
            counter += 1

        # Update payload
        payload.update(dict(tasks=tasks))

        # Insert tags
        payload.update(
            tags = {
                "PhD": "SMART GRID",
            }
        )

        # Environment
        payload.update(
            environments = [
                {
                    "environment_key" : "tgarcia",
                    "spec": {
                        "dependencies": ["dss-python", "altdss", "networkx", "torch", "scikit-learn", "tqdm", "toml", "cvxpy==1.6.0"],
                        "environment_version": "4",
                    }
                }
            ]
        )
        return payload
    
    def create_job(self, sub:str, feeder:str, max_concurrent_tasks:int=5) -> None:
        url = f"https://{self.databricks_instance}/api/2.2/jobs/"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        job_id = self.job_id()
        if job_id is not None:
            payload = dict(job_id = job_id, new_settings = self.acquire_payload(sub, feeder, max_concurrent_tasks=5))
            response = requests.post(url+"reset", headers=headers, data=json.dumps(payload))
        else:
            payload = self.acquire_payload(sub, feeder, max_concurrent_tasks=5)
            response = requests.post(url+"create", headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            job_id = self.job_id()
            print(f"Job criado/atualizado: https://{self.databricks_instance}/jobs/{job_id}")
        else:
            print(f"Problema ao criar/atualizar o job: {response.text}")
        return None