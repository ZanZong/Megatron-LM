from enum import Enum
import logging
import math
import os
import numpy as np
from typing import Dict, Any, Optional
# noinspection PyPackageRequirements
from gurobipy import GRB, Model, quicksum
from profiler.performance_model import LayerWiseCostModel

def mul(tup):
        return tup[0] * tup[1]

class ILPSolver:
    def __init__(self, model_config, cluster_config, device_type_map, cross_cluster_bd, gurobi_params: Dict[str, Any] = None):
        self.gurobi_params = gurobi_params
        self.model_config = model_config
        self.layer_num = model_config["layer_num"]
        self.cost_model = LayerWiseCostModel(model_configs=model_config, bandwidth=10E9)
        self.m = Model("hetero_parallel_opt_l{}".format(self.layer_num))
        if gurobi_params is not None:
            for k, v in gurobi_params.items():
                setattr(self.m.Params, k, v)
        # treat the bandwidth within a cluster as a constant
        self.het_cluster = cluster_config
        self.device_type_map2_name = device_type_map
        self.budgets = [self.het_cluster[clus]["budget"] for clus in sorted(self.het_cluster.keys()) \
                                for _ in range(mul(self.het_cluster[clus]['topo']))]
        self.device_types = [self.device_type_map2_name[clus] for clus in sorted(self.het_cluster.keys()) \
                                for _ in range(mul(self.het_cluster[clus]['topo']))]
        # the unit of bandwidths is GB/s
        self.cross_cluster_bd = cross_cluster_bd
        cluster_devices = [mul(self.het_cluster[key]['topo']) for key in self.het_cluster.keys()]
        self.shape = (sum(cluster_devices), self.layer_num)
        print(f"device number for each cluster {cluster_devices}")
        self.S = self.m.addVars(self.shape[0], self.layer_num, name="S", vtype=GRB.BINARY) # rep. layer distribution
        
    
    def build_model(self, part):
        # seed solver with a baseline strategy
        # for x in range(self.shape[0]):
        #     for y in range(self.shape[1]):
        #         self.init_constraints.append(self.m.addLConstr(self.S[x, y], GRB.EQUAL, 0))
        # self.m.update()
        
        #  define objective function
        tp_size = part["tp"]
        per_device_comp_time = [self.cost_model.get_compute_cost(self.device_types[i], "f") / tp_size for i in range(self.shape[0])]
        per_layer_mem_cost, per_layer_output_size = self.cost_model.get_memory_cost(tp_size, recompute=True)
        tp_cost = []
        offset = 0
        for key in range(len(self.het_cluster.keys())):
            cluster_device_num = mul(self.het_cluster[key]["topo"])
            per_layer_tp_comm_time = self.cost_model.get_tp_comm_cost(tp_size, self.het_cluster[key]["bandwidth"])
            tp_cost.extend([per_layer_tp_comm_time * quicksum(self.S[i, j] \
                                    for i in range(0, offset + cluster_device_num, tp_size) \
                                        for j in range(self.shape[1]))])
            offset += cluster_device_num
        pp_cost = [per_layer_output_size / self.cross_cluster_bd[(i, i + 1)] for i in range(len(self.het_cluster.keys()) - 1)]
        compute_cost = [per_device_comp_time[i] * self.S[i, j] for i in range(0, self.shape[0], tp_size) for j in range(self.shape[1])]
        total_cost = []
        total_cost.extend(tp_cost)
        total_cost.extend(pp_cost)
        total_cost.extend(compute_cost)
        self.m.setObjective(quicksum(total_cost), GRB.MINIMIZE)

        # add constarint
        # 1. restain with tp_size, tp group must have the same layer partitioning
        for i in range(0, self.shape[0], tp_size):
            if tp_size > 1:
                for step in range(tp_size - 1):
                    self.m.addLConstr(quicksum(self.S[i + step, j] for j in range(self.shape[1])), GRB.EQUAL, quicksum(self.S[i + step + 1, j] for j in range(self.shape[1])))
        # 2. each device must have layer, at most tp_size split
        for i in range(self.shape[0]):
            self.m.addLConstr(quicksum(self.S[i, j] for j in range(self.shape[1])), GRB.GREATER_EQUAL, 1)
        for j in range(self.shape[1]):
            self.m.addLConstr(quicksum(self.S[i, j] for i in range(self.shape[0])), GRB.EQUAL, tp_size)
        # 3. the layer number equal to the graph size
        self.m.addLConstr(quicksum(self.S[i, j] for i in range(0, self.shape[0], tp_size) for j in range(self.shape[1])), GRB.EQUAL, self.layer_num)
        # 4. memory budget
        for i in range(0, self.shape[0]):
            self.m.addLConstr(quicksum(self.S[i, j] * per_layer_mem_cost for j in range(self.shape[1])), GRB.LESS_EQUAL, self.budgets[i])

    def solve(self):
        self.m.message("\n\nStarting solve\n\n")
        self.m.optimize()
        
        infeasible = (self.m.status == GRB.INFEASIBLE)
        if infeasible:
            raise ValueError("Infeasible model, check constraints carefully. Insufficient memory?")
        if self.m.solCount < 1:
            raise ValueError(f"Model status is {self.m.status} (not infeasible), but solCount is {self.m.solCount}")
        
        s_out = np.zeros((self.shape[0], self.shape[1]), dtype=np.int32)
        p_out = np.zeros((self.shape[0]), dtype=np.int32)
        for t in range(self.shape[0]):
            for i in range(self.shape[1]):
                try:
                    s_out[t][i] = int(self.S[t, i].X)
                except (AttributeError, TypeError) as e:
                    print(e)
        return s_out, p_out


def solve_ilp_gurobi(model_config, cluster_config, device_type_map, cross_cluster_bd, part):
    """
    Memory-accurate solver with garbage collection.
    :param g: DFGraph -- graph definition extracted from model
    :param budget: int -- budget constraint for solving
    """
    param_dict = {'LogToConsole': 1,
                  'Threads': 2,
                  'TimeLimit': math.inf,
                  'OptimalityTol': 1e-2,
                  'IntFeasTol': 1e-3,
                  'Presolve': 2,
                  'StartNodeLimit': 10000000}
    ilpsolver = ILPSolver(model_config, cluster_config, device_type_map, cross_cluster_bd)
    ilpsolver.build_model(part)
    try:
        s_out, p_out = ilpsolver.solve()
        print(f"Solved S = \n{s_out}\n")
        # print(f"Solved P\n = {p_out}\n")
        ilp_feasible = True
    except ValueError as e:
        print(e)
        ilp_feasible = False
    return None


np.set_printoptions(threshold=np.inf)
MODEL_NAMES = ["llama", "gpt"]

# def extract_params():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model-name', default="VGG16", choices=list(sorted(MODEL_NAMES)))
#     parser.add_argument("-b", "--batch-size", type=int, default=1)

#     _args = parser.parse_args()
#     return _args


def prefix_min_np(values: np.ndarray):
    assert values.ndim == 1
    values_min = np.copy(values)
    for i in range(1, values.shape[0]):
        values_min[i] = min(values_min[i - 1], values[i])
    return values_min


def roundup(round_factor: int, number: float) -> int:
    """helper function to round up a number by a factor"""
    return int(np.ceil(float(number) / round_factor) * round_factor)


def dist_points(start, stop, n, min_val=0.0):
    assert start < stop, "Start range must be below end of range"
    assert start > 0 and stop > 0, "Start and stop ranges must be positive"
    pts = sorted(start + np.arange(0, 1, 1. / n) * (stop - start))
    return [p for p in pts if p > min_val]

def heurist_search():
    # using heurists to explore partition setting:
    # If sub-device mesh perform worse than existing sub-mesh 
    #    with the same size, break this depth-first loop.
    pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("GDTrainer")
    # args = extract_params()
    # args.batch_size = 32
    # args.model_name = 'gpt'
    from profiler.performance_model import DeviceTFOPS, DeviceType
    device_type_map = {0: DeviceType.a100, 1: DeviceType.v100, 2: DeviceType.a10} # small index should be high-end device
    cluster_config = {
        0: {'topo': [1, 4], "bandwidth": 12, "budget": 40},
        1: {'topo': [1, 2], "bandwidth": 12, "budget": 16},
        2: {'topo': [1, 2], "bandwidth": 12, "budget": 24},
    }
    cross_cluster_bd = {(0, 1): 1, (1, 2): 16} # GB/s
    # Transformer configuraions.
    GPT_2_1B = {
        'sequence_length': 1024,
        'hidden_size': 4096,
        'num_attention_heads': 32,
        'layer_num': 10,
        'vocab_size': 32000,
        'batch_size': 4,
    }
    GPT_4_7B = {
        'sequence_length': 1024,
        'hidden_size': 4096,
        'num_attention_heads': 32,
        'layer_num': 24,
        'vocab_size': 32000,
        'batch_size': 4,
    }
    GPT_6_2B = {
        'sequence_length': 1024,
        'hidden_size': 4096,
        'num_attention_heads': 32,
        'layer_num': 32,
        'vocab_size': 32000,
        'batch_size': 4,
    }
    GPT_11B = {
        'sequence_length': 1024,
        'hidden_size': 4096,
        'num_attention_heads': 32,
        'layer_num': 56,
        'vocab_size': 32000,
        'batch_size': 4,
    }
    
    # Tunable partition method and micro-batch size.
    parts = {
        'tp': 1,
        'pp': 8,
        'dp': 1,
    }
    solve_ilp_gurobi(GPT_11B, cluster_config, device_type_map, cross_cluster_bd, parts)

    
    