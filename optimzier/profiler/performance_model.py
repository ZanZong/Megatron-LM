import os
import numpy as np
from functools import lru_cache
from enum import Enum

def float_from_env(key, default=-1):
    if key in os.environ:
        return float(os.environ[key])
    return default


def switch_from_env(key, default=False):
    if key in os.environ:
        return os.environ[key] in ['1', 'ON']
    return default

# Transformer configuraions.
llama_config = {
    'sequence_length': 264,
    'hidden_size': 1792,
    'num_attention_heads': 16,
    'layer_num': 136,
    'vocab_size': 50257,
    'batch_size': 8,
}


def flops_calculator(model_config):
    """ flops calculator """
    hidden_size = model_config["hidden_size"]
    num_layers = model_config["layer_num"]
    vocab_size = model_config["vocab_size"]
    seq_len = model_config["sequence_length"]
    batch_size = model_config["batch_size"]
    recompute_granularity = "selective"
    recompute_num_layers_of_stages = 4 # if recompute_granularity is block
    fp16 = True
    
    # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
    # https://arxiv.org/pdf/2104.04473.pdf and 
    # https://arxiv.org/pdf/2205.05198.pdf).
    selective_recompute_factor = 1
    checkpoint_activations_factor = 3 # set to 1 if disable activation checkpointing.

    if recompute_granularity == 'selective':
        selective_recompute_factor = 0.5
    elif recompute_granularity == 'full':
        checkpoint_activations_factor = 4
        
        assert recompute_num_layers_of_stages is not None
        recompute_layers = 0
        for i in range(len(recompute_num_layers_of_stages)):
            recompute_layers += recompute_num_layers_of_stages[i]
        checkpoint_activations_factor = 3 + 1. * (recompute_layers / num_layers)
    else :
        assert recompute_granularity is None, \
            'This recompute_granularity does not support' \
            'the calculation of tflops'
            
    # training with mixed precision, use fp16 calculation
    factor = 4 if fp16 else 8
    flops_per_iteration = (factor * checkpoint_activations_factor * batch_size * seq_len \
        * num_layers * (hidden_size**2)) * (1. + (seq_len / (6. * selective_recompute_factor \
            * hidden_size)))+ (6 * batch_size * seq_len * hidden_size * vocab_size)
    flops_per_layer = flops_per_iteration // num_layers
    
    return flops_per_layer, flops_per_iteration


class DeviceType(Enum):
    a100 = "a100",
    v100 = "v100",
    a10 = "a10",
    
class DeviceTFOPS():
    a100 = 312, # TFLOPS fp16
    v100 = 125,
    a10 = 125,

class LayerWiseCostModel:
    """ Cost model for estimating execution time, communication time and memory costs.
    """
    def __init__(self, model_configs, bandwidth=None) -> None:
        self.model_configs = model_configs
        self.sigma = 4 # 2 if mixed_precision
        self.alpha = 4 # h->4h->h
        # Save loaded performance model
        self.perf_model = None
        self.bw_net = float_from_env('NET_BANDWIDTH', 50 * 1e9 / 8) # Bytes per second, PCIe default
        self.bw_net = bandwidth if bandwidth is not None else self.bw_net
    
    @lru_cache(10)
    def get_compute_cost(self, device, fw_or_bw='f'):
        """ Predict the compute cost for a micro-batch according to paritioning and execution phase.

        Args:
            device (DeviceType): choose a supported device.
            fw_or_bw (str): Choose forward or backward option to estimate.

        Returns:
            int: computation time cost in seconds.
        """
        assert device is not None, f"device cannot be none, support {DeviceType}."
        real_tflops, total_tflops = flops_calculator(self.model_configs)
        cost = real_tflops / getattr(DeviceTFOPS, device.name)[0]
        if fw_or_bw == 'b':
            cost *= 2
        return cost / 1E9
    
    @lru_cache(10)
    def get_model_parameter_num(self, in_billion=False):
        numbers = []
        config = self.model_configs
        # https://towardsdatascience.com/how-to-estimate-the-number-of-parameters-in-transformer-models-ca0f57d8dff0
        ffn_hidden_size = self.alpha * config['hidden_size']
        attention = self.alpha * config['hidden_size'] ** 2
        feed_forward = 2 * config['hidden_size'] * ffn_hidden_size
        layer_norm = 2 * config['hidden_size']
        parameter_num = (attention + feed_forward + 2 * layer_norm) * config['layer_num']
        if in_billion:
            parameter_num /= 10**9 # Counted in billion 
        numbers.append(parameter_num)
        # print(f"parameter num={parameter_num}")
        return sum(numbers)
    
    # def get_communication_cost(self, parts):
    #     """Count the collective communication time for one layer with tp sharding.
    #         * `bw_net`: bandwidth of the network (GBps)
    #     """
    #     config = self.model_configs
    #     all_reduce_time = []
    #     # 1. tensor parallel all-reduce
    #     # Megatron paper, 2 for forward and 2 for backward
    #     message_size = self.sigma * 4 * (config['batch_size'] * config['hidden_size'] * config['sequence_length']) * config['layer_num']
    #     # 2. data parallel all-reduce
    #     message_size += round(self.get_model_parameter_num() * self.sigma / parts['pp'] /  parts['tp'])
    #     all_reduce_time.append(2 * (parts['dp'] - 1) / parts['dp'] * message_size  / self.bw_net)
    #     return sum(all_reduce_time)

    def get_tp_comm_cost(self, tp_size, bandwidth):
        """Count the collective communication time for one layer with tp sharding.
            * `bw_net`: bandwidth of the network (GBps)
        """
        config = self.model_configs
        # 1. tensor parallel all-reduce
        # Megatron paper, 2 for forward and 2 for backward
        message_size = self.sigma * 4 * (config['batch_size'] * config['hidden_size'] * config['sequence_length'])
        return 2 * (tp_size - 1) / tp_size * message_size  / bandwidth

    def get_dp_comm_cost(self, dp_size, bandwidth):
        """Count the collective communication time for data parallel.
            * `bw_net`: bandwidth of the network (GBps)
        """
        config = self.model_configs
        all_reduce_time = []
        # 2. data parallel all-reduce
        message_size = round(self.get_model_parameter_num() * self.sigma / dp_size)
        all_reduce_time.append(2 * (dp_size - 1) / dp_size * message_size / bandwidth)
        return sum(all_reduce_time)
    
    def get_memory_cost(self, tp_size, recompute=False):
        """Count the parameter, optimizer state and activations per layer according to ZeRO paper.
        """
        memory_cost = []
        config = self.model_configs
        Phi = self.get_model_parameter_num(in_billion=True)
        per_layer_param = Phi / config["layer_num"] / tp_size
        mem_cost = config['batch_size'] * config['sequence_length'] * (config['hidden_size'] / tp_size) * \
            ((16*self.sigma+2) + (2*self.sigma+1)*config['num_attention_heads']*config['sequence_length']/config['hidden_size']/tp_size)
        activation = int(mem_cost) / (1E9) # Activations in GB            
        # print(f"per_stage_param{per_stage_param}")
        optimizer_state = 5 * self.sigma * per_layer_param # Optimizer states
        # print(f"optimizer_state={optimizer_state}")
        mem_cost_in_GB = round(activation + optimizer_state, 3)
        # print(mem_cost_in_GB)
        if recompute:
            # Selective recompute effectiveness claimed in the Megatron3 paper.
            mem_cost_in_GB *= 0.35
        memory_cost.append(mem_cost_in_GB)
        return sum(memory_cost), activation
    

if __name__ == '__main__':
#     perf_model = CostModel(llama_config, perf_model_path='/home/zanzong/workspace/Megatron-LM/optimzier/profiler/dummy-model/dummy.joblib')
#     p_time = perf_model.get_compute_cost(parts)
#     print(f"Predicted Time: {p_time} s")
#     print(f"memory_cost={perf_model.get_memory_cost(parts)} GB")
#     print(f"comm_cost={perf_model.get_communication_cost(parts)} s")
    cost_m = LayerWiseCostModel(llama_config, bandwidth=10E9)
    print(cost_m.get_compute_cost(DeviceType.v100, 'f'))
    print(cost_m.get_memory_cost(parts={"pp":2, "tp":2, "dp":2}, recompute=True))
    print(cost_m.get_communication_cost(parts={"pp":2, "tp":2, "dp":2}))