import os
import itertools
          
def run_grid_search():
    base_cmd = "bash run_silent.sh search GPT-1.3B {} {} {} {} {} nico1"
    tp = [1, 2, 4, 8]
    dp = [1, 2, 4, 8]
    pp = [1, 2, 4, 8]
    global_batch_size = [64, 128]
    micro_batch_size = [8, 16, 32]
    total_gpu_num = 8

    paral_space = itertools.product(*[tp, dp, pp, global_batch_size, micro_batch_size])
    cout = 0
    for tup in paral_space:
        t, d, p, gbs, mbs = tup
        if t * d * p == total_gpu_num:
            cout += 1
            os.system(base_cmd.format(t, d, p, gbs, mbs))

def run_manual_search():
    cmds = []
    cmds.append("bash run_silent.sh tunemem GPT-1.3B 1 8 1 64 8 nico1")
    cmds.append("bash run_silent.sh tunemem GPT-1.3B 1 8 1 64 12 nico1")
    cmds.append("bash run_silent.sh tunemem GPT-1.3B 1 8 1 64 16 nico1")
    cmds.append("bash run_silent.sh tunemem GPT-1.3B 1 8 1 128 8 nico1")
    cmds.append("bash run_silent.sh tunemem GPT-1.3B 1 8 1 128 12 nico1")
    cmds.append("bash run_silent.sh tunemem GPT-1.3B 1 8 1 128 16 nico1")
    cmds.append("bash run_silent.sh tunemem GPT-1.3B 2 1 4 128 8 nico1")
    cmds.append("bash run_silent.sh tunemem GPT-1.3B 2 1 4 128 12 nico1")
    cmds.append("bash run_silent.sh tunemem GPT-1.3B 2 1 4 128 16 nico1")
    cmds.append("bash run_silent.sh tunemem GPT-1.3B 2 2 2 128 8 nico1")
    cmds.append("bash run_silent.sh tunemem GPT-1.3B 2 2 2 128 12 nico1")
    cmds.append("bash run_silent.sh tunemem GPT-1.3B 2 2 2 128 16 nico1")
    cmds.append("bash run_silent.sh tunemem GPT-1.3B 2 4 1 128 8 nico1")
    cmds.append("bash run_silent.sh tunemem GPT-1.3B 2 4 1 128 12 nico1")
    cmds.append("bash run_silent.sh tunemem GPT-1.3B 2 4 1 128 16 nico1")
    for cmd in cmds:
        os.system(cmd)


def analysis(exp_name):
    log_path = f"/home/zanzong/workspace/Megatron-LM/logs/{exp_name}"
    import shutil, re
    import numpy as np
    import pathlib
    num_iter_to_count = 5
    throughput_info = {}
    for root, dirs, files in os.walk(log_path):
        if len(files) != 0:
            for file in files:
                iter_times = []
                # calc the avg per iteration cost
                with open(os.path.join(root, file), "r") as log:
                    lines = log.readlines()
                    lines.reverse()
                    for line in lines:
                        iter_time = re.findall(r"iteration \(ms\): (.+?) \|", line)
                        if len(iter_time) == 0:
                            continue
                        iter_times.append(float(iter_time[0]))
                        if len(iter_times) >= num_iter_to_count:
                            break
                # occur training err, e.g., OOM
                if len(iter_times) == 0:
                    continue
                time_cost = np.mean(iter_times) / 1000.0
                
                # extract parallel parameter
                t, p, d, gbs, mbs = re.findall(r"_t(.+)_p(.+)_d(.+)_gbs(.+)_mbs(.+)_", file)[0]
                thpt = round(float(gbs) / float(time_cost), 2)
                throughput_info[thpt] = f"tp={t}, pp={p}, dp={d}, global bs={gbs}, micro bs={mbs}, training throughput={thpt} sample/sec."

    for key in sorted(throughput_info.keys(), reverse=True):
        print(throughput_info[key])
    


if __name__ == "__main__":
    # run_grid_search()
    analysis("search")
    

