import subprocess
import re
import numpy as np
import time
import os
import shutil

def check_error_in_file(file_path, error_message):
    """
    检查文件中是否包含特定的错误信息字符串。

    :param file_path: str, 要检查的文件的路径
    :param error_message: str, 要搜索的错误信息字符串
    :return: bool, 如果文件中包含错误信息则返回True，否则返回False
    """
    try:
        with open(file_path, 'r') as file:
            # 读取文件的所有内容
            content = file.read()
            # 检查特定的错误信息是否存在于文件内容中
            if error_message in content:
                return True
        return False
    except FileNotFoundError:
        print("文件未找到，请检查文件路径是否正确")
        return False
    except IOError:
        print("文件读取出错")
        return False
    
def extract_vf_tf_lists(file_path):
    """
    从文件中分别提取 vision forward 时间和 text forward 时间，
    并将它们分别存储在两个列表中。

    :param file_path: str, 要分析的文件的路径
    :return: tuple, 包含两个列表的元组，第一个是 vf 时间列表，第二个是 tf 时间列表
    """
    vf_list = []
    tf_list = []
    # 正则表达式匹配 vf 和 tf 数值
    pattern = r"vf:([0-9\.]+)\|tf:([0-9\.]+)"
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    # 将匹配的 vf 和 tf 转换为浮点数，并分别添加到对应的列表中
                    vf_list.append(float(match.group(1)))
                    tf_list.append(float(match.group(2)))
    except FileNotFoundError:
        print("文件未找到，请检查文件路径是否正确")
    except IOError:
        print("文件读取出错")
    
    return vf_list, tf_list

def clear_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def do_profile(vL, tL, pp_list, mbs_list):
    oom_pp = 0
    oom_mbs = 999
    clear_directory(f"./profile_logs/FlexProfile_vL{vL}_tL{tL}")
    profile_start_time = time.time()
    profile_record_path = f"./profile_logs/FlexProfile_vL{vL}_tL{tL}/profile_record_vL{vL}_tL{tL}.txt"
    with open(profile_record_path, 'w') as record:
        record.write(f"# Vision layer {vL}, # Text layer {tL}\n==================================\n")
        for pp in pp_list:
            for mbs in mbs_list:
                if pp<=oom_pp and mbs>=oom_mbs:
                    print(f"Skip profiling FlexModal with pp={pp}, mbs={mbs} due to CUDA out of memory error")
                    print("==================================")
                    continue
                gbs = mbs * 4
                print(f"Profiling FlexModal with pp={pp}, mbs={mbs}, gbs={gbs}, vL={vL}, tL={tL}")
                # try:
                #     command = ['bash', 'run_FlexModal_silent.sh', str(pp), str(1), str(mbs), str(gbs), str(vL), str(tL)]
                #     # command = ['bash', 'run_FlexModal_silent_pjlab.sh', str(pp), str(1), str(mbs), str(gbs)] # on pjlab cluster
                #     process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                #     stdout, stderr = process.communicate(timeout=120)
                #     # result = subprocess.run(['bash', 'run_FlexModal_silent.sh', str(pp), str(1), str(mbs), str(gbs), str(vL), str(tL)], timeout=120, capture_output=True, text=True)
                # except subprocess.TimeoutExpired: # 超时以后不会断开
                #     print(f"FlexModal with pp={pp}, mbs={mbs}, gbs={gbs} took too long to run")
                #     process.kill()
                #     stdout, stderr = process.communicate()
                #     print("==================================")
                #     continue
                output_path = f'./profile_logs/FlexProfile_vL{vL}_tL{tL}/PP{pp}_mbs{mbs}.log'
                if check_error_in_file(output_path, "CUDA out of memory"):
                    print(f"CUDA out of memory error occurred when running FlexModal with pp={pp}, mbs={mbs}, gbs={gbs}")
                    print("==================================")
                    oom_pp = pp
                    oom_mbs = mbs
                    continue
                vf_list, tf_list = extract_vf_tf_lists(output_path)
                vf_time = np.mean(vf_list[2*pp:]) # 丢弃前两个iteration的值
                tf_time = np.mean(tf_list[2*pp:])
                # 结果不能是nan
                if np.isnan(vf_time) or np.isnan(tf_time):
                    print(f"Something went wrong, FlexModal with pp={pp}, mbs={mbs}, gbs={gbs} has nan value")
                    print("==================================")
                    continue
                record.write(f"{pp}, {mbs} | {vf_time:.2f}, {tf_time:.2f}\n")

                print(f"Finished running FlexModal with pp={pp}, mbs={mbs}, gbs={gbs}, Avg vf time: {vf_time}, Avg tf time: {tf_time}")
                print("==================================")

    profile_end_time = time.time()
    print(f"Finished profiling in {profile_end_time - profile_start_time} seconds, record has been saved in {profile_record_path}")

if __name__ == "__main__":
    # Layers must be in line with model configuration
    vL = 168
    tL = 88
    pp_list = [2, 4, 8, 16]
    mbs_list = [1, 2, 4, 8, 16]
    do_profile(vL, tL, pp_list, mbs_list)