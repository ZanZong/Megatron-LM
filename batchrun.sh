nohup bash run.sh hetero-train GPT-4.7B 1 8 1 128 1 > GPT-11B-gbs128.log 2>&1
nohup bash run.sh hetero-train GPT-4.7B 1 8 1 256 1 > GPT-11B-gbs256.log 2>&1
nohup bash run.sh hetero-train GPT-4.7B 1 8 1 512 1 > GPT-11B-gbs512.log 2>&1
nohup bash run.sh hetero-train GPT-4.7B 1 8 1 1024 1 > GPT-11B-gbs1024.log 2>&1
