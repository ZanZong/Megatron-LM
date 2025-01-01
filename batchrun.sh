nohup bash run.sh hetero-train GPT-2.1B 1 8 1 128 1 1 > GPT-2.1B-gbs128-c.log 2>&1
nohup bash run.sh hetero-train GPT-2.1B 1 8 1 128 1 0 > GPT-2.1B-gbs128.log 2>&1
nohup bash run.sh hetero-train GPT-4.7B 1 8 1 128 1 1 > GPT-4.7B-gbs128-c.log 2>&1
nohup bash run.sh hetero-train GPT-4.7B 1 8 1 128 1 0 > GPT-4.7B-gbs128.log 2>&1

# nohup bash run.sh hetero-train GPT-6.2B 1 8 1 128 1 1 > GPT-6.2B-gbs128-c.log 2>&1
# nohup bash run.sh hetero-train GPT-6.2B 1 8 1 128 1 0 > GPT-6.2B-gbs128.log 2>&1
# nohup bash run.sh hetero-train GPT-11B 1 8 1 128 1 1 > GPT-11B-gbs128-c.log 2>&1
# nohup bash run.sh hetero-train GPT-11B 1 8 1 128 1 0 > GPT-11B-gbs128.log 2>&1

