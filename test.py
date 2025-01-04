import re

# Regex pattern to match the loss value
pattern = r"\| lm loss: ([0-9]+\.[0-9]+E[+-][0-9]+) \|"
loss_values = []

with open('/home/zanzong/workspace/Megatron-LM/GPT-4.7B-gbs128-c.log', 'r') as log:
    line = log.readline()
    while line:
        if str(line)[:10] != ' iteration':
            line = log.readline()
            continue
        print(line)
        # Extract loss values
        match = re.search(pattern, line)
        if match:
            loss_values.append(float(match.group(1)))
        line = log.readline()
print(f"got {len(loss_values)} loss values")
print(loss_values)
            