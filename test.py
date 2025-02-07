import matplotlib.pyplot as plt
import numpy as np

# Define the number of GPUs
num_gpus = 3

# Define the execution times for each GPU (forward and backward)
# Format: [(forward_time, backward_time), ...]
execution_times = [
    (3, 3),  # GPU 1
    (2, 2),  # GPU 2
    (4, 4)   # GPU 3
]

# Define the number of micro-batches
num_micro_batches = 5

# Initialize the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Colors for different GPUs
forward_colors = ['skyblue', 'lightgreen', 'lightcoral']
backward_colors = ['deepskyblue', 'limegreen', 'indianred']

# Plotting the pipeline execution
current_time = 0

for micro_batch in range(num_micro_batches):
    for gpu in range(num_gpus):
        forward_time, backward_time = execution_times[gpu]
        
        # Forward pass
        ax.broken_barh([(current_time, forward_time)], (gpu * 10, 4), facecolors=(forward_colors[gpu]), edgecolor='black', label='Forward' if micro_batch == 0 and gpu == 0 else "")
        current_time += forward_time
        
        # Ensure backward pass starts after the forward pass completes for this micro-batch
        backward_start_time = current_time
        
        # Backward pass
        ax.broken_barh([(backward_start_time, backward_time)], (gpu * 10 + 5, 4), facecolors=(backward_colors[gpu]), edgecolor='black', label='Backward' if micro_batch == 0 and gpu == 0 else "")
        current_time += backward_time

# Set the labels and title
ax.set_xlabel('Time')
ax.set_ylabel('GPUs')
ax.set_yticks([5 + 10 * i for i in range(num_gpus)])
ax.set_yticklabels([f'GPU {i+1}' for i in range(num_gpus)])
ax.set_title('Pipeline Execution for One Forward One Backward Schedule')

# Add legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper right')

# Show grid lines
ax.grid(True)
# Show the plot
plt.savefig('pipeline.pdf')
    

# # Regex pattern to match the loss value
# pattern = r"\| lm loss: ([0-9]+\.[0-9]+E[+-][0-9]+) \|"
# loss_values = []

# with open('/home/zanzong/workspace/Megatron-LM/GPT-4.7B-gbs128-c.log', 'r') as log:
#     line = log.readline()
#     while line:
#         if str(line)[:10] != ' iteration':
#             line = log.readline()
#             continue
#         print(line)
#         # Extract loss values
#         match = re.search(pattern, line)
#         if match:
#             loss_values.append(float(match.group(1)))
#         line = log.readline()
# print(f"got {len(loss_values)} loss values")
# print(loss_values)
            