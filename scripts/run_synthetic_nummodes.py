import os
import subprocess as sp

configs = [
    'configs/synthetic/resnet18/synthetic_sequential.yaml',
    'configs/synthetic/resnet18/synthetic_parallel_naive.yaml',
    'configs/synthetic/resnet18/synthetic_parallel_improved.yaml',
    'configs/synthetic/resnet50/synthetic_sequential.yaml',
    'configs/synthetic/resnet50/synthetic_parallel_naive.yaml',
    'configs/synthetic/resnet50/synthetic_parallel_improved.yaml',
]

for config_file in configs:
    for num_modes in range(1, 8):
        print(config_file, num_modes)
        cmd = f'python benchmark_synthetic.py --config {config_file} --num_modes {num_modes}'
        sp.run(cmd.split())