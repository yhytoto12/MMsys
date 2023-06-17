import os
import subprocess as sp

configs = [
    'configs/synthetic/resnet18/synthetic_sequential.yaml',
    'configs/synthetic/resnet18/synthetic_parallel_naive.yaml',
    'configs/synthetic/resnet18/synthetic_parallel_improved.yaml',
    'configs/synthetic/resnet34/synthetic_sequential.yaml',
    'configs/synthetic/resnet34/synthetic_parallel_naive.yaml',
    'configs/synthetic/resnet34/synthetic_parallel_improved.yaml',
    'configs/synthetic/resnet50/synthetic_sequential.yaml',
    'configs/synthetic/resnet50/synthetic_parallel_naive.yaml',
    'configs/synthetic/resnet50/synthetic_parallel_improved.yaml',
    'configs/synthetic/resnet101/synthetic_sequential.yaml',
    'configs/synthetic/resnet101/synthetic_parallel_naive.yaml',
    'configs/synthetic/resnet101/synthetic_parallel_improved.yaml',
]

for config_file in configs:
    print(config_file)
    cmd = f'python benchmark_synthetic.py --config {config_file}'
    sp.run(cmd.split())