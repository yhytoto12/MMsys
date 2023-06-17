import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
from torchvision import transforms
import sys
import time
from torchvision.models import resnet18, resnet50
from viztracer import log_sparse

transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.Resize(224, antialias = True),
])

def func(x):
    return transform(x)

class Fusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, features):
        time.sleep(0.02)
        return features

# @log_sparse(stack_depth=10)
def worker(preprocessing_func, feature_extractor, input, i):
    print(i)
    with record_function(f'preprocessing [{i}]'):
        x = preprocessing_func(input)

    with record_function(f'load data to GPU [{i}]'):
        x = x.to('cuda', non_blocking=True)

    with record_function(f'feauture_extract [{i}]'):
        x = feature_extractor(x)

    return x

@log_sparse(stack_depth=10)
def parallel_inference(preprocessing_funcs, feature_extractors, fusion_model, inputs, pool : mp.Pool):
    results = []
    def collect_result(result):
        results.append(result)

    n_processes = len(inputs)

        # for i in range(len(inputs)):
    for i in range(n_processes):
        pool.apply_async(worker, args = (preprocessing_funcs[i], feature_extractors[i], inputs[i], i), callback = collect_result)

    with record_function('fusion'):
        fusion_model(results)

    return results

@log_sparse(stack_depth=10)
def better_inference(preprocessing_funcs, feature_extractors, fusion_model, inputs):
    outputs = []
    for i in range(len(inputs)):
        with record_function(f'preprocessing [{i}]'):
            x = preprocessing_funcs[i](inputs[i])

        with record_function(f'load data to GPU [{i}]'):
            x = x.to('cuda', non_blocking=True)

        with record_function(f'feauture_extract [{i}]'):
            outputs.append(feature_extractors[i](x))

    with record_function('fusion'):
        result = fusion_model(outputs)

    return result

@log_sparse(stack_depth=10)
def naive_inference(preprocessing_funcs, feature_extractors, fusion_model, inputs):
    xs = []
    for i in range(len(inputs)):
        with record_function(f'preprocessing [{i}]'):
            x = preprocessing_funcs[i](inputs[i])

        with record_function(f'load data to GPU [{i}]'):
            x = x.to('cuda', non_blocking=True)
            xs.append(x)

    outputs = []
    for i in range(len(inputs)):
        with record_function(f'feauture_extract [{i}]'):
            outputs.append(feature_extractors[i](xs[i]))

    with record_function('fusion'):
        result = fusion_model(outputs)

    return result

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    M = int(sys.argv[1])

    inputs = [torch.randn(1, 3, 1080, 720) for i in range(M)]
    preprocessing_funcs = [func for i in range(M)]
    feature_extractors = [resnet18().cuda() for i in range(M)]
    fusion_model = Fusion().cuda()

    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     schedule=schedule(wait=1, warmup=1, active=2),
    #     on_trace_ready=tensorboard_trace_handler('./log/multimodal')
    # ) as prof:
    #     for i in range(4):
    #         naive_inference(preprocessing_funcs, feature_extractors, fusion_model, inputs)
    #         prof.step()

    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     schedule=schedule(wait=1, warmup=1, active=2),
    #     on_trace_ready=tensorboard_trace_handler('./log/multimodal')
    # ) as prof:
    #     for i in range(4):
    #         better_inference(preprocessing_funcs, feature_extractors, fusion_model, inputs)
    #         prof.step()

    workers = []
    for i in range(M):
        w = mp.Process(
            target = worker,
            args = ()
        )

    with torch.no_grad():
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     schedule=schedule(wait=0, warmup=2, active=2, repeat=3),
    #     on_trace_ready=tensorboard_trace_handler('./log/multimodal')
    # ) as prof:
        for i in range(5):
            parallel_inference(preprocessing_funcs, feature_extractors, fusion_model, inputs, workers)
            # prof.step()

        # for i in range(10):
        #     naive_inference(preprocessing_funcs, feature_extractors, fusion_model, inputs)
        #     # prof.step()

        # for i in range(10):
        #     better_inference(preprocessing_funcs, feature_extractors, fusion_model, inputs)
        #     # prof.step()