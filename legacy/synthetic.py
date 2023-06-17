import torch
import torch.nn as nn
import torchvision
import torchaudio
import sys
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity, record_function
from torchvision.models import resnet18, resnet50, resnet34, resnet101
from torch.utils.data import IterableDataset
from realtime_dataloader import DataLoader
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor

PROFILE_N_RUN = 10

def read_image():
    return np.random.randint(0, 256, (1080, 720, 3), dtype=np.uint8)

def read_audio():
    return np.random.randn(3, 48000).astype(np.float32)

class RealTimeDataMulti(IterableDataset):
    def __init__(self, transforms):
        self.transforms = transforms
        self.n_modes = len(transforms)

    def __iter__(self):
        return self

    def __next__(self):
        data = []
        for i in range(self.n_modes):
            with record_function(f'[{i}] read data'):
                data.append(read_image())

        for i in range(self.n_modes):
            with record_function(f'[{i}] transform data'):
                data[i] = self.transforms[i](data[i])

        return data

class RealTimeData(IterableDataset):
    def __init__(self, transform, read_func, flag):
        self.transform = transform
        self.read_func = read_func
        self.flag = flag

    def __iter__(self):
        return self

    def __next__(self):
        with record_function(f'[{self.flag}] read data'):
            data = self.read_func()

        with record_function(f'[{self.flag}] transform data'):
            data = self.transform(data)

        return data

# def fusion(results):
#     data = torch.cat(results, dim = 1)
#     data =

def inference_navie(dataloader, feature_extractors, prof, fusion_net):
    n_modes = len(feature_extractors)
    dl = iter(dataloader)

    for i in range(PROFILE_N_RUN + 3):
        data = next(dl)

        results = []
        for j in range(n_modes):
            d = data[j].cuda()
            with record_function(f'[{j}] feature extract'):
                results.append(feature_extractors[j](d))

        with record_function('fusion'):
            results = torch.cat(results, dim = 1)
            fusion_net(results)

        with record_function('post_process'):
            time.sleep(0.02)

        prof.step()

def inference_ourb(dataloaders, feature_extractors, prof, fusion_net):
    n_modes = len(feature_extractors)

    def process_data(loader):
        data = next(loader)
        data = data.cuda()
        return data

    loaders = [iter(dataloader) for dataloader in dataloaders]

    for i in range(PROFILE_N_RUN + 3):
        results = []

        with ThreadPoolExecutor() as executor:
            for j in range(n_modes):
                results.append(executor.submit(process_data, loaders[j]))

        results = [res.result() for res in results]

        for j in range(n_modes):
            with record_function(f'[{j}] feature extract'):
                results[j] = feature_extractors[j](results[j])


        with record_function('fusion'):
            results = torch.cat(results, dim = 1)
            fusion_net(results)

        with record_function('post_process'):
            time.sleep(0.02)

        prof.step()

def inference_ourc(dataloaders, feature_extractors, prof, fusion_net):
    n_modes = len(feature_extractors)

    def extract_feature(loader, feature_extractor):
        data = next(loader)
        data = data.cuda()
        with record_function(f'feature extract'):
            data = feature_extractor(data)
        return data

    loaders = [iter(dataloader) for dataloader in dataloaders]

    for i in range(PROFILE_N_RUN + 3):
        results = []

        with ThreadPoolExecutor() as executor:
            for j in range(n_modes):
                results.append(executor.submit(extract_feature, loaders[j], feature_extractors[j]))

        results = [res.result() for res in results]

        with record_function('fusion'):
            results =  torch.cat(results, dim = 1)
            fusion_net(results)

        with record_function('post_process'):
            time.sleep(0.02)

        prof.step()

if __name__ == '__main__':
    n_modes = int(sys.argv[1])
    run_type = sys.argv[2]

    transforms = [
        torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Resize(224, antialias = True),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]) for _ in range(n_modes)
    ]

    fusion_net = nn.Linear(n_modes * 2048, 1024).cuda()

    # model = eval()

    feature_extractors = []
    for i in range(n_modes):
        model = resnet101()
        model._modules['fc'] = nn.Identity()
        feature_extractors.append(model.cuda().eval())

    # feature_extractors = [
    #     resnet101().cuda().eval() for _ in range(n_modes)
    # ]

    with profile(
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule = schedule(wait = 1, warmup = 2, active = PROFILE_N_RUN),
        on_trace_ready = tensorboard_trace_handler('./final_log/synthetic-4', f'{n_modes}-{run_type}_{os.getpid()}')
    ) as prof:
        if 'naive0' in run_type:
            dataset = RealTimeDataMulti(transforms)
            dataloader = DataLoader(dataset, batch_size = 1, num_workers = 0, shuffle = False)
            inference_navie(dataloader, feature_extractors, prof, fusion_net)
        elif 'naive1' in run_type:
            dataset = RealTimeDataMulti(transforms)
            dataloader = DataLoader(dataset, batch_size = 1, num_workers = 1, shuffle = False)
            inference_navie(dataloader, feature_extractors, prof, fusion_net)
        elif 'ourb' in run_type:
            datasets = [RealTimeData(transforms[i], read_image, i) for i in range(n_modes)]
            dataloaders = [
                DataLoader(datasets[i], num_workers = 1, batch_size = 1, shuffle = False) for i in range(n_modes)
            ]
            inference_ourb(dataloaders, feature_extractors, prof, fusion_net)
        elif 'ourc' in run_type:
            datasets = [RealTimeData(transforms[i], read_image, i) for i in range(n_modes)]
            dataloaders = [
                DataLoader(datasets[i], num_workers = 1, batch_size = 1, shuffle = False) for i in range(n_modes)
            ]
            inference_ourc(dataloaders, feature_extractors, prof, fusion_net)