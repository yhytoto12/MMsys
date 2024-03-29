import os
import argparse
from omegaconf import OmegaConf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101

from torch.utils.data import DataLoader as DataLoader
from realtime_dataloader import DataLoader as RealTimeDataLoader
from utils import get_profile_stats

from multimodal_dataloader import (
    RealTimeMultimodalDataPipe,
    RealTimeDataPipe,
    MultimodalDataLoadManager,
)

from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler, record_function

def read_image():
    return np.random.randint(0, 256, (1280, 720, 3), dtype=np.uint8)

class fusion_net(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fuse = nn.Linear(in_features, out_features)

    def forward(self, features):
        x = torch.cat(list(features.values()), dim = -1)
        x = self.fuse(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/synthetic.yaml')
    parser.add_argument('--num_modes', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default=None)

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    if args.num_modes == 0:
        num_modes = config.num_modes
    else:
        num_modes = args.num_modes

    manager_type = config.manager.type # sequential, parallel

    if args.log_dir is not None:
        if manager_type == 'sequential':
            profile_name = f'{manager_type}_{config.feature_extractors[0]}_{num_modes}'
        else:
            if config.manager.preprocess_only:
                profile_name = f'{manager_type}_naive_{config.feature_extractors[0]}_{num_modes}'
            else:
                profile_name = f'{manager_type}_improved_{config.feature_extractors[0]}_{num_modes}'

        trace_handler = tensorboard_trace_handler(args.log_dir, profile_name)
    else:
        trace_handler = None

    feature_extractors = config.feature_extractors

    modes = [f'M{i}' for i in range(num_modes)]

    read_fn_dict = {
        mode : read_image for mode in modes
    }

    preprocess_fn_dict = {
        mode : transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias = True),
            transforms.CenterCrop(224),
            transforms.Normalize(0.5, 0.5),
        ]) for mode in modes
    }

    total_feature_dim = 0
    feature_extractor_dict = {}
    for i, mode in enumerate(modes):
        if len(feature_extractors) < num_modes:
            model = eval(feature_extractors[0])()
        else:
            model = eval(feature_extractors[i])()
        total_feature_dim += model.fc.in_features
        model.fc = nn.Identity()
        model.eval()
        feature_extractor_dict[mode] = model.cuda()

    fusion = fusion_net(total_feature_dim, config.fusion.out_features).cuda()
    torch.cuda.synchronize()

    if 'sequential' in manager_type:
        datapipes = RealTimeMultimodalDataPipe(read_fn_dict, preprocess_fn_dict)
        dataloader = RealTimeDataLoader(datapipes, batch_size = 1, num_workers = 0, shuffle = False)
        dl = iter(dataloader)
        with profile(
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule = schedule(wait = 1, warmup = 2, active = config.profiler.num_iterations),
            on_trace_ready = trace_handler,
        ) as prof:
            for i in range(3 + config.profiler.num_iterations):
                data = next(dl)
                with record_function('extract features'):
                    features = {
                        mode : feature_extractor_dict[mode](v.cuda()) for mode, v in data.items()
                    }

                output = fusion(features)
                torch.cuda.synchronize()
                prof.step()

            avg_time, std = get_profile_stats(prof)
            print(f'average time : {avg_time:.3f} ± {std:.3f} ms')

    elif 'parallel' in manager_type:
        datapipe_dict = {
            mode : RealTimeDataPipe(
                read_fn = read_fn_dict[mode],
                preprocess_fn = preprocess_fn_dict[mode],
                name = mode,
            ) for mode in modes
        }

        dataloader_dict = {
            mode : RealTimeDataLoader(
                datapipe_dict[mode],
                num_workers = 1,
                batch_size = 1,
                shuffle = False,
            ) for mode in modes
        }

        manager = MultimodalDataLoadManager(
            dataloader_dict,
            feature_extractor_dict,
            preprocess_only = config.manager.preprocess_only,
        )

        with profile(
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule = schedule(wait = 1, warmup = 2, active = config.profiler.num_iterations),
            on_trace_ready = trace_handler,
        ) as prof:
            for i in range(3 + config.profiler.num_iterations):
                features = manager.get_data()

                if config.manager.preprocess_only:
                    with record_function("extract features"):
                        features = {mode : feature_extractor_dict[mode](v) for mode, v in features.items()}

                output = fusion(features)
                torch.cuda.synchronize()
                prof.step()

            avg_time, std = get_profile_stats(prof)
            print(f'average time : {avg_time:.3f} ± {std:.3f} ms')

    else:
        raise NotImplementedError
