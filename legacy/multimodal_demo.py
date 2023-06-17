import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
import torchvision
import torchaudio
import numpy as np
import time
from viztracer import VizTracer, log_sparse

from torch.utils.data import IterableDataset
from realtime_dataloader import DataLoader
from concurrent.futures import ThreadPoolExecutor
# torch.multiprocessing.set_start_method('spawn')

def read_image():
    return np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

def read_audio():
    return np.random.randn(3, 48000).astype(np.float32)

class RealTimeDataMulti(IterableDataset):
    def __init__(self, image_transform, audio_transform, image_fe = None, audio_fe = None):
        self.image_transform = image_transform
        self.audio_transform = audio_transform

    def __iter__(self):
        return self

    def __next__(self):
        with record_function('read image'):
            image = read_image()

        with record_function('read audio'):
            audio = read_audio()

        with record_function('preprocess image'):
            preprocessed_image = self.image_transform(image)

        with record_function('preprocess audio'):
            preprocessed_audio = self.audio_transform(audio)

        return preprocessed_image, preprocessed_audio

class RealTimeData(IterableDataset):
    def __init__(self, transform, read_func):
        self.transform = transform
        self.read_func = read_func
        # self.fe = fe.cuda()

    def __iter__(self):
        return self

    def __next__(self):
        with record_function('read data'):
            data = self.read_func()

        with record_function('transform data'):
            data = self.transform(data)

        # data = data.cuda()

        # data = self.fe(data)
        # yield data

        return data


class MultiDataLoader:
    def __init__(self, datasets):
        self.datasets = datasets
        self.n_datasets = len(datasets)
        self.data_loaders = [iter(DataLoader(
            dataset, num_workers = 1, batch_size = 1
        )) for dataset in datasets]

    # def get_data_loaders(self):
    #     # return zip(*[DataLoader(
    #     #     dataset, num_workers = 1, batch_size = None, prefetch_factor = 1
    #     # ) for dataset in self.datasets])
    #     return zip(*self.data_loaders)

    # def __iter__(self):
    #     # for loaders in self.get_data_loaders():
    #     for loaders in self.get_data_loaders():
    #         data_list = []
    #         for data in loaders:
    #             data_list.append(data)

    #         yield data_list

    def get_next(self, i):
        loader = self.data_loaders[i]
        # print(loader)
        data = next(loader)
        return data

doit = False

@log_sparse(stack_depth=10)
def inference(dataloader, feature_extractors):
    n = len(feature_extractors)
    dl = iter(dataloader)

    for i in range(5):
        data = next(dl)

        results = []
        for j in range(n):
            d = data[j].cuda()
            results.append(feature_extractors[j](d))

        time.sleep(0.03)

def fe(loader, feature_extractor):
    data = next(loader)
    data = data.cuda()
    data = feature_extractor(data)
    return data

@log_sparse(stack_depth=10)
def inference2(dataloaders, feature_extractors):

    loaders = [iter(dataloader) for dataloader in dataloaders]
    # loaders = dataloaders
    n = len(loaders)

    for i in range(5):
        # data = []
        results = []

        with ThreadPoolExecutor(max_workers=n) as executor:
            for j in range(n):
                results.append(executor.submit(fe, loaders[j], feature_extractors[j]))
        # for loader in loaders:
        #     data.append(next(loader))

        results = [result.result() for result in results]

        time.sleep(0.03)



if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')

    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.Resize(224, antialias = True),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    audio_transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda x : torch.from_numpy(x)),
        torchaudio.transforms.MFCC(sample_rate=16000),
        torchvision.transforms.Resize((224, 224), antialias = True),
    ])

    from torchvision.models import resnet18, resnet50

    feature_extractors = [
        resnet50().cuda(),
        resnet50().cuda(),
    ]

    # fe = torchvision.models.resnet18()



    # av_data = RealTimeDataMulti(image_transform, audio_transform)
    # # dataloader = DataLoader(av_data, batch_size = 1, num_workers = 1, prefetch_factor = 1, shuffle = False)
    # dataloader = DataLoader(av_data, batch_size = 1, num_workers = 0, shuffle = False)
    # inference(dataloader, feature_extractors)


    image_data = RealTimeData(image_transform, read_image)
    audio_data = RealTimeData(audio_transform, read_audio)
    datasets = [image_data, audio_data]
    better_dataloader2 = [
        DataLoader(image_data, num_workers=1, batch_size=None),
        DataLoader(audio_data, num_workers=1, batch_size=None),
    ]

    inference2(better_dataloader2, feature_extractors)

    # time.sleep(2)


    # with profile(
    #     activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     schedule = schedule(wait=1, warmup=1, active=2),
    #     on_trace_ready = tensorboard_trace_handler('./log/audio-visual'),
    # ) as prof:

    # with VizTracer(output_file="result.json") as tracer:
    # better_dataloader = MultiDataLoader([image_data, audio_data])
    # for i, batch in enumerate(better_dataloader):
    #     image, audio = batch

    #     if i >= 3:
    #         break
