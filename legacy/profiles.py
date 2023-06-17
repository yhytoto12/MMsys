import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import torchaudio
import torchvision.models as models
import torch.multiprocessing as mp

from torchvision import transforms
from torchvision.transforms import InterpolationMode

if __name__ == '__main__':
    transform1 = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(224, interpolation=InterpolationMode.BILINEAR)
    ])

    transform2 = torchaudio.transforms.MFCC(sample_rate = 16000, n_mfcc = 40)

    model = models.resnet18()
    model = model.eval()
    # model = torch.compile(model)
    # model = model.cuda()

    input1 = torch.randn(1, 3, 512, 512)
    input2 = torch.randn(1, 1, 16000)

    def trans1(x):
        return transform1(x)

    def trans2(x):
        return transform2(x)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/test'),
        # profile_memory=True,
        record_shapes=True) as prof:
        for i in range(1):

            with record_function(f'transform'):
                x1 = trans1(input1)
                x2 = trans2(input2)

            with record_function(f"[{i}] model inference"):
                model(x1)

            prof.step()