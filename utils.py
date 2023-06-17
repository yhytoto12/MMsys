import numpy as np
import torch.profiler

def get_profile_stats(prof : torch.profiler.profile):
    cpu_times = np.array([
        x.cpu_time for x in filter(
            lambda y : 'ProfilerStep' in y.name,
            prof.events()
        )
    ]) / 1000.0

    avg_time = cpu_times.mean()
    std = cpu_times.std()

    return avg_time, std