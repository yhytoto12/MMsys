import torch
from torch.utils.data import IterableDataset
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from torch.profiler import record_function


class RealTimeMultimodalDataPipe(IterableDataset):
    def __init__(self, read_fn_dict, preprocess_fn_dict):
        self.num_modes = len(preprocess_fn_dict)
        self.modes = list(preprocess_fn_dict.keys())
        self.read_fn_dict = read_fn_dict
        self.preprocess_fn_dict = preprocess_fn_dict

    def __iter__(self):
        return self

    def __next__(self):
        data = {}
        for mode in self.modes:
            with record_function(f'read'):
                data[mode] = self.read_fn_dict[mode]()

        for mode in self.modes:
            with record_function(f'preprocess'):
                data[mode] = self.preprocess_fn_dict[mode](data[mode])

        return data

class RealTimeDataPipe(IterableDataset):
    def __init__(self, read_fn, preprocess_fn, *args, **kwargs):
        self.preprocess_fn = preprocess_fn
        self.read_fn = read_fn
        self.name = kwargs.get('name', '')

    def __iter__(self):
        return self

    def __next__(self):
        with record_function('read'):
            data = self.read_fn()

        with record_function('preprocess'):
            data = self.preprocess_fn(data)

        return data

class MultimodalDataLoadManager:
    def __init__(self, dataloader_dict, feature_extractor_dict, preprocess_only = False):
        self.num_modes = len(dataloader_dict)
        self.modes = list(dataloader_dict.keys())
        self.dataloader_dict = { mode : iter(dataloader) for mode, dataloader in dataloader_dict.items() }
        self.feature_extractor_dict = feature_extractor_dict
        self.preprocess_only = preprocess_only
        # self.executor = ThreadPoolExecutor(max_workers = self.num_modes,
        #                                    thread_name_prefix = 'DataLoadManager',
        #                                    initializer = self._init_thread)

    def _init_thread(self):
        # torch.set_num_threads(1)
        pass

    def fetch(self, mode):
        with record_function('fetch'):
            data = next(self.dataloader_dict[mode])

        with record_function('load to GPU'):
            data = data.cuda()
        if not self.preprocess_only:
            with record_function('extract feature'):
                data = self.feature_extractor_dict[mode](data)
        return data

    def get_data(self):
        features = {}
        with ThreadPoolExecutor(max_workers = self.num_modes,
                                thread_name_prefix = 'DataLoadManager',
                                initializer = self._init_thread) as executor:
        # with ProcessPoolExecutor(max_workers = self.num_modes,
        #                         # thread_name_prefix = 'DataLoadManager',
        #                         initializer = self._init_thread) as executor:
            for mode in self.modes:
                features[mode] = executor.submit(self.fetch, mode)

            features = { mode : v.result() for mode, v in features.items()}

        return features