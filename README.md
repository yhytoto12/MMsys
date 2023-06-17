# Improved Multimodal Real-Time Data Loading

- Describe our methods
<img src='assets/algorithm.png'>


## Requirements

- Python 3.10
- PyTorch 2.0.1
- CUDA 11.2

## Quick Start
### Install Requirements

``` shell
conda create -n aisys python=3.10
conda activate aisys
pip install -r requirements.txt
```

### How to use MultimodalDataLoadManager
- This is an example for image-audio multimodal system
``` python
from multimodal_dataloader import MultimodalDataLoadManager, RealTimeDataPipe
from realtime_dataloader import DataLoader as RealTimeDataLoader

# Your multimodal model is here
model = model.cuda()
model.eval()

# Create RealTimeDataPipes and RealTimeDataLoaders
datapipes = {}
dataloaders = {}

datapipes['image'] = RealTimeDataPipe(read_image, preprocess_image)
dataloaders['image'] = RealTimeDataLoader(datapipes['image'], num_workers = 1, batch_size = 1, shuffle = False)

datapipes['audio'] = RealTimeDataPipe(read_audio, preprocess_audio)
dataloaders['image'] = RealTimeDataLoader(datapipes['audio'], num_workers = 1, batch_size = 1, shuffle = False)

# Define feature extractor dictionary
feature_extractors = {}
feature_extractors['image'] = model.extract_feature_image
feature_extractors['audio'] = model.extract_feature_audio

# Create MultimodalDataLoadManager
manager = MultimodalDataLoadManager(
    dataloaders,
    feature_extractors,
)

# Inference loop
for step in range(5):
    features = manager.get_data()

    output = model.fusion(features)
    ...
```


## Experiments
### Run Synthetic Experiments
``` shell
CUDA_VISIBLE_DEVICE=0 python exp_synthetic.py --config [config_file]
```
All configs for the synthetic experiments are in [configs/synthetic](configs/synthetic)

### Run AVSR Experiments

- TODO (Keighley)

``` shell
CUDA_VISIBLE_DEVICE=0 python exp_avsr.py --config [config_file]
```
All configs for the synthetic experiments are in [configs/avsr](configs/synthetic)