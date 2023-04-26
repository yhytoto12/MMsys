import os
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchaudio.datasets import YESNO
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from utils import DeviceDataLoader

_SAMPLE_DIR = "_assets"
YESNO_DATASET_PATH = os.path.join(_SAMPLE_DIR, "yes_no")
os.makedirs(YESNO_DATASET_PATH, exist_ok=True)

def load_yesno(device, batch_size=128, val_seed=43):
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    """

    dataset = YESNO(YESNO_DATASET_PATH, download=True)
    #dataset = CIFAR10(root='data/', download=True, transform=transform_train)
    #test_dataset = CIFAR10(root='data/', train=False, transform=transform_test)


    torch.manual_seed(val_seed)
    val_size = 10
    train_size = len(dataset) - 2 * val_size
    test_size = val_size

    # THIS IS JUST FOR DUMMY PURPOSE (THE DATASET DOES NOT HAVE TRAIN/VAL/TEST split)
    train_ds, test_ds, val_ds = random_split(dataset, [train_size, test_size, val_size])


    batch_size=batch_size


    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size*2, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size*2, num_workers=0, pin_memory=True)


    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)

    return train_loader, val_loader, test_loader