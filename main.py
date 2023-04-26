import torch
from utils import get_default_device, to_device, plot_losses, plot_accuracies
from dataset.load_cifar import load_cifar
from dataset.load_yesno import load_yesno
from model.dla.dla import DLA
from model.densenet.densenet import DenseNet121, DenseNet169, DenseNet201, DenseNet161, densenet_cifar
from model.resnext.resnext import ResNeXt29_2x64d, ResNeXt29_4x64d, ResNeXt29_8x64d, ResNeXt29_32x4d
from model.HCGNet.hcgnet import HCGNet_A1, HCGNet_A2, HCGNet_A3
from model.efficientnetv2.efficientnetv2 import effnetv2_s, effnetv2_m, effnetv2_l
from train_model import fit
from test_model import evaluate
import argparse

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar-10' , help='dataset specification')
parser.add_argument('--n_epoch', type=int, default='350' , help='number of epoch')
parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay rate")
parser.add_argument('--milestones', type=list, default=[150,225], help="milestones for lr scheduler")
parser.add_argument('--gamma', type=float, default=0.1, help="gamma for lr scheduler")
args = parser.parse_args()


# CPU or CUDA
device = get_default_device()
print(device)

# Dataset
# Image
image_train_loader, image_val_loader, image_test_loader = load_cifar(device)
# Audio
train_audio_loader, val_audio_loader, test_audio_loader = load_yesno(device)
# Audio
train_audio_loader, val_audio_loader, test_audio_loader = load_yesno(device)


# Model to device
model = to_device(HCGNet_A1(), device)

# Initial loss and accuracy (epoch 0)
history = [evaluate(model, image_val_loader)]
print(history)

# Train the model (including validation at each epoch)
history += fit(args.n_epoch, args.lr, args.momentum, args.weight_decay, args.milestones, args.gamma, model, image_train_loader, image_val_loader)

# Test the model
#plot_losses(history)
#plot_accuracies(history)
print(evaluate(model, image_test_loader))

PATH = './hcg_for_cifar.pth'
torch.save(model.state_dict(), PATH)
