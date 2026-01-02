from models import MMFNet
import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description='SparseTSF & other models for Time Series Forecasting')

# basic config

parser.add_argument('--is_transfering', type=int, default=0, help='status')


# Data loader
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M: multivariate predict multivariate, S: univariate predict univariate, MS: multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s: secondly, t: minutely, h: hourly, d: daily, b: business days, w: weekly, m: monthly]')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# Forecasting task
parser.add_argument('--seq_len', type=int, default=1440, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=720, help='prediction sequence length')

# Hyperparameters for MixLinear, SparseTSF, PatchTST, and other models (unchanged)

# Optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=4, help='gpu')
parser.add_argument('--use_multi_gpu', type=int, default=0, help='use multiple gpus')
parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help='device ids of multiple gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

# Set random seeds
random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)
if args.use_gpu:
    torch.cuda.manual_seed_all(2023)

args.enc_in=7
args.period_len=24
args.lpf=19
args.alpha=0.95

# Setup GPU
args.use_gpu = torch.cuda.is_available() and args.use_gpu
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
    torch.cuda.set_device(args.gpu)

# Print configuration
print('Args in experiment:')
print(args)



model = MMFNet.Model(args).float()


# Load the checkpoint
checkpoint_path = os.path.join(args.checkpoints, 'ETTm2_1440_720_MMFNet_ETTm2_ftM_sl1440_pl720_test_0.01_129_0_seed2023/checkpoint.pth')  # Specify the correct path
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
else:
    print(f"Checkpoint not found at {checkpoint_path}")

# Ensure model is on the correct device
if args.use_gpu:
    model = model.to(args.gpu)

# Example of accessing model components
# print(f"Model TLinear1: {model.TLinear1.weight.data if hasattr(model, 'TLinear1') else 'Not found'}")
# print(f"Model TLinear2: {model.TLinear2.weight.data if hasattr(model, 'TLinear2') else 'Not found'}")

# print(f"Model FLinear1: {model.FLinear1.weight.data if hasattr(model, 'FLinear1') else 'Not found'}")
# print(f"Model FLinear2: {model.FLinear2.weight.data if hasattr(model, 'FLinear2') else 'Not found'}")
# print(f"Model FLinear2: {model.FLinear3.weight.data if hasattr(model, 'FLinear3') else 'Not found'}")
import matplotlib.pyplot as plt
import torch


# Function to normalize weights to the range [0, 1]
def normalize_weights(weight_data):
    # Convert weight data to a numpy array (if it's not already)
    weight_np = weight_data.cpu().detach().numpy()

    # Normalize to the range [0, 1]
    weight_min = weight_np.min()
    weight_max = weight_np.max()

    # Avoid division by zero if the min and max are the same
    if weight_max != weight_min:
        normalized_weight = (weight_np - weight_min) / (weight_max - weight_min)
    else:
        # If all values are the same, set to 0 (or 1, or any constant value)
        normalized_weight = weight_np * 0

    return normalized_weight


# Function to plot the normalized weight matrix
def plot_weights(weight_data, title):
    # Normalize the weights
    normalized_weight = normalize_weights(weight_data)

    # Plot the normalized weight matrix using imshow
    plt.figure(figsize=(9, 6))
    plt.imshow(normalized_weight, cmap='Reds', aspect='auto',interpolation="nearest",vmin=0.4)  # You can change the colormap
    plt.colorbar()  # Show a color bar
    plt.title(title)
    # plt.show()

    plt.savefig(title+".pdf")



# # Check if the layers exist and print their weights, then visualize
# if hasattr(model, 'FLinear1'):
#     print(f"FLinear1 weights:\n{model.FLinear1.weight.data}")
#     plot_weights(model.FLinear1.weight.data, "FLinear1 Normalized Weights")
# else:
#     print("FLinear1 not found")
#
if hasattr(model, 'FLinear2'):
    print(f"FLinear2 weights:\n{model.FLinear2.weight.data}")
    plot_weights(model.FLinear2.weight.data, "FLinear2 Normalized Weights")
else:
    print("FLinear2 not found")

if hasattr(model, 'FLinear3'):
    print(f"FLinear3 weights:\n{model.FLinear3.weight.data}")
    plot_weights(model.FLinear3.weight.data, "FLinear3 Normalized Weights")
else:
    print("FLinear3 not found")



# Check if the layers exist and print their weights, then visualize
# if hasattr(model, 'mask1'):
#     print(f"mask1 weights:\n{model.mask1.data}")
#     plot_weights(model.mask1.data.unsqueeze(0), "mask1 Weights")
# else:
#     print("mask1 not found")
#
# if hasattr(model, 'mask2'):
#     print(f"mask2 weights:\n{model.mask2.data}")
#     plot_weights(model.mask2.data.unsqueeze(0), "mask2 Weights")
# else:
#     print("mask2 not found")
#
# if hasattr(model, 'mask3'):
#     print(f"mask3 weights:\n{model.mask3.data}")
#     plot_weights(model.mask3.data.unsqueeze(0), "mask3 Weights")
# else:
#     print("mask3 not found")
