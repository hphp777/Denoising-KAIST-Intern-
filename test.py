#!/usr/bin/python3

import argparse
import sys
import os
import numpy as np
import glob
import gc

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from tqdm import tqdm
from PIL import Image
from utils import Logger

from models import Generator
from datasets import ImageDataset

from unet.unet_model import UNet
from Ect import PSNR

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='C:\\Users\\bispl2219\Desktop\CycleGan\PyTorch-CycleGAN\\noise2clean(npy)\\', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--size', type=int, default=512, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', default=True , help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='C:\\Users\\bispl2219\Desktop\CycleGan\PyTorch-CycleGAN\output\\netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='C:\\Users\\bispl2219\Desktop\CycleGan\PyTorch-CycleGAN\output\\netG_B2A.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = UNet(1,1)
netG_B2A = UNet(1,1)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, 1, opt.size, opt.size)
input_B = Tensor(opt.batchSize, 1, opt.size, opt.size)

# Dataset loader
dataloader = DataLoader(ImageDataset(mode='test'), shuffle=False, batch_size=1)

# PSNR
psnr = PSNR()

###################################

root = 'C:\\Users\\bispl2219\Desktop\CycleGan\PyTorch-CycleGAN\\noise2clean(npy)\\'
mode = 'test'
files_A = sorted(glob.glob(root + mode + 'A' + '\*.npy'))
files_B = sorted(glob.glob(root + mode + 'B' + '\*.npy'))
A_length = len(files_A)
B_length = len(files_B)



###### Testing######

# Create output dirs if they don't exist
if not os.path.exists('output/A'):
    os.makedirs('output/A')
if not os.path.exists('output/B'):
    os.makedirs('output/B')

total_psnr_A = 0
total_psnr_B = 0

for i, batch in enumerate(dataloader):

    gc.collect()
    torch.cuda.empty_cache()

    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))


    # Generate output
    # fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    fake_B = netG_A2B(real_A)
    fake_A = netG_B2A(real_B)

    total_psnr_A += psnr(fake_A.data, real_A.data)
    total_psnr_B += psnr(fake_B.data, real_B.data)
    
    # fake_B = 0.5*(fake_B.data + 1.0)
    # fake_A = 0.5*(fake_A.data + 1.0)

#     print(psnr(fake_A,  real_A))


    # Save image files
    # save_image(fake_A, 'output/A/%04d.png' % (i+1))
    # save_image(fake_B, 'output/B/%04d.png' % (i+1))
    np.save('output/A/'+str(i+1),fake_A.cpu().detach().numpy())
    np.save('output/B/'+str(i+1),fake_B.cpu().detach().numpy())

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

final_psnr_A = total_psnr_A / len(dataloader)
final_psnr_B = total_psnr_B / len(dataloader)

sys.stdout.write('\n')
print("Average Test PSNR(Full Dose) : ", final_psnr_A)
print("Average Test PSNR(Quarter Dose) : ", final_psnr_B)

###################################
