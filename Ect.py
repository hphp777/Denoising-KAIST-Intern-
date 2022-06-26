# full_dose = trainA, quarter_dose = trainB

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from piqa import PSNR
import torch
from torch import Tensor
from unet.unet_model import UNet
import torchvision.transforms as transforms
from torch.autograd import Variable

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

# Missing : 703 753 815 868 910 911 1004 1150 1166 1170 1187 1203 1206
def savePNG():
    for i in range(1542,1829):
        try: 
            np_array = np.load('C:/Users/bispl2219/Desktop/CycleGan/noise2clean/trainA/' + str(i) +'.npy')
            plt.imsave('C:/Users/bispl2219/Desktop/CycleGan/data/trainA/' + str(i) + '.png', np_array)

            np_array = np.load('C:/Users/bispl2219/Desktop/CycleGan/noise2clean/trainB/' + str(i) +'.npy')
            plt.imsave('C:/Users/bispl2219/Desktop/CycleGan/data/trainB/' + str(i) + '.png', np_array)
        except:
            continue

    for i in range(1373,3840):
        try: 
            np_array = np.load('C:/Users/bispl2219/Desktop/CycleGan/noise2clean/testA/' + str(i) +'.npy')
            plt.imsave('C:/Users/bispl2219/Desktop/CycleGan/data/testA/' + str(i) + '.png', np_array)
        except:
            continue


    for i in range(703,3840):
        try:
            np_array = np.load('C:/Users/bispl2219/Desktop/CycleGan/noise2clean/testB/' + str(i) +'.npy')
            plt.imsave('C:/Users/bispl2219/Desktop/CycleGan/data/testB/' + str(i) + '.png', np_array)
        except:
            continue

def mse(x: Tensor, y: Tensor):
    r"""Returns the Mean Squared Error (MSE) between :math:`x` and :math:`y`.

    .. math::
        \text{MSE}(x, y) = \frac{1}{\text{size}(x)} \sum_i (x_i - y_i)^2

    Args:
        x: An input tensor, :math:`(N, *)`.
        y: A target tensor, :math:`(N, *)`.

    Returns:
        The MSE vector, :math:`(N,)`.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = mse(x, y)
        >>> l.size()
        torch.Size([5])
    """

    return ((x - y) ** 2).view(x.size(0), -1).mean(dim=-1)

def sample():
    img = np.load('output/A/27.npy')
    img_original = np.load('PyTorch-CycleGAN/noise2clean(npy)/testA/1374.npy')
    change = img[0][0] - img_original
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 6))
    ax1.imshow(img[0][0],cmap='gray')
    ax1.set_title('(1) Fake Full Dose')
    ax2.imshow(img_original, cmap='gray')
    ax2.set_title('(2) Original Quarter Dose')
    ax3.imshow(change, cmap='gray')
    ax3.set_title('(1) - (2)')
    plt.show()

    psnr = PSNR()
    l1 = psnr(torch.Tensor(img_original),torch.Tensor(img[0][0]))
    print(l1.item())

def interupt():
    netG_A2B = UNet(1,1)
    netG_B2A = UNet(1,1)

    weight_A2B = 'C:\\Users\\bispl2219\Desktop\CycleGan\PyTorch-CycleGAN\output\\netG_A2B.pth'
    weight_B2A = 'C:\\Users\\bispl2219\Desktop\CycleGan\PyTorch-CycleGAN\output\\netG_B2A.pth'

    netG_A2B.load_state_dict(torch.load(weight_A2B))
    netG_B2A.load_state_dict(torch.load(weight_B2A))

    netG_A2B.eval()
    netG_B2A.eval()

    size = 512

    transform = transforms.Compose([ 
                transforms.ToPILImage(),
                transforms.ToTensor(),
                ])

    imgA = np.load('PyTorch-CycleGAN/noise2clean(npy)/testA/1374.npy')
    imgA = (imgA - 0.0192) / (0.0192 * 1000)
    imgA = (imgA - imgA.min()) / (imgA.max() - imgA.min())
    print(imgA.min(), imgA.max())
    imgB = np.load('PyTorch-CycleGAN/noise2clean(npy)/testB/1374.npy')
    imgB = (imgB - 0.0192) / (0.0192 * 1000)
    imgB = (imgB - imgB.min()) / (imgB.max() - imgB.min())

    input_B = Tensor(1, 1, size, size)
    input_A = Tensor(1, 1, size, size)
    real_B = Variable(input_B.copy_(transform(imgB)))
    real_A = Variable(input_A.copy_(transform(imgA)))

    fake_A = netG_B2A(real_B).cpu().detach().numpy()[0][0]
    fake_B = netG_A2B(real_A).cpu().detach().numpy()[0][0]

    change = imgA - fake_A

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 6))
    ax1.imshow(imgA,cmap='gray')
    ax1.set_title('(1) Original Quarter Dose')
    ax2.imshow(fake_A, cmap='gray')
    ax2.set_title('(2) Fake Quarter Dose')
    ax3.imshow(change, cmap='gray', vmin = change.min(), vmax = change.max())
    ax3.set_title('(1) - (2)')
    plt.show()

    psnr = PSNR()
    l1 = psnr(torch.Tensor(imgB),torch.Tensor(imgA))

    print(l1)



def value_range():
    img = np.load('PyTorch-CycleGAN/noise2clean(npy)/trainA/1.npy')
    # print(img)
    print("min = ", img.min())
    print("max = ", img.max())

interupt()