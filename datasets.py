import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class ImageDataset(Dataset):
    def __init__(self,transforms_=None, unaligned=False, mode='train'):
        # self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        root = 'C:\\Users\\bispl2219\Desktop\CycleGan\PyTorch-CycleGAN\\noise2clean(npy)\\'
        self.size = 128
        # self.files_A = os.listdir(root + mode + 'A')
        self.files_A = sorted(glob.glob(root + mode + 'A' + '\*.npy'))
        self.files_B = sorted(glob.glob(root + mode + 'B' + '\*.npy'))

    def __getitem__(self, index):
        # imgA = Image.open(self.files_A[index % len(self.files_A)])
        imgA = np.load(self.files_A[index % len(self.files_A)])
        imgA = (imgA - 0.0192) / (0.0192 * 1000)
        # imgA = (imgA - imgA.min()) / (imgA.max() - imgA.min())

        transform = transforms.Compose([ 
                transforms.ToPILImage(),
                # transforms.Resize(int(self.size*1.12), Image.BICUBIC), 
                transforms.RandomCrop(self.size), 
                # transforms.Resize([256,256]),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ])

        item_A = transform(imgA)

        if self.unaligned:
            imgB = np.load(self.files_B[random.randint(0, len(self.files_B) - 1)])
            imgB = (imgB - 0.0192) / (0.0192 * 1000)
            # imgB = (imgB - imgB.min()) / (imgB.max() - imgB.min())
            item_B = transform(imgB)
        else:
            imgB = np.load(self.files_B[index % len(self.files_B)])
            imgB = (imgB - 0.0192) / (0.0192 * 1000)
            # imgB = (imgB - imgB.min()) / (imgB.max() - imgB.min())
            item_B = transform(imgB)

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))