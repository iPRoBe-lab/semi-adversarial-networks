#################################################
#             Semi-Adversarial Network          #
# (modified VGG-face for auxiliary face matcher)#
#               iPRoBe lab                      #
#                                               #
#################################################
"""
Extracting VGG-face descriptors for training the SAN model
  (Original model obtained from http://www.robots.ox.ac.uk/~vgg/software/vgg_face/)
"""
import torch
import torch.nn as nn

import numpy as np

use_cuda = True

## read the gray-correction array
with open('../model/vgg_gray_corr_ptch.npz', 'rb') as fp:
    vgg_gray_corr_ptch = np.load(fp)['arr']
vgg_gray_corr_ptch = torch.from_numpy(vgg_gray_corr_ptch)
print(vgg_gray_corr_ptch.size())


class VGGface(nn.Module):
    def __init__(self, num_classes=2):
        super(VGGface, self).__init__()
        if use_cuda:
            self.gray_correction = torch.autograd.Variable(
                    vgg_gray_corr_ptch, requires_grad=False).cuda()
        else:
            self.gray_correction = torch.autograd.Variable(
                    vgg_gray_corr_ptch, requires_grad=False)
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 2622, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.features[0](x)
        x = x - self.gray_correction

        for i in range(1, len(self.features)):
            x = self.features[i](x)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.normalize(x, p=2)

        return x
