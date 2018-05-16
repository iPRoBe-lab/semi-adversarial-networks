############################################
#          Semi-Adversarial Network        #
#          (convolutional autoencoder)     #
#               iPRoBe lab                 #
#                                          #
############################################

import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.autoencoder = nn.Sequential(
            ## Encoder
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 12, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),

            ## Decoder
            nn.Conv2d(12, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        self.protocombiner = nn.Sequential(
            nn.Conv2d(131, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, imgs, same_proto, oppo_proto):

        x = torch.cat([imgs, same_proto], dim=1)
        x = self.autoencoder(x)

        rec_same = torch.cat([x, same_proto], dim=1)
        rec_oppo = torch.cat([x, oppo_proto], dim=1)

        return self.protocombiner(rec_same), self.protocombiner(rec_oppo)
