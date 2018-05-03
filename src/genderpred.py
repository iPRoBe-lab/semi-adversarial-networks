############################################
#          Semi-Adversarial Network        #
#        (auxiliary gender classifier)     #
#               iPRoBe lab                 #
#                                          #
############################################

import torch.nn as nn


class GenderPredictor(nn.Module):
    def __init__(self, num_classes=2):
        super(GenderPredictor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Dropout2d(p=0.25),
            nn.AvgPool2d(kernel_size=(7, 7))
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.5),
            # nn.Linear(64 * 7 * 7, 100),
            # nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, 2),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
