import torch.nn as nn
from functools import reduce

def forward_block(in_feat, out_feat):
    return nn.Sequential(
        nn.BatchNorm2d(in_feat),
        nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(out_feat),
        nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    )
 
def backward_block(in_feat, out_feat, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.BatchNorm2d(in_feat),
        nn.ConvTranspose2d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(out_feat),
        nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(inplace=True),
    )

class FullyConvNetwork(nn.Module):
    UP_PASS = [3, 64, 64]
    DOWN_PASS = [64, 64, 8]

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        up = [forward_block(in_feat, out_feat) for in_feat, out_feat in zip(self.UP_PASS, self.UP_PASS[1:])]
        down = [backward_block(in_feat, out_feat) for in_feat, out_feat in zip(self.DOWN_PASS, self.DOWN_PASS[1:])]
        ### FILL: add more CONV Layers
        ### None: since last layer outputs RGB channels, may need specific activation function
        self.up = nn.Sequential(*up)
        self.down = nn.Sequential(*down)

        self.final = nn.Sequential(
            nn.Conv2d(self.DOWN_PASS[-1], self.DOWN_PASS[-1], kernel_size=1, stride=1),
            nn.BatchNorm2d(self.DOWN_PASS[-1]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.DOWN_PASS[-1], self.DOWN_PASS[-1], kernel_size=1, stride=1),
            nn.BatchNorm2d(self.DOWN_PASS[-1]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.DOWN_PASS[-1], 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh(),
        )
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    def forward(self, x):
        # Encoder forward pass
        y = self.up(x)
        y = self.down(y)
        y = self.final(y)
        return y
