import lightning as L
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.model = nn.Sequential(
        #     nn.BatchNorm2d(in_channels),
        #     nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
        #     nn.LeakyReLU()
        # )
        self.model = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.model(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.model(x)

class UNetBlock_Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            DownSample(in_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.model(x)

class UNetBlock_Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            UpSample(in_channels * 2, out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x, residual):
        xr = torch.cat([x, residual], dim=1)
        return self.model(xr)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        HID_SIZE = 8
        self.pre = nn.Sequential(
            nn.Conv2d(3, HID_SIZE, 3, padding=1),
            nn.BatchNorm2d(HID_SIZE),
            nn.LeakyReLU(),
            nn.Conv2d(HID_SIZE, HID_SIZE, 3, padding=1),
            nn.BatchNorm2d(HID_SIZE),
            nn.LeakyReLU()
        )
        self.encoders = nn.ModuleList([
            UNetBlock_Down(HID_SIZE, HID_SIZE * 2),
            UNetBlock_Down(HID_SIZE * 2, HID_SIZE * 4),
            UNetBlock_Down(HID_SIZE * 4, HID_SIZE * 8),
            UNetBlock_Down(HID_SIZE * 8, HID_SIZE * 16)
        ])
        
        self.processor = nn.Sequential(
            nn.Conv2d(HID_SIZE * 8, HID_SIZE * 8, 3, padding=1),
            nn.BatchNorm2d(HID_SIZE * 8),
            nn.LeakyReLU(),
            nn.Conv2d(HID_SIZE * 8, HID_SIZE * 8, 3, padding=1),
            nn.BatchNorm2d(HID_SIZE * 8),
            nn.LeakyReLU()
        )

        self.decoders = nn.ModuleList([
            UNetBlock_Up(HID_SIZE * 16, HID_SIZE * 8),
            UNetBlock_Up(HID_SIZE * 8, HID_SIZE * 4),
            UNetBlock_Up(HID_SIZE * 4, HID_SIZE * 2),
            UNetBlock_Up(HID_SIZE * 2, HID_SIZE)
        ])
        self.post = nn.Sequential(
            nn.Conv2d(HID_SIZE, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, x):
        features = []
        x = self.pre(x)
        for encoder in self.encoders:
            x = encoder(x)
            features.append(x)
        for decoder in self.decoders:
            x = decoder(x, features.pop())

        return self.post(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        HID_SIZE = 4 # smaller, not that powerful.
        self.model = nn.Sequential(
            nn.Conv2d(6, HID_SIZE, 3, padding=1),
            nn.BatchNorm2d(HID_SIZE),
            nn.LeakyReLU(),
            nn.Conv2d(HID_SIZE, HID_SIZE, 3, padding=1),
            nn.BatchNorm2d(HID_SIZE),
            nn.LeakyReLU(),
            DownSample(HID_SIZE, HID_SIZE),
            DownSample(HID_SIZE, HID_SIZE * 2),
            DownSample(HID_SIZE * 2, HID_SIZE * 2),
            DownSample(HID_SIZE * 2, HID_SIZE * 4),
            DownSample(HID_SIZE * 4, HID_SIZE * 4),
            DownSample(HID_SIZE * 4, HID_SIZE * 8),
            DownSample(HID_SIZE * 8, HID_SIZE * 8),
            DownSample(HID_SIZE * 8, HID_SIZE * 8),
            nn.Flatten(),
            nn.Linear(HID_SIZE * 8, HID_SIZE * 4),
            nn.LeakyReLU(),
            nn.Linear(HID_SIZE * 4, HID_SIZE * 2),
            nn.LeakyReLU(),
            nn.Linear(HID_SIZE * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb, seg):
        y = self.model(torch.cat([rgb, seg], dim=1))
        return y
