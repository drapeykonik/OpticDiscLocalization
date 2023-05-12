import torch
from torch import nn


class VGGRegressor(nn.Module):
    """
    VGG19 with additional layers CNN for regression problem solving

    Parameters
    ----------
    in_channels: int
        Number of input channels of the processed image.
        Model could be used with color channel choosing transformation, so
        this parameter give a opportunity to specify the channels amount
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels

        # 512x512
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )
        self.pool0 = nn.MaxPool2d(kernel_size=2)
        # 256x256
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # 128x128
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # 64x64
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        # 32x32
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        # 16x16
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        # 8x8
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )
        self.pool6 = nn.MaxPool2d(kernel_size=2)
        # 4x4
        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )
        self.pool7 = nn.MaxPool2d(kernel_size=2)
        # 2x2
        self.fc0 = nn.Linear(in_features=2 * 2 * 512, out_features=2048)
        self.fc1 = nn.Linear(in_features=2048, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward path of the model

        Parameters
        ----------
        x: torch.Tensor
            Batch of the images for location prediction
        """

        x = self.pool0(self.conv0(x))
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        x = self.pool5(self.conv5(x))
        x = self.pool6(self.conv6(x))
        x = self.pool7(self.conv7(x))
        x = x.view(-1, 2 * 2 * 512)
        x = nn.functional.relu(self.fc0(x))
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return x
