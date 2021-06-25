import torch
import torch.nn as nn
import torch.nn.functional as F


# to minimize checkerboard pattern
class ResizeConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResizeConvolution, self).__init__()
        self.resize_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        )

    def forward(self, x):
        return self.resize_conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, scale=3, features=None):
        super(UNet, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ups = nn.ModuleList()

        # Encoder/Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Center/Bottleneck part
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Decoder/Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                ResizeConvolution(feature*2, feature)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.conv1_output = nn.Conv2d(features[0], scale * (out_channels ** 2), kernel_size=(3, 3), padding=(1, 1))
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        steps = 2
        for idx in range(0, len(self.ups), steps):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//steps]
            if x.shape != skip_connection.shape:
                skip_connection = F.interpolate(skip_connection, size=x.shape[2:])
            x = torch.cat([x, skip_connection], dim=1)
            x = self.ups[idx+1](x)

        x = self.conv1_output(x)
        x = self.pixel_shuffle(x)
        x = self.output_activation(x)
        return x


if __name__ == '__main__':
    test_image = torch.rand((64, 3, 64, 64))
    model = UNet()
    print(model)
    output = model(test_image)
    print(output)
    print(output.size())  # ([64, 3, 192, 192])
