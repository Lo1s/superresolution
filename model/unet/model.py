import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
        nn.ReLU(inplace=True)
    )


def crop_img(tensor_img, target_tensor_img):
    target_size = target_tensor_img.size()[2]
    tensor_size = tensor_img.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor_img[:, :, delta:tensor_size - delta, delta:tensor_size - delta]


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        # encoder
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_down_1 = double_conv(3, 64)
        self.conv2_down_2 = double_conv(64, 128)
        self.conv2_down_3 = double_conv(128, 256)
        self.conv2_down_4 = double_conv(256, 512)
        self.conv2_down_5 = double_conv(512, 1024)

        # center
        self.conv2_center = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(1, 1)),
            nn.ReLU(inplace=True)
        )

        # decoder
        self.conv_trans_1 = nn.ConvTranspose2d(1024, 512, kernel_size=(2, 2), stride=(2, 2))
        self.conv2_up_1 = double_conv(1024, 512)
        self.conv_trans_2 = nn.ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2, 2))
        self.conv2_up_2 = double_conv(512, 256)
        self.conv_trans_3 = nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2))
        self.conv2_up_3 = double_conv(256, 128)
        self.conv_trans_4 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))
        self.conv2_up_4 = double_conv(128, 64)
        # output layer
        self.conv1_output = nn.Conv2d(64, 3 * (3 ** 2), kernel_size=(3, 3), padding=(1, 1))
        self.pixel_shuffle = nn.PixelShuffle(3)
        self.output_activation = nn.Sigmoid()

    def forward(self, image):
        # image (batch_size, channels, height, width)
        # encoder
        # 1. block
        x1 = self.conv2_down_1(image)  # copy to decoder part
        x2 = self.max_pool_2x2(x1)
        # 2. block
        x3 = self.conv2_down_2(x2)  # copy to decoder part
        x4 = self.max_pool_2x2(x3)
        # 3. block
        x5 = self.conv2_down_3(x4)  # copy to decoder part
        x6 = self.max_pool_2x2(x5)
        # 4. block
        x7 = self.conv2_down_4(x6)  # copy to decoder part
        x8 = self.max_pool_2x2(x7)
        # 5. block
        x9 = self.conv2_down_5(x8)

        # center
        x_center = self.conv2_center(x9)

        # decoder
        # 1. block
        x = self.conv_trans_1(x_center)
        x7_cropped = crop_img(x7, x)
        x = torch.cat([x, x7_cropped], 1)
        x = self.conv2_up_1(x)
        # 2. block
        x = self.conv_trans_2(x)
        x5_cropped = crop_img(x5, x)
        x = torch.cat([x, x5_cropped], 1)
        x = self.conv2_up_2(x)
        # 3. block
        x = self.conv_trans_3(x)
        x3_cropped = crop_img(x3, x)
        x = torch.cat([x, x3_cropped], 1)
        x = self.conv2_up_3(x)
        # 4. block
        x = self.conv_trans_4(x)
        x1_cropped = crop_img(x1, x)
        x = torch.cat([x, x1_cropped], 1)
        x = self.conv2_up_4(x)
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
    print(output.size())
