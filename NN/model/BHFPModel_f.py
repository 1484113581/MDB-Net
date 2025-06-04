# -*- coding:utf-8 -*-
# @Time      :2025/2/22 19:35
# @Author    :Chen
import torch
from torch import nn

from NN.util.useful_modules import *
from torchsummary import summary
# ********** 通用部分 *************************************
# 卷积，通道数变化
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, activation='relu'):
        super(unetConv2, self).__init__()
        self.activation = activation

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False), nn.BatchNorm2d(out_size), nn.ReLU()  #
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_size, out_size, 3, 1, 1, bias=False), nn.BatchNorm2d(out_size)
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.activation == 'relu':
            out = self.relu(out)
        else:
            out = self.sigmoid(out)
        return out


# 转置卷积，通道数减少2倍，长宽加倍
class TransConv2(nn.Module):
    def __init__(self, in_size, out_size):
        super(TransConv2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False), nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_size, out_size, 3, 1, 1, bias=False), nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(out_size, out_size, kernel_size=2, stride=2, bias=False),  # 长宽增倍
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.upsample(out)
        return out


# 转置卷积，通道数减少4倍，长宽加倍
class TransConv4(nn.Module):
    def __init__(self, in_size, out_size):
        super(TransConv4, self).__init__()
        mid_size = in_size // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, mid_size, 3, 1, 1, bias=False), nn.BatchNorm2d(mid_size),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_size, out_size, 3, 1, 1, bias=False), nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_size, out_size, 3, 1, 1, bias=False), nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(out_size, out_size, kernel_size=2, stride=2, bias=False),  # 长宽增倍
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.upsample(out)
        return out


class ZoomModule(nn.Module):
    def __init__(self,in_size):
        super(ZoomModule, self).__init__()
        # 先上采样再下采样
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2, bias=False),  # 长宽增倍
            # nn.BatchNorm2d(in_size),
            nn.ReLU(inplace=True),
            unetConv2(in_size=in_size,out_size=16),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels=in_size, out_channels=16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=16),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.conv_x(x)

        out = self.layer1(x)
        out += identity
        out = self.relu(out)
        return out
# *********编码部分*************************************
class First_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(First_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, bias=False,padding=1),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=5, stride=1, bias=False,
                               padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, bias=False,padding=1),
            nn.BatchNorm2d(num_features=out_channel),
        )

        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(negative_slope=0.01)

        self.x_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=out_channel)
        )

    def forward(self, x):
        identity = self.x_conv(x)

        out = self.conv1(x)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        out += identity  # 残差连接
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, stride=2):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride,
                               bias=False,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channel)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(negative_slope=0.01)

        self.x_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(num_features=out_channel)
        )

    def forward(self, x):
        identity = x  # 将原始输入暂存为shortcut的输出
        identity = self.x_conv(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity  # 残差连接
        out = self.relu(out)

        return out


# s2编码部分
class UencoderOptical(nn.Module):
    def __init__(self, in_channel):
        super(UencoderOptical, self).__init__()
        self.in_channel = in_channel

        # 第一层
        self.layer1 = nn.Sequential(
            ZoomModule(in_size=in_channel),
            MultiScaleFusion(in_channels=16,reduction_ratio=4),
            First_block(16, 16)
        )
        self.layer2 = Bottleneck(16, 32)
        self.layer21 = TransConv2(32,16)
        self.layer3 = nn.Sequential(
            Bottleneck(32, 64),
            ChannelMultiScaleSpatialAttention(in_channels=64)
        )
        self.layer4 = Bottleneck(64, 128)
        # self.layer5 = Bottleneck(128, 256)

    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv21 = self.layer21(conv2)
        conv3 = self.layer3(conv2)
        conv4 = self.layer4(conv3)
        # conv5 = self.layer5(conv4)

        return conv1, conv2,conv21, conv3,conv4


# s1编码部分
class Uencoder(nn.Module):
    def __init__(self, in_channel):
        super(Uencoder, self).__init__()
        self.in_channel = in_channel

        # 第一层不下采样
        self.layer1 = nn.Sequential(
            ZoomModule(in_size=in_channel),
            MultiScaleFusion(in_channels=16, reduction_ratio=4),
            First_block(16, 16)
        )
        self.layer2 = Bottleneck(16, 32)
        self.layer21 = TransConv2(32,16)
        self.layer3 = nn.Sequential(
            Bottleneck(32, 64),
            ChannelMultiScaleSpatialAttention(in_channels=64)
        )
        self.layer4 = Bottleneck(64, 128)
        # self.layer5 = Bottleneck(128, 256)

    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv21 = self.layer21(conv2)
        conv3 = self.layer3(conv2)
        conv4 = self.layer4(conv3)
        # conv5 = self.layer5(conv4)

        return conv1, conv2,conv21, conv3,conv4

# ******* 解码部分 ********************************************
class UdecoderOptical(nn.Module):
    def __init__(self):
        super(UdecoderOptical, self).__init__()

        self.firstConv = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            TransConv2(128, 64),
            unetConv2(64, 64)
        )
        # self.aff1 = AdaptiveFeatureFusion(low_dim=64,high_dim=64)
        self.up1 =  nn.Sequential(
            TransConv2(128, 64),
            unetConv2(64,32)
        )
        # self.aff2 = AdaptiveFeatureFusion(low_dim=32, high_dim=32)
        self.up3 = nn.Sequential(
            TransConv2(64, 32),
        )
        # self.aff3 = AdaptiveFeatureFusion(low_dim=32, high_dim=32)

        self.conv1 = unetConv2(64, 16, activation='relu')
        self.conv2 = unetConv2(16, 16, activation='relu')

        self.conv3 = unetConv2(16, 1, activation='sigmoid')


    def forward(self, conv1, conv2,conv21, conv3,merge):
        optical1 = self.up2(merge)
        optical1 =self.up1(torch.cat((conv3,optical1),dim=1))

        optical2 = self.up3(torch.cat((conv2,optical1),dim=1))

        optical3 = self.conv1(torch.cat((conv1,conv21,optical2), dim=1))
        optical4 = self.conv2(optical3)

        out = self.conv3(optical4)

        return out,optical1,optical2,optical3,optical4


class UdecoderSar(nn.Module):
    def __init__(self):
        super(UdecoderSar, self).__init__()

        # self.aff1 = AdaptiveFeatureFusion(low_dim=64, high_dim=64)
        # self.aff2 = AdaptiveFeatureFusion(low_dim=32, high_dim=32)
        # self.aff3 = AdaptiveFeatureFusion(low_dim=32, high_dim=32)
        self.aff4 = AdaptiveFeatureFusion(low_dim=32, high_dim=32)
        self.aff5 = AdaptiveFeatureFusion(low_dim=32, high_dim=32)
        self.aff6 = AdaptiveFeatureFusion(low_dim=16, high_dim=16)
        self.aff7 = AdaptiveFeatureFusion(low_dim=16, high_dim=16)

        self.conv1 = nn.Sequential(
            TransConv2(128, 64),
            unetConv2(64, 64)
        )
        self.conv2 = nn.Sequential(
            TransConv2(128, 64),
            unetConv2(64, 32)
        )
        self.conv22 = nn.Sequential(
            unetConv2(64, 32)
        )
        self.conv3 = TransConv2(64, 32)

        self.conv33 = nn.Sequential(
            unetConv2(64, 32)
        )

        self.conv4 = nn.Sequential(
            unetConv2(64, 16),
            unetConv2(16, 16, activation='relu')
        )

        self.conv5 = unetConv2(32, 16)
        self.conv6 = unetConv2(32, 16)
        self.conv7 = nn.Sequential(
            unetConv2(32, 16, activation='relu'),
            unetConv2(16, 1, activation='relu')
        )

    def forward(self, x, optical1,optical2,optical3,optical4,conv1, conv2,conv21,conv3):
        out = self.conv1(x)
        out = self.conv2(torch.cat((conv3,out),dim=1))
        out = self.conv22(self.aff4(optical1,out))
        out = self.conv3(torch.cat((conv2,out),dim=1))
        out = self.conv33(self.aff5(optical2,out))
        out = self.conv4(torch.cat((out,conv21,conv1), dim=1))
        out = self.conv5(self.aff6(optical3,out))
        out = self.conv7(self.aff7(optical4,out))
        return out

class BHFPModel_f(nn.Module):
    def __init__(self):
        super(BHFPModel_f, self).__init__()
        self.relu = nn.ReLU()

        # 编码器
        self.encoder_optical = UencoderOptical(in_channel=4)
        self.encoder_s1sar = Uencoder(in_channel=5)

        self.fusion = CrossModalChannelAttention(opt_channels=128, sar_channels=128)

        # 解码器
        self.udecoder_optical = UdecoderOptical()
        self.udecoder_sar = UdecoderSar()

    def forward(self, x):
        # 处理一下x 把光学的通道和sar的通道分开
        optical, s1sar= torch.split(x, [4,5], dim=1)

        # 四个分支分别 encoder
        conv1_optical, conv2_optical,conv21_optical, conv3_optical,conv4_optical = self.encoder_optical(optical)
        conv1_s1sar, conv2_s1sar,conv21_s1sar, conv3_s1sar,conv4_s1sar = self.encoder_s1sar(s1sar)

        merge_opt,merge_sar = self.fusion(conv4_optical, conv4_s1sar)

        # 解码部分
        fp, optical1,optical2,optical3, optical4 = self.udecoder_optical(conv1_optical, conv2_optical,conv21_optical,conv3_optical,merge_opt)
        bh = self.udecoder_sar(merge_sar,optical1,optical2,optical3,optical4,conv1_s1sar, conv2_s1sar,conv21_s1sar, conv3_s1sar)

        return bh, fp

if __name__ == '__main__':
    # model = BHFPModel_f()
    # input = torch.ones((8, 9, 128, 128))
    # output = model(input)
    # print(output[1].shape)

    model = BHFPModel_f()
    # print(model)
    summary(model, (9, 128, 128), batch_size=1, device="cpu")