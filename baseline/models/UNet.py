# U-Net baseline will operate on an input with 14/22 channels
# It output is always two channels!

from torch import nn as nn
import torch
import torch.nn.functional as F

BASE_FILTER = 16  # has to be divideable by 4, originally 16

def batch_norm(
    out_channels, dim=2
):  # Does not have the same parameters as the original batch normalization used in the tensorflow 1.14 version of this project
    if dim == 2:
        bnr = torch.nn.BatchNorm2d(out_channels)
    else:
        bnr = torch.nn.BatchNorm3d(out_channels)
    return bnr

# 1x2x2 max pool function
def pool(x, dim=2):
    if dim == 2:
        return F.max_pool2d(x, kernel_size=[2, 2], stride=[2, 2])
    else:
        return F.max_pool3d(x, kernel_size=[1, 2, 2], stride=[1, 2, 2])

# 3x3x3 convolution module
def Conv(in_channels, out_channels, filter_sizes=3, dim=2):
    if dim == 2:
        return nn.Conv2d(
            in_channels, out_channels, filter_sizes, padding=(filter_sizes - 1) // 2
        )
    else:
        return nn.Conv3d(
            in_channels, out_channels, filter_sizes, padding=(filter_sizes - 1) // 2
        )

# Activation function
def activation(x):
    # Simple relu
    return F.relu(x, inplace=True)

# Channel concatenation function
def concat(prev_conv, up_conv):
    assert prev_conv.dim() == 4 and up_conv.dim() == 4, 'for now, we only support 4D tensors!'
    if prev_conv.shape != up_conv.shape:
        pad_ = (0, 0, 1, 0)
        p_c_s = prev_conv.size()
        u_c_s = up_conv.size()
        prev_conv_crop = prev_conv[:,
                         :,
                         ((p_c_s[2] - u_c_s[2]) // 2): u_c_s[2] + ((p_c_s[2] - u_c_s[2]) // 2),
                         ((p_c_s[3] - u_c_s[3]) // 2): u_c_s[3] + ((p_c_s[3] - u_c_s[3]) // 2)
                         ]
        # up_concat = torch.cat((prev_conv_crop, up_conv), dim=1)
        up_concat = torch.cat((prev_conv, F.pad(up_conv, pad_,  mode='replicate')), dim=1)
        return up_concat

    return torch.cat((prev_conv, up_conv), dim=1)

# 1x2x2 nearest-neighbor upsample function
def upsample(x, dim=2):
    if dim == 2:
        return F.interpolate(x, scale_factor=[2, 2], mode="bilinear")
    else:
        return F.interpolate(x, scale_factor=[1, 2, 2], mode="trilinear")


# Conv Batch Relu module
class ConvBatchRelu(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes=3):
        super(ConvBatchRelu, self).__init__()
        self.conv = Conv(in_channels, out_channels, filter_sizes)
        self.bnr = batch_norm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = activation(self.bnr(x))
        return x


class UNet(nn.Module):
    def __init__(
            self, in_channels=14, args=None
    ):
        super(UNet, self).__init__()

        self.dim = 2

        ic = in_channels
        oc = 2

        # Number of channels per layer
        l0_size = BASE_FILTER * 2
        l1_size = BASE_FILTER * 4
        l2_size = BASE_FILTER * 8

        # Convolutions
        self.enc_l0_0 = ConvBatchRelu(ic, l0_size)
        self.enc_l0_1 = ConvBatchRelu(l0_size, l0_size)
        self.enc_l1_0 = ConvBatchRelu(l0_size, l1_size)
        self.enc_l1_1 = ConvBatchRelu(l1_size, l1_size)
        self.enc_l2_0 = ConvBatchRelu(l1_size, l2_size)
        self.enc_l2_1 = ConvBatchRelu(l2_size, l2_size)

        self.dec_l2_0 = ConvBatchRelu(l2_size, l2_size)
        self.dec_l2_1 = ConvBatchRelu(l2_size, l1_size)
        self.dec_l1_0 = ConvBatchRelu(l1_size + l1_size, l1_size)
        self.dec_l1_1 = ConvBatchRelu(l1_size, l0_size)
        self.dec_l0_0 = ConvBatchRelu(l0_size + l0_size, l0_size)
        self.dec_l0_1 = ConvBatchRelu(l0_size, l0_size)

        self.dec_l0_2 = nn.Sequential(
            Conv(l0_size, oc, 3)
        )
        # self.dec_l0_2 = ConvBatchRelu(l0_size, oc)

    def forward(self, inp):

        # Encoder: Level 0
        enc_conv_l0_0 = self.enc_l0_0(inp)
        enc_conv_l0_1 = self.enc_l0_1(enc_conv_l0_0)
        enc_max_l0 = pool(enc_conv_l0_1, dim=self.dim)

        # Encoder: Level 1
        enc_conv_l1_0 = self.enc_l1_0(enc_max_l0)
        enc_conv_l1_1 = self.enc_l1_1(enc_conv_l1_0)
        enc_max_l1 = pool(enc_conv_l1_1, dim=self.dim)

        # Encoder: Level 2
        enc_conv_l2_0 = self.enc_l2_0(enc_max_l1)
        enc_conv_l2_1 = self.enc_l2_1(enc_conv_l2_0)

        # Decoder
        dec_conv_l2_0 = self.dec_l2_0(enc_conv_l2_1)
        dec_conv_l2_1 = self.dec_l2_1(dec_conv_l2_0)

        dec_cat_l1 = concat(enc_conv_l1_1, upsample(dec_conv_l2_1, dim=self.dim))

        dec_conv_l1_0 = self.dec_l1_0(dec_cat_l1)
        dec_conv_l1_1 = self.dec_l1_1(dec_conv_l1_0)

        dec_cat_l0 = concat(enc_conv_l0_1, upsample(dec_conv_l1_1, dim=self.dim))

        dec_conv_l0_0 = self.dec_l0_0(dec_cat_l0)
        dec_conv_l0_1 = self.dec_l0_1(dec_conv_l0_0)
        output = self.dec_l0_2(dec_conv_l0_1)

        return output

def UNet_prepare(args):

    if args.target_var in ['t2m']:
        return UNet(22, args)
    return UNet(14, args)