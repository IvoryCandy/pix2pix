import torch.nn as nn
import numpy as np


def normal_init(m, mean=0.0, std=0.02):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zeor_()


class Generator(nn.Module):
    def __init__(self, in_channel, out_channel, g_conv_dim=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6):
        assert(n_blocks >= 0)
        super(Generator, self).__init__()
        self.input_nc = in_channel
        self.output_nc = out_channel
        self.ngf = g_conv_dim

        model = [nn.Conv2d(in_channel, g_conv_dim, kernel_size=7, padding=3),
                 norm_layer(g_conv_dim, affine=True),
                 nn.ReLU(inplace=True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            current_conv_dim = 2 ** i
            model += [nn.Conv2d(g_conv_dim * current_conv_dim, g_conv_dim * current_conv_dim * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(g_conv_dim * current_conv_dim * 2, affine=True),
                      nn.ReLU(True)]

        current_conv_dim = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(g_conv_dim * current_conv_dim, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            current_conv_dim = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(g_conv_dim * current_conv_dim, int(g_conv_dim * current_conv_dim / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(g_conv_dim * current_conv_dim / 2), affine=True),
                      nn.ReLU(inplace=True)]

        model += [nn.Conv2d(g_conv_dim, out_channel, kernel_size=7, padding=3)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def normal_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        return self.model(x)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    @staticmethod
    def build_conv_block(dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        assert(padding_type == 'zero')
        p = 1

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        x = x + self.conv_block(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channel, d_conv_dim=64, num_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(Discriminator, self).__init__()

        kernel_size = 4
        pad = int(np.ceil((kernel_size - 1) / 2))
        sequence = [nn.Conv2d(in_channel, d_conv_dim, kernel_size=kernel_size, stride=2, padding=pad),
                    nn.LeakyReLU(0.2, True)]

        current_conv_dim = 1
        for n in range(1, num_layers):
            prev_conv_dim = current_conv_dim
            current_conv_dim = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(d_conv_dim * prev_conv_dim, d_conv_dim * current_conv_dim, kernel_size=kernel_size, stride=2,
                          padding=pad), norm_layer(d_conv_dim * current_conv_dim, affine=True), nn.LeakyReLU(0.2, True)]

        prev_conv_dim = current_conv_dim
        current_conv_dim = min(2 ** num_layers, 8)
        sequence += [nn.Conv2d(d_conv_dim * prev_conv_dim, d_conv_dim * current_conv_dim, kernel_size=kernel_size, stride=1,
                               padding=pad), norm_layer(d_conv_dim * current_conv_dim, affine=True), nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(d_conv_dim * current_conv_dim, 1, kernel_size=kernel_size, stride=1, padding=pad)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def normal_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        return self.model(x)
