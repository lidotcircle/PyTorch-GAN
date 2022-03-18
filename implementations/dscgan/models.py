from typing import List, Tuple
import torch.nn as nn
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            # Padding for keeping the shape of output tensor same as input tensor under convolution
            # nn.ReflectionPad2d(paddingsize),
            nn.ReflectionPad2d(1),
            # nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)



###################################
#        Domain Encoder
###################################
class DomainEncoder(nn.Module):
    def __init__(self, input_channels: int, n_residual_blocks: int):
        super(DomainEncoder, self).__init__()

        out_features = 64
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, out_features, kernel_size=7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        for _ in range(4):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        self.out_features = 64
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features, self.out_features, kernel_size=7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


###################################
#        Domain Decoder
###################################
class DomainDecoder(nn.Module):
    def __init__(self, in_features: int, n_residual_blocks: int):
        super(DomainDecoder, self).__init__()

        out_features = in_features * 8
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features, out_features, kernel_size=7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]

        for _ in range(4):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(out_features)]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features, 3, kernel_size=7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)


##############################
#        Domain Discriminator
##############################
class DomainDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(DomainDiscriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers: List[nn.Module] = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
