from typing import List, Tuple
import torch.nn as nn
import torch
import utils


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
#        Encoder-Decoder
###################################


class EncoderResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(EncoderResNet, self).__init__()
        assert len(input_shape) == 3

        channels = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features
        height += 6
        width += 6
        height, width = utils.output_shape_of_conv2d(model[1], (height, width))

        # Downsampling
        for _ in range(2):
            out_features *= 2
            padding = 1
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=padding),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            height, width = utils.output_shape_of_conv2d(model[-3], (height, width))

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        self.model = nn.Sequential(*model)
        self.output_shape = (out_features, height, width)

    def forward(self, x):
        return self.model(x)


class DecoderResNet(nn.Module):
    def __init__(self, input_shape, output_shape, num_residual_blocks):
        super(DecoderResNet, self).__init__()

        assert len(output_shape) == 3
        out_channels = output_shape[0]
        out_height = output_shape[1]
        out_width = output_shape[2]

        out_features = 64
        model: List[nn.Module] = []
        model = [nn.ReflectionPad2d(3), nn.Conv2d(out_features, out_channels, 7), nn.Tanh()]

        # Upsampling
        for _ in range(2):
            in_features = out_features * 2
            conv_layer = nn.Conv2d(in_features, out_features, 3, stride=1, padding=1)
            model = [
                nn.Upsample(scale_factor=2),
                conv_layer,
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ] + model
            out_features = in_features
            out_height, out_width = utils.lb_input_shape_of_conv2d(conv_layer, (out_height, out_width))
            assert out_height % 2 == 0 and out_width % 2 == 0
            out_height //= 2
            out_width //= 2

        # Residual blocks
        for _ in range(num_residual_blocks):
            resblock: nn.Module = ResidualBlock(out_features)
            model.insert(0, resblock)

        assert len(input_shape) == 3
        assert input_shape[1] - out_height == input_shape[2] - out_width
        kernel_size = input_shape[1] - out_height + 1
        padding = 0
        if (kernel_size < 3):
            padding = 3 - kernel_size
            kernel_size = 3

        assert kernel_size <= 7
        model = [
            nn.Conv2d(input_shape[0], out_features, kernel_size, stride = 1, padding=padding),
            nn.ReLU(),
        ] + model

        # Output layer
        self.model = nn.Sequential(*model)


    def forward(self, content: torch.Tensor, style: torch.Tensor):
        assert len(style.shape) == len(content.shape) == 4
        ints = torch.cat((style, content), dim = 1)
        return self.model(ints)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers: List[ nn.Module ] = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
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


class ContentCodeDiscriminator(nn.Module):
    def __init__(self, input_shape: Tuple[int,int,int]) -> None:
        super(ContentCodeDiscriminator, self).__init__()

        channels, height, width = input_shape
        self.output_shape = (1, height // 2 ** 2, width // 2 ** 2)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers: List[ nn.Module ] = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels * 1, channels * 2, normalize=False),
            *discriminator_block(channels * 2, channels * 4),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(channels * 4, 1, 4, padding=1)
        )

    def forward(self, input):
        return self.model(input)
