import torch
from torch import nn


class DoubleConv(nn.Module):
    """
    Consists of two times a convolution, followed by batch normalization and ReLU activation
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        kernel_height, kernel_width = kernel_size

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=(kernel_height, kernel_width),
                      padding=(kernel_height // 2, kernel_width // 2)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(kernel_height, kernel_width),
                      padding=(kernel_height // 2, kernel_width // 2)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    """
    Encoder block:
    Consists of a double convolution followed by max pooling.
    Here the number of filters increases and the width decreases by half the size of the input.
    """
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.conv = DoubleConv(in_c, out_c, kernel_size)
        self.pool = nn.MaxPool2d((1, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    """
    Decoder block:
    Begins with a transpose convolution, followed by a concatenations with the skip
    connection from the encoder block, ending with a double convolution. Here the
    number of filters decreases and the width doubles.
    """
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c,
                                     kernel_size=(1, 2),
                                     stride=(1, 2),
                                     padding=0)
        self.conv = DoubleConv(out_c + out_c, out_c, kernel_size)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    Implementation of a U-net with a specified depth using PyTorch.
    """
    def __init__(self, depth, kernel_size, num_filters, device):
        super().__init__()
        self.encoders = nn.ModuleList()

        input_size = 1
        for i in range(depth):
            output_size = num_filters * 2 ** i
            self.encoders.append(
                EncoderBlock(input_size, output_size, kernel_size).to(device))
            input_size = output_size

        output_size = num_filters * 2 ** depth
        self.bottleneck = DoubleConv(input_size,
                                     output_size,
                                     kernel_size).to(device)
        input_size = output_size
        self.decoders = nn.ModuleList()
        for i in range(depth):
            output_size = num_filters * 2 ** (depth - 1 - i)
            self.decoders.append(
                DecoderBlock(input_size, output_size, kernel_size).to(device))
            input_size = output_size

        self.outputs = nn.Conv2d(in_channels=input_size,
                                 out_channels=1,
                                 kernel_size=1,
                                 padding=0).to(device)

    def forward(self, inputs):
        """
        Encoder:
        Where s is the skip connect layer and p the output of the max pooling layer
        """
        ss = []
        for encoder in self.encoders:
            s, p = encoder(inputs)
            ss.append(s)
            inputs = p

        """ 
        Bottleneck:
        Where b is the output of the bottleneck
        """
        b = self.bottleneck(inputs)

        """ 
        Decoder:
        Where s is the corresponding skip connect layer from the encoder block and 
        d the output of the decoder block
        """
        for decoder in self.decoders:
            s = ss.pop()
            d = decoder(b, s)
            b = d

        """ 
        Classifier:
        Final convolution resulting in logits
        """
        outputs = self.outputs(b)
        return outputs
