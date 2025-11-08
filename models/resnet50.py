import torch
import torch.nn as nn

from models.attention import ChannelAttention


class BottleneckBlock(nn.Module):
    """
    Single Bottleneck Block with skip connection

    Structure: 1x1 (reduce) -> 3x3 (main conv) -> 1x1 (expand) + skip connection

    Args:
        in_channels: input channel dimension
        mid_channels: intermediate channel dimension (bottleneck)
        out_channels: output channel dimension (usually mid_channels * 4)
        stride: stride for the 3x3 conv layer (controls spatial downsampling)
    """

    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.attention = ChannelAttention(out_channels, reduction=16)

        self.relu = nn.ReLU(inplace=True)

        self.skip_connection = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.attention(out)

        out = out + identity
        out = self.relu(out)

        return out


class BottleneckGenerator(nn.Module):
    """
    Blocks generation for each conv layer in ResNet(50)

    Args:
        blocks_amount: amount of blocks to generate
        input_shape: input dimension for first block
        shape1-3: shapes to represent in blocks
        stride: stride for each conv layer (1 as default)
    """

    def __init__(self, blocks_count, in_channels, mid_channels, out_channels, stride=1):
        super().__init__()
        blocks = []
        blocks.append(
            BottleneckBlock(in_channels, mid_channels, out_channels, stride=stride)
        )
        for _ in range(1, blocks_count):
            blocks.append(
                BottleneckBlock(out_channels, mid_channels, out_channels, stride=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ResNet(nn.Module):
    """
    Actual ResNet50.

    Args:
        in_channels - shape of input for this model
    """

    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        self.in_channels = in_channels
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv0.apply(self._init_conv0_weights)

        self.conv1 = BottleneckGenerator(3, 64, 64, 256, 1)
        self.conv2 = BottleneckGenerator(4, 256, 128, 512, 2)
        self.conv3 = BottleneckGenerator(6, 512, 256, 1024, 2)
        self.conv4 = BottleneckGenerator(3, 1024, 512, 2048, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _init_conv0_weights(self, m):
        """
        He (Kaiming) weight initialization for stem layer of ResNet50
        """
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward-propagation of resnet
        """
        x = self.conv0(x)
        for block in self.conv1.blocks:
            x = block(x)

        for block in self.conv2.blocks:
            x = block(x)

        for block in self.conv3.blocks:
            x = block(x)

        for block in self.conv4.blocks:
            x = block(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# test
if __name__ == "__main__":
    model = ResNet(in_channels=3)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(output)
