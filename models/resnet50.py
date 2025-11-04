import torch 
import torch.nn as nn

class BottleneckGenerator(nn.Module):
    """
    Blocks generation for each conv layer in ResNet(50)

    Args:
        blocks_amount: amount of blocks to generate
        input_shape: input dimension for first block 
        shape1-3: shapes to represent in blocks
        stride: stride for each conv layer (1 as default)
    """
    def __init__(self, blocks_amount, input_shape, shape1, shape2, shape3, stride=1):
        super().__init__()
        self.blocks_amount = blocks_amount
        self.input_shape = input_shape
        self.shape1, self.shape2, self.shape3 = shape1, shape2, shape3
        self.stride = stride
        self.blocks = self._generate_blocks()

    def _generate_blocks(self):
        """
        Generates bottleneck blocks (1x1, 3x3, 1x1)
        """
        blocks = nn.ModuleList()
        for i in range(self.blocks_amount):
            in_channels = self.input_shape if i == 0 else self.shape3 
            curr_stride = self.stride if i == 0 else 1

            block = nn.Sequential(
            nn.Conv2d(in_channels, self.shape1, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.shape1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.shape1, self.shape2, kernel_size=3, padding=1, stride=curr_stride, bias=False),
            nn.BatchNorm2d(self.shape2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.shape2, self.shape3, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.shape3),
            )
            
            self._init_weights(block)
            blocks.append(block)
        return blocks

    def _init_weights(self, block):
        """
        He (Kaiming) weight initialization with normal distribution.
        """
        for m in block.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class ResNet(nn.Module):
    """
    Actual ResNet50.

    Args:
        in_channels - shape of input for this model
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv0 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv0.apply(self.init_conv0_weights)
        
        self.conv1 = BottleneckGenerator(3, 64, 64, 64, 256, 1)
        self.conv2 = BottleneckGenerator(4, 256, 128, 128, 512, 2)
        self.conv3 = BottleneckGenerator(6, 512, 256, 256, 1024, 2)
        self.conv4 = BottleneckGenerator(3, 1024, 512, 512, 2048, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000)

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
