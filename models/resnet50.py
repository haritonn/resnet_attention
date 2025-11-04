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
            nn.ReLU(),
            nn.Conv2d(self.shape1, self.shape2, kernel_size=3, padding=1, stride=curr_stride, bias=False),
            nn.BatchNorm2d(self.shape2),
            nn.ReLU(),
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
    pass
