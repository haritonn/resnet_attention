import torch
import torch.nn as nn
import config.config as config


class ChannelAttention(nn.Module):
    """
    Channel attention mechanism to implement in ResNet.
    Key idea: output from sigmoid will be coefficient to multiply on current feature maps.
    So this block will learn which features are more useful, and which kind of useless

    Args:
        channels_in - channels in from previous convolutional layer
        reduction - coefficient to reduce (some autoencoder action)
    """

    def __init__(self, channels_in, reduction=config.ATTENTION_REDUCTION):
        super().__init__()
        self.channels_in = channels_in
        self.reduction = reduction

        self.pool = nn.AdaptiveAvgPool2d(1)

        self._make_layers()
        self.layers.apply(self._init_weights)

    def _make_layers(self):
        reduced_channels = self.channels_in // self.reduction
        layers = nn.Sequential(
            nn.Linear(self.channels_in, reduced_channels, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_channels, self.channels_in, bias=False),
            nn.Sigmoid(),
        )

        self.layers = layers

    # Keep using Kaiming init because we're focusing on internal relu function.
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        pool = self.pool(x)
        pool = pool.view(pool.size(0), -1)
        att_coeffs = self.layers(pool)

        att_coeffs = att_coeffs.view(att_coeffs.size(0), att_coeffs.size(1), 1, 1)

        return att_coeffs * x
