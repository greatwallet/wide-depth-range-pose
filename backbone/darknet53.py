"""
    DarkNet-53 for ImageNet-1K, implemented in PyTorch.
    Original source: 'YOLOv3: An Incremental Improvement,' https://arxiv.org/abs/1804.02767.
"""

__all__ = ['DarkNet53', 'darknet53']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .common import conv1x1_block, conv3x3_block

class DarkUnit(nn.Module):
    """
    DarkNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    alpha : float
        Slope coefficient for Leaky ReLU activation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 alpha):
        super(DarkUnit, self).__init__()
        assert (out_channels % 2 == 0)
        mid_channels = out_channels // 2

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            activation=nn.LeakyReLU(
                negative_slope=alpha,
                inplace=True))
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=nn.LeakyReLU(
                negative_slope=alpha,
                inplace=True))

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + identity


class DarkNet53(nn.Module):
    """
    DarkNet-53 model from 'YOLOv3: An Incremental Improvement,' https://arxiv.org/abs/1804.02767.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    alpha : float, default 0.1
        Slope coefficient for Leaky ReLU activation.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels: int,
                 init_block_channels: int,
                 alpha: float = 0.1,
                 in_channels: int =3,
                 in_size: tuple[int] = (224, 224),
                 num_classes: int = 1000):
        super(DarkNet53, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            activation=nn.LeakyReLU(
                negative_slope=alpha,
                inplace=True))) # res x= 1
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                if j == 0:
                    stage.add_module("unit{}".format(j + 1), conv3x3_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=2,
                        activation=nn.LeakyReLU(
                            negative_slope=alpha,
                            inplace=True)))
                else:
                    stage.add_module("unit{}".format(j + 1), DarkUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        alpha=alpha))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
                    
    def forward(self, x):
        """
        Result: list of 5 stages output
        """
        out1 = self.features.init_block(x)
        out1 = self.features.stage1(out1)
        out2 = self.features.stage2(out1)
        out3 = self.features.stage3(out2)
        out4 = self.features.stage4(out3)
        out5 = self.features.stage5(out4)
        # 
        return [out1, out2, out3, out4, out5]


def get_darknet53(model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".torch", "models"),
                  **kwargs):
    """
    Create DarkNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 32
    layers = [2, 3, 9, 9, 5]
    channels_per_layers = [64, 128, 256, 512, 1024]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = DarkNet53(
        channels=channels,
        init_block_channels=init_block_channels,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def darknet53(**kwargs):
    """
    DarkNet-53 'Reference' model from 'YOLOv3: An Incremental Improvement,' https://arxiv.org/abs/1804.02767.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_darknet53(model_name="darknet53", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count
