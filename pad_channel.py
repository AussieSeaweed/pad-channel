import torch
import torch.nn as nn
import torchvision


class PadChannelConv2d(nn.Conv2d):
    @classmethod
    def from_conv2d(cls, conv2d):
        if conv2d.padding_mode != 'zeros':
            raise ValueError('unsupported padding mode')

        pad_channel_conv2d = cls(
            conv2d.in_channels,
            conv2d.out_channels,
            conv2d.kernel_size,
            conv2d.stride,
            conv2d.padding,
            conv2d.dilation,
            conv2d.groups,
            conv2d.bias is not None,
            conv2d.padding_mode,
        )

        pad_channel_conv2d.weight.data[:, :-1, :, :] = conv2d.weight.data

        if conv2d.bias is not None:
            pad_channel_conv2d.bias.data = conv2d.bias.data.clone()

        return pad_channel_conv2d

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels + 1, *args, **kwargs)
        
    def forward(self, input_):
        ones = torch.ones_like(input_[:, :1, ...])
        input_ = torch.cat([input_, ones], dim=1)

        return super().forward(input_)


def get_model_with_pad_channel(model_name, *args, **kwargs):
    model = torchvision.models.get_model(model_name, *args, **kwargs)

    match model_name:
        case 'resnet18' | 'resnet34' | 'resnet50' | 'resnet101' | 'resnet152':
            model.conv1 = PadChannelConv2d.from_conv2d(model.conv1)
        case 'vgg11_bn' | 'vgg13_bn' | 'vgg16_bn' | 'vgg19_bn':
            model.features[0] = PadChannelConv2d.from_conv2d(model.features[0])
        case _:
            raise ValueError('model not supported')

    return model
