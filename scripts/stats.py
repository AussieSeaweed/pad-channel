import os
import sys

from ptflops import get_model_complexity_info
from ptflops.pytorch_ops import conv_flops_counter_hook
import torch
import torchvision

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
)

from pad_channel import get_model_with_pad_channel, PadChannelConv2d

MODEL_NAMES = (
    'vgg11_bn',
    'vgg13_bn',
    'vgg16_bn',
    'vgg19_bn',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
)


def main():
    print('Labels\n')

    for model_name in MODEL_NAMES:
        model = torchvision.models.get_model(model_name)
        pc_model = get_model_with_pad_channel(model_name)
        _, parameter_count = get_model_complexity_info(model, (3, 224, 224), print_per_layer_stat=False, as_strings=False, custom_modules_hooks={PadChannelConv2d: conv_flops_counter_hook})
        _, pc_parameter_count = get_model_complexity_info(pc_model, (3, 224, 224), print_per_layer_stat=False, as_strings=False, custom_modules_hooks={PadChannelConv2d: conv_flops_counter_hook})

        print(f'{model_name}: w/o PC: {parameter_count} w/ PC: {pc_parameter_count} Diff: {pc_parameter_count - parameter_count} Diff %: {((pc_parameter_count - parameter_count) / parameter_count) * 100}%')

    print('\nLaTeX Table\n')

    for model_name in MODEL_NAMES:
        model = torchvision.models.get_model(model_name)
        pc_model = get_model_with_pad_channel(model_name)
        _, parameter_count = get_model_complexity_info(model, (3, 224, 224), print_per_layer_stat=False, as_strings=False, custom_modules_hooks={PadChannelConv2d: conv_flops_counter_hook})
        _, pc_parameter_count = get_model_complexity_info(pc_model, (3, 224, 224), print_per_layer_stat=False, as_strings=False, custom_modules_hooks={PadChannelConv2d: conv_flops_counter_hook})

        model_name = model_name.replace('_', '\\_')

        print(f'{model_name} & {parameter_count} & {pc_parameter_count} & {pc_parameter_count - parameter_count} & {((pc_parameter_count - parameter_count) / parameter_count) * 100}\\% \\\\')

    print('Labels\n')

    for model_name in MODEL_NAMES:
        model = torchvision.models.get_model(model_name)
        pc_model = get_model_with_pad_channel(model_name)
        mac, _ = get_model_complexity_info(model, (3, 224, 224), print_per_layer_stat=False, as_strings=False, custom_modules_hooks={PadChannelConv2d: conv_flops_counter_hook})
        pc_mac, _ = get_model_complexity_info(pc_model, (3, 224, 224), print_per_layer_stat=False, as_strings=False, custom_modules_hooks={PadChannelConv2d: conv_flops_counter_hook})

        print(f'{model_name}: w/o PC: {mac / 1e9:.2f} GMACs w/ PC: {pc_mac / 1e9:.2f} GMACs Diff: {(pc_mac - mac) / 1e9:.2f} GMACs Diff %: {((pc_mac - mac) / mac) * 100:.3f}%')

    print('\nLaTeX Table\n')

    for model_name in MODEL_NAMES:
        model = torchvision.models.get_model(model_name)
        pc_model = get_model_with_pad_channel(model_name)
        mac, _ = get_model_complexity_info(model, (3, 224, 224), print_per_layer_stat=False, as_strings=False, custom_modules_hooks={PadChannelConv2d: conv_flops_counter_hook})
        pc_mac, _ = get_model_complexity_info(pc_model, (3, 224, 224), print_per_layer_stat=False, as_strings=False, custom_modules_hooks={PadChannelConv2d: conv_flops_counter_hook})

        model_name = model_name.replace('_', '\\_')

        print(f'{model_name} & {mac / 1e9:.2f} GMACs & {pc_mac / 1e9:.2f} GMACs & {(pc_mac - mac) / 1e9:.2f} GMACs & {((pc_mac - mac) / mac) * 100:.3f}\\% \\\\')


if __name__ == '__main__':
    main()
