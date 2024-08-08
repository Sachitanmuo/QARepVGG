import torch
import torch.nn as nn
import numpy as np
from repvgg import create_RepVGG_A0

def quantize_tensor_signed(tensor, q_range):
    max_ = torch.max(abs(tensor))
    scaling_factor = 128. / max_
    quantized_tensor = tensor * scaling_factor
    quantized_tensor = torch.clamp(quantized_tensor, -128, 127)
    quantized_tensor = torch.round(quantized_tensor)
    return quantized_tensor, 1. / scaling_factor

def quantize_tensor_unsigned(tensor, q_range):
    max_ = torch.max(tensor)
    scaling_factor = max_ / 255.
    quantized_tensor = tensor / scaling_factor
    quantized_tensor = torch.clamp(quantized_tensor, 0, 255)
    quantized_tensor = torch.round(quantized_tensor)
    return quantized_tensor, scaling_factor

class QuantizedRepVGG(nn.Module):
    def __init__(self, repvgg_model):
        super(QuantizedRepVGG, self).__init__()
        self.stage0 = repvgg_model.stage0
        self.stage1 = repvgg_model.stage1
        self.stage2 = repvgg_model.stage2
        self.stage3 = repvgg_model.stage3
        self.stage4 = repvgg_model.stage4
        self.gap = repvgg_model.gap
        self.linear = repvgg_model.linear


        self.quantized_weights = {}
        self.scaling_factors = {}


        self.quantize_layer(self.stage0, 'stage0', q_range_unsigned=16, q_range_signed=2, first_layer=True)


        self.quantize_stage(self.stage1, 'stage1', q_ranges=[(16, 1), (8, 4)])
        self.quantize_stage(self.stage2, 'stage2', q_ranges=[(8, 1)] * 4)
        self.quantize_stage(self.stage3, 'stage3', q_ranges=[(8, 2)] + [(8, 1)] * 13)
        self.quantize_stage(self.stage4, 'stage4', q_ranges=[(16, 2)])

        self.quantize_linear(self.linear, 'linear', q_range_unsigned=4, q_range_signed=1)

    def quantize_stage(self, stage, stage_name, q_ranges):
        for idx, layer in enumerate(stage):
            q_range_unsigned, q_range_signed = q_ranges[idx]
            self.quantize_layer(layer, stage_name, q_range_unsigned, q_range_signed, idx)

    def quantize_layer(self, layer, stage_name, q_range_unsigned, q_range_signed, idx=None, first_layer=False):
        if first_layer:
            quantized_weights, s_weight = quantize_tensor_signed(layer.rbr_reparam.weight, q_range_signed)
            layer_name = stage_name
        else:
            quantized_weights, s_weight = quantize_tensor_signed(layer.rbr_reparam.weight, q_range_signed)
            layer_name = f'{stage_name}_{idx}'

        self.quantized_weights[f'{layer_name}_weight'] = quantized_weights
        self.scaling_factors[f'{layer_name}_weight'] = s_weight

    def quantize_linear(self, layer, layer_name, q_range_unsigned, q_range_signed):
        quantized_weights, s_weight = quantize_tensor_signed(layer.weight, q_range_signed)
        self.quantized_weights[f'{layer_name}_weight'] = quantized_weights
        self.scaling_factors[f'{layer_name}_weight'] = s_weight

    def forward(self, x):
        x = self.apply_quantized_layer(self.stage0, x, 'stage0', first_layer=True)
        x = self.apply_quantized_stage(self.stage1, x, 'stage1')
        x = self.apply_quantized_stage(self.stage2, x, 'stage2')
        x = self.apply_quantized_stage(self.stage3, x, 'stage3')
        x = self.apply_quantized_stage(self.stage4, x, 'stage4')

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.apply_quantized_linear(self.linear, x, 'linear')
        return x

    def apply_quantized_stage(self, stage, x, stage_name):
        for idx, layer in enumerate(stage):
            x = self.apply_quantized_layer(layer, x, f'{stage_name}_{idx}')
        return x

    def apply_quantized_layer(self, layer, x, layer_name, first_layer=False):
        if first_layer:
            weight = self.quantized_weights[f'{layer_name}_weight']
            s_weight = self.scaling_factors[f'{layer_name}_weight']
            x = nn.functional.conv2d(x, weight * s_weight, bias=layer.rbr_reparam.bias, stride=layer.rbr_reparam.stride, padding=layer.rbr_reparam.padding)
        else:
            weight = self.quantized_weights[f'{layer_name}_weight']
            s_weight = self.scaling_factors[f'{layer_name}_weight']
            x = nn.functional.conv2d(x, weight * s_weight, bias=layer.rbr_reparam.bias, stride=layer.rbr_reparam.stride, padding=layer.rbr_reparam.padding)
        x = nn.functional.relu(x)
        return x

    def apply_quantized_linear(self, layer, x, layer_name):
        weight = self.quantized_weights[f'{layer_name}_weight']
        s_weight = self.scaling_factors[f'{layer_name}_weight']
        x = nn.functional.linear(x, weight * s_weight, bias=layer.bias)
        return x



