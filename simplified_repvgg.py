import torch.nn as nn
import numpy as np
import torch
from functools import partial
from se_block import SEBlock
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU(), drop_path_ratio=0.0):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.use_scale = use_scale

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2
        if isinstance(act, (type, )):
            if act == nn.PReLU:
                self.nonlinearity = nn.PReLU(num_parameters=out_channels)
            else:
                self.nonlinearity = act()
        else:
            self.nonlinearity = act

        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        if self.rbr_identity is None:
            return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
        else:
            return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel = self.rbr_dense.conv.weight + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.conv.weight)
        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(self.rbr_dense.conv.weight.device)
            kernel = kernel + id_tensor
        return kernel, 0

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

class QARepVGGBlockV2(RepVGGBlock):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU(), drop_path_ratio=0.0):
        super(QARepVGGBlockV2, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se, use_scale, act, drop_path_ratio)
        if not deploy:
            self.bn = nn.BatchNorm2d(out_channels)
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
        self._id_tensor = None

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.bn(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)))

class RepVGG(nn.Module):
    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False, block_cls=RepVGGBlock):
        super(RepVGG, self).__init__()
        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map
        self.use_se = use_se

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = self._make_stage(block_cls, 3, self.in_planes, num_blocks[0], stride=2, deploy=deploy)
        self.stage1 = self._make_stage(block_cls, self.in_planes, int(64 * width_multiplier[0]), num_blocks[1], stride=2, deploy=deploy)
        self.stage2 = self._make_stage(block_cls, int(64 * width_multiplier[0]), int(128 * width_multiplier[1]), num_blocks[2], stride=2, deploy=deploy)
        self.stage3 = self._make_stage(block_cls, int(128 * width_multiplier[1]), int(256 * width_multiplier[2]), num_blocks[3], stride=2, deploy=deploy)
        self.stage4 = self._make_stage(block_cls, int(256 * width_multiplier[2]), int(512 * width_multiplier[3]), 1, stride=2, deploy=deploy)

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, block, in_planes, planes, num_blocks, stride, deploy):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(str(len(blocks)), 1) if self.override_groups_map is not None else 1
            blocks.append(block(in_planes, planes, kernel_size=3, stride=stride, padding=1, groups=cur_groups, deploy=deploy, use_se=self.use_se))
            in_planes = planes

        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def create_QARepVGGBlockV2_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV2)