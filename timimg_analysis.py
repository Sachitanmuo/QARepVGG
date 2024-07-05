import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from simplified_repvgg import *

import torch.nn as nn
import numpy as np
import torch
from functools import partial
from se_block import SEBlock
import time
model = create_QARepVGGBlockV2_A0()
inputs = torch.randn(1, 3, 224, 224)
layer_times = []
iterations = 1
def profile_layer(layer, inputs):
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        layer(inputs)
    layer_time = prof.self_cpu_time_total
    return layer_time

def profile_block(stage, inputs):
    layer_times = {}
    
    layer_times["stage total"] = profile_layer(stage, inputs)
    layer_times["rbr_dense"] = profile_layer(stage.rbr_dense, inputs)
    layer_times["rbr_dense_conv"] = profile_layer(stage.rbr_dense.conv, inputs)
    temp = stage.rbr_dense.conv(inputs)
    layer_times["rbr_dense_bn"] = profile_layer(stage.rbr_dense.bn, temp)
    layer_times["rbr_1x1"] = profile_layer(stage.rbr_1x1, inputs)
    id_out = 0 if stage.rbr_identity is None else stage.rbr_identity(inputs)
    
    output = stage.rbr_dense(inputs) + stage.rbr_1x1(inputs) + id_out 
    layer_times["bn"] = profile_layer(stage.bn, output)
    layer_times["nonlinearity"] = profile_layer(stage.nonlinearity, output)
    #layer_times["add"] = layer_times["stage total"] - layer_times["rbr_dense"] - layer_times["rbr_1x1"] - layer_times["bn"] - layer_times["nonlinearity"]
    output = stage(inputs)
    
    return layer_times, output

def profile_block_deploy(stage, inputs):
    layer_times = {}
    
    layer_times["stage total"] = profile_layer(stage, inputs)
    layer_times["rbr_reparam"] = profile_layer(stage.rbr_reparam, inputs)
    output = stage.rbr_reparam(inputs)
    layer_times["nonlinearity"] = profile_layer(stage.nonlinearity, output)
    #layer_times["add"] = layer_times["stage total"] - layer_times["rbr_dense"] - layer_times["rbr_1x1"] - layer_times["bn"] - layer_times["nonlinearity"]
    output = stage(inputs)
    
    return layer_times, output

items = 7
stages = [model.stage0, model.stage1, model.stage2, model.stage3]
stage_layers = [2, 4, 14, 1]
time_infos = np.zeros((len(stages), max(stage_layers), items))

for x in range (0, iterations):
    inputs = torch.randn(1, 3, 224, 224)
    for i, stage in enumerate(stages):
        for j in range(0, stage_layers[i]):
            layer_times, _ = profile_block(stage[j], inputs)
            inputs = stage[j](inputs)
            for k, (layer_name, time) in enumerate(layer_times.items()):
                time_infos[i, j, k] += time

time_infos /= iterations



#Inference
inputs = torch.randn(1, 3, 224, 224)
model = create_QARepVGGBlockV2_A0(deploy= True)
layer_times_ = []
items_ = 3
stages_ = [model.stage0, model.stage1, model.stage2, model.stage3]
stage_layers = [2, 4, 14, 1]
time_infos_deploy = np.zeros((len(stages_), max(stage_layers), items_))

for x in range (0, iterations):
    inputs = torch.randn(1, 3, 224, 224)
    for i, stage in enumerate(stages_):
        for j in range(0, stage_layers[i]):
            layer_times_ , _ = profile_block_deploy(stage[j], inputs)
            inputs = stage[j](inputs)
            for k, (layer_name, time) in enumerate(layer_times_.items()):
                time_infos_deploy[i, j, k] += time

time_infos_deploy /= iterations

print("=========Train  Mode============")
for i in range(len(stages)):
    print(f"\nStage {i:2d}")
    for j in range(stage_layers[i]):
        print('|')
        print(f"-----Layer{j:2d}")
        
        for k, layer_name in enumerate(layer_times.keys()):
            print("     |")
            print(f"     -----{layer_name.ljust(15)}: {int(time_infos[i, j, k]):4d} us")
print("================================\n\n")

print("=========Deploy Mode============")
for i in range(len(stages_)):
    print(f"\nStage {i:2d}")
    for j in range(stage_layers[i]):
        print('|')
        print(f"-----Layer{j:2d}")
        
        for k, layer_name in enumerate(layer_times_.keys()):
            print("     |")
            print(f"     -----{layer_name.ljust(15)}: {int(time_infos_deploy[i, j, k]):4d} us")
print("================================\n\n")

time_infos = time_infos.reshape((len(stages)*max(stage_layers), items))
time_infos_deploy = time_infos_deploy.reshape((len(stages_)*max(stage_layers), items_))
np.savetxt('time_infos.csv', time_infos.reshape(-1, items), delimiter=',', fmt='%d')
np.savetxt('time_infos_deploy.csv', time_infos_deploy.reshape(-1, items_), delimiter=',', fmt='%d')
#print(time_infos.shape)
#print(time_infos_deploy.shape)


def count_parameters(layer):
    return sum(p.numel() for p in layer.parameters())

for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        num_params = count_parameters(layer)
        print(f"Layer: {name}, Parameters: {num_params}")


#with torch.autograd.profiler.profile(use_cuda=False) as prof:
#    model(inputs)
#print(prof)








