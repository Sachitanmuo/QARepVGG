import os
from repvgg import *
import torch.quantization

#first analysis one layer
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = './pattern/'
#layer = QARepVGGBlockV2()


def flatten_and_format(state_dict):
    return '\n'.join(map(str, state_dict.flatten().tolist()))

def tensor_to_binary(tensor):
    array = tensor.numpy()
    binary_array = [format(x, '08b') for x in array.flatten()]
    return binary_array


def gen_layer_pattern(layer, file):
    threebythree_weights = layer.rbr_dense.conv.state_dict()['weight']
    onebyone_weights = layer.rbr_1x1.state_dict()['weight']
    bn3x3_weights = layer.rbr_dense.bn.state_dict()['weight']
    bn3x3_bias = layer.rbr_dense.bn.state_dict()['bias']
    bn3x3_running_mean = layer.rbr_dense.bn.state_dict()['running_mean']
    bn3x3_running_var = layer.rbr_dense.bn.state_dict()['running_var']
    bn3x3_nums_batches_tracked = layer.rbr_dense.bn.state_dict()['num_batches_tracked']

    bn_total_weights = layer.bn.state_dict()['weight']
    bn_total_bias = layer.bn.state_dict()['bias']
    bn_total_running_mean = layer.bn.state_dict()['running_mean']
    bn_total_running_var = layer.bn.state_dict()['running_var']
    bn_total_nums_batches_tracked = layer.bn.state_dict()['num_batches_tracked']
    eps = 1e-5

    std = (bn3x3_running_var + eps).sqrt()
    t = (bn3x3_weights/std).reshape(-1, 1, 1, 1)

    with open("./pattern/kernel_3x3.txt", "w") as f:
        f.write(flatten_and_format(threebythree_weights))
    with open("./pattern/running_mean.txt", "w") as f:
        f.write(flatten_and_format(bn3x3_running_mean))
    with open("./pattern/runnung_var.txt", "w") as f:
        f.write(flatten_and_format(bn3x3_running_var))
    with open("./pattern/gamma.txt", "w") as f:
        f.write(flatten_and_format(bn3x3_weights))
    with open("./pattern/beta.txt", "w") as f:
        f.write(flatten_and_format(bn3x3_bias))
    with open("./pattern/kernel_1x1.txt", "w") as f:
        f.write(flatten_and_format(onebyone_weights))
    with open("./pattern/last_running_mean.txt", "w") as f:
        f.write(flatten_and_format(bn_total_running_mean))
    with open("./pattern/last_running_var.txt", "w") as f:
        f.write(flatten_and_format(bn_total_running_var))
    with open("./pattern/last_gamma.txt", "w") as f:
        f.write(flatten_and_format(bn_total_weights))
    with open("./pattern/last_beta.txt", "w") as f:
        f.write(flatten_and_format(bn_total_bias))
    
    with open("./pattern/fusebn_weights.txt", "w") as f:
        f.write(flatten_and_format(threebythree_weights*t))
    with open("./pattern/fusebn_bias.txt", "w") as f:
        f.write(flatten_and_format(bn3x3_bias - bn3x3_running_mean* bn3x3_weights / std))
    
    

    '''
    file.write(f"3x3 kernel: {flatten_and_format(threebythree_weights)}\n")
    file.write(f"1x1 kernel: {flatten_and_format(onebyone_weights)}\n")
    file.write(f"bn3x3_weights: {flatten_and_format(bn3x3_weights)}\n")
    file.write(f"bn3x3_bias: {flatten_and_format(bn3x3_bias)}\n")
    file.write(f"bn3x3_running_mean: {flatten_and_format(bn3x3_running_mean)}\n")
    file.write(f"bn3x3_running_var: {flatten_and_format(bn3x3_running_var)}\n")
    file.write(f"num_batches_tracked: {flatten_and_format(bn3x3_nums_batches_tracked)}\n")

    file.write(f"bn_weights: {flatten_and_format(bn_total_weights)}\n")
    file.write(f"bn_bias: {flatten_and_format(bn_total_bias)}\n")
    file.write(f"bn_running_mean: {flatten_and_format(bn_total_running_mean)}\n")
    file.write(f"bn_running_var: {flatten_and_format(bn_total_running_var)}\n")
    file.write(f"num_batches_tracked: {flatten_and_format(bn_total_nums_batches_tracked)}\n")

    print("3x3 kernel shape:", threebythree_weights.shape)
    print("1x1 kernel shape:", onebyone_weights.shape)
    print("bn3x3 weights shape:", bn3x3_weights.shape)
    print("bn3x3 bias shape:", bn3x3_bias.shape)
    print("bn3x3 running mean shape:", bn3x3_running_mean.shape)
    print("bn3x3 running var shape:", bn3x3_running_var.shape)
    print("bn total weights shape:", bn_total_weights.shape)
    print("bn total bias shape:", bn_total_bias.shape)
    print("bn total running mean shape:", bn_total_running_mean.shape)
    print("bn total running var shape:", bn_total_running_var.shape)
    '''

def gen_layer_pattern_deploy(layer, file):
    reparam_weights = layer.rbr_reparam.state_dict()['weight']
    reparam_bias = layer.rbr_reparam.state_dict()['bias']
    '''
    file.write(f"3x3 kernel: {flatten_and_format(reparam_weights)}\n")
    file.write(f"1x1 kernel: {flatten_and_format(reparam_bias)}\n")

    print("reparam 3x3 kernel shape:", reparam_weights.shape)
    print("bias shape:", reparam_bias.shape)
    print(layer.rbr_reparam.state_dict())
    '''
    with open("./pattern/reparam_kernel.txt", "w") as f:
        f.write(flatten_and_format(reparam_weights))
    with open("./pattern/reparam_bias.txt", "w") as f:
        f.write(flatten_and_format(reparam_bias))
    

def gen_layer_pattern_deploy_quantized(layer, file):
    print(layer.rbr_reparam.state_dict()['weight'][0].int_repr())
    x = tensor_to_binary(layer.rbr_reparam.state_dict()['weight'][0].int_repr())
    print(x)


def main():
    model = create_QARepVGGBlockV2_A0(deploy=False).to(device)
    checkpoint = torch.load('QARepVGGV2-A0testtest/best_checkpoint.pth')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    stage0_weights = model.stage0.state_dict()
    with open('./pattern/stage_0_weights.txt', 'w') as file:
        gen_layer_pattern(model.stage0, file)
    model.stage0.switch_to_deploy()

    with open('./pattern/stage_0_weights_deploy.txt', 'w') as file:
        gen_layer_pattern_deploy(model.stage0, file)

    torch.backends.quantized.engine = 'fbgemm'
    model.stage0.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model.stage0, inplace=True)
    torch.quantization.convert(model.stage0, inplace=True)
    #print(model.stage0)
    with open('./pattern/stage_0_weights_deploy_quantized.txt', 'w') as file:
        gen_layer_pattern_deploy_quantized(model.stage0, file)

    #torch.save(model.stage0.state_dict(), 'test.pth')

if __name__ == "__main__":
    main()

    
#print(model)