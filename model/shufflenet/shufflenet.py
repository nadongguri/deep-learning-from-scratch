# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
sys.path.append('../../')
import numpy as np
from dataset.mnist import load_mnist
from common.layers import Convolution
from common.layers import GroupConvolution
from common.layers import DWConvolution
from common.layers import Pooling
from common.layers import AVGPooling
from common.layers import Affine
from common.layers import Softmax
from common.layers import BatchNormalization
from common.layers import Relu
from common.trainer import Trainer
import argparse
import json

from torchvision import transforms # PyTorch is required for processing input images
from PIL import Image

#method from https://github.com/jaxony/ShuffleNet.git
def get_transformer():
  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])

  transformer = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor(),
      normalize
  ])
  return transformer

def preprocess(image, transformer):
  x = transformer(image).numpy()
  return x.reshape(-1, x.shape[0], x.shape[1], x.shape[2])

def savetxt(name, data, fmt='%4.3f'):
    np.savetxt(name, data.reshape(-1), newline='\n', fmt=fmt)

stage2_str='./data/' + 'ShuffleUnit_Stage2_'
stage3_str='./data/' + 'ShuffleUnit_Stage3_'
stage4_str='./data/' + 'ShuffleUnit_Stage4_'

#by using numpy only not torch
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.shape
    #print('channel_shuffle ', batchsize, num_channels, height, width)

    channels_per_group = num_channels // groups
    #print('channels_per_group ', channels_per_group, num_channels, groups)

    # reshape
    x = x.reshape(batchsize, groups,
        channels_per_group, height, width)
    x = x.transpose(0,2,1,3,4)
    x = x.reshape(batchsize, -1, height, width)
    return x

def stage_shuffle(input_data, stage, repeat_num, groups):
    avgpool_layer = AVGPooling(3,3,2,1)
    residual = avgpool_layer.forward(input_data)
    #savetxt('./dump/' + 'avg_pool.txt', residual)

    w = np.load(stage + '0.g_conv_1x1_compress.conv1x1.weight.npy')
    b = np.load(stage + '0.g_conv_1x1_compress.conv1x1.bias.npy')

    if 'Stage2' in stage:
        conv_layer = Convolution(w, b, stride=1, pad=0)
    else:
        conv_layer = GroupConvolution(w, b, stride=1, pad=0, groups=groups)
    out = conv_layer.forward(input_data)
    out_N, out_C, out_H, out_W = out.shape

    gamma = np.load(stage+'0.g_conv_1x1_compress.batch_norm.weight.npy').reshape((-1,1))
    beta  = np.load(stage+'0.g_conv_1x1_compress.batch_norm.bias.npy').reshape((-1,1))
    mean  = np.load(stage+'0.g_conv_1x1_compress.batch_norm.running_mean.npy').reshape((-1,1))
    var   = np.load(stage+'0.g_conv_1x1_compress.batch_norm.running_var.npy').reshape((-1,1))
    bn_layer = BatchNormalization(gamma, beta, running_mean=mean, running_var=var)
    out = bn_layer.forward(out.reshape(out_C, -1), train_flg=False)
    relu_layer = Relu()
    out = relu_layer.forward(out).reshape(out_N, out_C, out_H, out_W)
    #savetxt('./dump/' + '1x1_comp.txt', out)

    out = channel_shuffle(out, groups)
    #savetxt('./dump/' + 'channel_shuffle.txt', out)

    w = np.load(stage + '0.depthwise_conv3x3.weight.npy').transpose(1,0,2,3)
    b = np.load(stage + '0.depthwise_conv3x3.bias.npy')
    dwconv_layer = DWConvolution(w, b, stride=2, pad=1)
    out = dwconv_layer.forward(out)
    #savetxt('./dump/' + 'dwconv.txt', out)

    gamma = np.load(stage+'0.bn_after_depthwise.weight.npy').reshape((-1,1))
    beta  = np.load(stage+'0.bn_after_depthwise.bias.npy').reshape((-1,1))
    mean  = np.load(stage+'0.bn_after_depthwise.running_mean.npy').reshape((-1,1))
    var   = np.load(stage+'0.bn_after_depthwise.running_var.npy').reshape((-1,1))
    bn_layer = BatchNormalization(gamma, beta, running_mean=mean, running_var=var)
    out_N, out_C, out_H, out_W = out.shape
    out = bn_layer.forward(out.reshape(out_C, -1), train_flg=False).reshape(out_N, out_C, out_H, out_W)
    #savetxt('./dump/' + 'after_bn.txt', out)

    w = np.load(stage + '0.g_conv_1x1_expand.conv1x1.weight.npy')
    b = np.load(stage + '0.g_conv_1x1_expand.conv1x1.bias.npy')
    groupconv_layer = GroupConvolution(w, b, stride=1, pad=0, groups=groups)
    out = groupconv_layer.forward(out)

    gamma = np.load(stage+'0.g_conv_1x1_expand.batch_norm.weight.npy').reshape((-1,1))
    beta  = np.load(stage+'0.g_conv_1x1_expand.batch_norm.bias.npy').reshape((-1,1))
    mean  = np.load(stage+'0.g_conv_1x1_expand.batch_norm.running_mean.npy').reshape((-1,1))
    var   = np.load(stage+'0.g_conv_1x1_expand.batch_norm.running_var.npy').reshape((-1,1))
    bn_layer = BatchNormalization(gamma, beta, running_mean=mean, running_var=var)
    out_N, out_C, out_H, out_W = out.shape
    out = bn_layer.forward(out.reshape(out_C, -1), train_flg=False).reshape(out_N, out_C, out_H, out_W)
    #savetxt('./dump/' + 'gconv.txt', out)

    out = np.concatenate((residual, out), 1)
    #savetxt('./dump/' + 'combine.txt', out)
    relu_layer = Relu()
    out_N, out_C, out_H, out_W = out.shape
    out = relu_layer.forward(out).reshape(out_N, out_C, out_H, out_W)
    #savetxt('./dump/' + 'stage2.txt', out)

    for i in range(1, repeat_num + 1):
        residual = out
        w = np.load(stage + str(i) + '.g_conv_1x1_compress.conv1x1.weight.npy')
        b = np.load(stage + str(i)+'.g_conv_1x1_compress.conv1x1.bias.npy')
        groupconv_layer = GroupConvolution(w, b, stride=1, pad=0, groups=groups)
        out = groupconv_layer.forward(out)
        out_N, out_C, out_H, out_W = out.shape

        gamma = np.load(stage+str(i)+'.g_conv_1x1_compress.batch_norm.weight.npy').reshape((-1,1))
        beta  = np.load(stage+str(i)+'.g_conv_1x1_compress.batch_norm.bias.npy').reshape((-1,1))
        mean  = np.load(stage+str(i)+'.g_conv_1x1_compress.batch_norm.running_mean.npy').reshape((-1,1))
        var   = np.load(stage+str(i)+'.g_conv_1x1_compress.batch_norm.running_var.npy').reshape((-1,1))
        bn_layer = BatchNormalization(gamma, beta, running_mean=mean, running_var=var)
        out = bn_layer.forward(out.reshape(out_C, -1), train_flg=False)
        relu_layer = Relu()
        out = relu_layer.forward(out).reshape(out_N, out_C, out_H, out_W)
        #savetxt('./dump/' + str(i) + '_1x1_comp.txt', out)

        out = channel_shuffle(out, groups)
        #savetxt('./dump/' + 'channel_shuffle.txt', out)

        w = np.load(stage + str(i)+'.depthwise_conv3x3.weight.npy').transpose(1,0,2,3)
        b = np.load(stage + str(i)+'.depthwise_conv3x3.bias.npy')
        dwconv_layer = DWConvolution(w, b, stride=1, pad=1)
        out = dwconv_layer.forward(out)
        #savetxt('./dump/' + 'dwconv.txt', out)

        gamma = np.load(stage+str(i)+'.bn_after_depthwise.weight.npy').reshape((-1,1))
        beta  = np.load(stage+str(i)+'.bn_after_depthwise.bias.npy').reshape((-1,1))
        mean  = np.load(stage+str(i)+'.bn_after_depthwise.running_mean.npy').reshape((-1,1))
        var   = np.load(stage+str(i)+'.bn_after_depthwise.running_var.npy').reshape((-1,1))
        bn_layer = BatchNormalization(gamma, beta, running_mean=mean, running_var=var)
        out_N, out_C, out_H, out_W = out.shape
        out = bn_layer.forward(out.reshape(out_C, -1), train_flg=False).reshape(out_N, out_C, out_H, out_W)
        #savetxt('./dump/' + 'after_bn.txt', out)

        w = np.load(stage + str(i)+'.g_conv_1x1_expand.conv1x1.weight.npy')
        b = np.load(stage + str(i)+'.g_conv_1x1_expand.conv1x1.bias.npy')
        groupconv_layer = GroupConvolution(w, b, stride=1, pad=0, groups=groups)
        out = groupconv_layer.forward(out)

        gamma = np.load(stage+str(i)+'.g_conv_1x1_expand.batch_norm.weight.npy').reshape((-1,1))
        beta  = np.load(stage+str(i)+'.g_conv_1x1_expand.batch_norm.bias.npy').reshape((-1,1))
        mean  = np.load(stage+str(i)+'.g_conv_1x1_expand.batch_norm.running_mean.npy').reshape((-1,1))
        var   = np.load(stage+str(i)+'.g_conv_1x1_expand.batch_norm.running_var.npy').reshape((-1,1))
        bn_layer = BatchNormalization(gamma, beta, running_mean=mean, running_var=var)
        out_N, out_C, out_H, out_W = out.shape
        out = bn_layer.forward(out.reshape(out_C, -1), train_flg=False).reshape(out_N, out_C, out_H, out_W)
        #savetxt('./dump/' + 'gconv.txt', out)

        out = np.add(residual, out)
        #savetxt('./dump/' + str(i) + '_combine.txt', out)
        relu_layer = Relu()
        out_N, out_C, out_H, out_W = out.shape
        out = relu_layer.forward(out).reshape(out_N, out_C, out_H, out_W)
        #savetxt('./dump/' + str(i) + '_stage.txt', out)
    return out

def infererence(args):
    groups = 8

    print('Loading image')
    image = Image.open(args.image)
    print('Preprocessing')
    transformer = get_transformer()
    input_data = preprocess(image, transformer)

    print('input_data  ', input_data.shape)
    #conv layer
    w = np.load('./data/' + 'module.conv1.weight.npy')
    b = np.load('./data/' + 'module.conv1.bias.npy')
    conv_layer = Convolution(w, b, stride=2, pad=1)
    out = conv_layer.forward(input_data)
    #savetxt('./dump/' + 'conv1_out.txt', out)

    #max pooling
    maxpool_layer = Pooling(3,3,2,1)
    out = maxpool_layer.forward(out)
    #savetxt('./dump/' + 'maxpool_out.txt', out)

    out = stage_shuffle(out, stage2_str, 3, groups)
    #savetxt('./dump/' + 'stage2.txt', out)
    out = stage_shuffle(out, stage3_str, 7, groups)
    #savetxt('./dump/' + 'stage3.txt', out)
    out = stage_shuffle(out, stage4_str, 3, groups)
    #savetxt('./dump/' + 'stage4.txt', out)

    h, w= out.shape[-2:]
    avgpool_layer = AVGPooling(h,w,1,0)
    out = avgpool_layer.forward(out).reshape(1,-1)

    w = np.load('./data/' + 'module.fc.weight.npy')
    b = np.load('./data/' + 'module.fc.bias.npy')
    w = w.transpose(1,0)

    fc_layer = Affine(w,b)
    out = fc_layer.forward(out)

    softmax_layer = Softmax()
    out = softmax_layer.forward(out).reshape(-1)

    result = []
    with open(args.idx_to_class) as json_file:
        json_data = json.load(json_file)
        '''
        for key in json_data:
            print(key, json_data[key])
        '''
    for i in range(0, 1000):
        item = (out[i], json_data[str(i)])
        result.append(item)

    result = sorted(result, key=lambda item: item[0], reverse=True)
    for i in range(0, 10):
        print(result[i])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Path to image that we want to classify')
    parser.add_argument('idx_to_class', type=str, help='Path to JSON file mapping indexes to class names')
    args = parser.parse_args()
    infererence(args)
