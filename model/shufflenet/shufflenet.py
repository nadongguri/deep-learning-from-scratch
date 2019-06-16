# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
sys.path.append('../../')
import numpy as np
from dataset.mnist import load_mnist
from common.layers import Convolution
from common.layers import Pooling
from common.trainer import Trainer
import argparse

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

def savetxt(name, data):
    np.savetxt(name, data.reshape(-1), newline='\n', fmt='%4.3f')


def infererence(args):
    print('Loading image')
    image = Image.open(args.image)
    print('Preprocessing')
    transformer = get_transformer()
    input_data = preprocess(image, transformer)

    #conv layer
    conv1_w = np.load('./data/' + 'module.conv1.weight.npy')
    conv1_b = np.load('./data/' + 'module.conv1.bias.npy')
    conv1_layer = Convolution(conv1_w, conv1_b, stride=2, pad=1)
    conv1_out = conv1_layer.forward(input_data)
    #savetxt('./dump/' + 'conv1_out.txt', conv1_out)

    #max pooling
    maxpooling = Pooling(3,3,2,1)
    maxpool_out = maxpooling.forward(conv1_out)
    #savetxt('./dump/' + 'maxpool_out.txt', maxpool_out)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Path to image that we want to classify')
    parser.add_argument('idx_to_class', type=str, help='Path to JSON file mapping indexes to class names')
    args = parser.parse_args()
    infererence(args)
