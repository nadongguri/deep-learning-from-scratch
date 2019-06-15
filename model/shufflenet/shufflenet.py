# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
sys.path.append('../../')
import numpy as np
from dataset.mnist import load_mnist
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


def infererence(args):
    print('Loading image')
    image = Image.open(args.image)
    print('Preprocessing')
    transformer = get_transformer()
    x = preprocess(image, transformer)
    print(x.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Path to image that we want to classify')
    parser.add_argument('idx_to_class', type=str, help='Path to JSON file mapping indexes to class names')
    args = parser.parse_args()
    print('+_+ ',args)
    infererence(args)
