import os
from pathlib import Path
import torch
import wandb
from torch import nn

from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets
from torchvision.transforms import transforms

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent)
print(BASE_PATH)

import sys
sys.path.append(BASE_PATH)

from _99_common_utils.utils import get_num_cpu_cores, is_linux, is_windows


def get_augmented_fashion_mnist_data():
  data_path = "."

  f_mnist_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transforms.ToTensor())

  f_mnist_train, f_mnist_validation = random_split(f_mnist_train, [55_000, 5_000])

  # Image Augmentation 적용
  # -> RandomCrop : zero-padding을 추가하고 Crop한다.
  # -> RandomHorizontalFlip : 랜덤하게 수평으로 뒤집는다.
  # -> ColorJitter : 밝기, 대비, 채도를 변경한다.
  f_mnist_transforms = nn.Sequential(
      transforms.RandomCrop([28, 28], padding=2),
      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
      transforms.RandomHorizontalFlip()
  )

  # 기존 데이터에 대한 Image Augmentation 결과 생성
  transformed_train_data = []
  for image, label in f_mnist_train:
    transformed_image = f_mnist_transforms(image)
    transformed_train_data.append((transformed_image, label))

  # 원본과 Image Augmentation 결과를 Concat!
  f_mnist_train = ConcatDataset([f_mnist_train, transformed_train_data])

  print("Num Train Samples: ", len(f_mnist_train))
  print("Num Validation Samples: ", len(f_mnist_validation))
  print("Sample Shape: ", f_mnist_train[0][0].shape)  # torch.Size([1, 28, 28])

  num_data_loading_workers = get_num_cpu_cores() if is_linux() or is_windows() else 0
  print("Number of Data Loading Workers:", num_data_loading_workers)

  train_data_loader = DataLoader(
    dataset=f_mnist_train, batch_size=2048, shuffle=True,
    pin_memory=True, num_workers=num_data_loading_workers
  )

  validation_data_loader = DataLoader(
    dataset=f_mnist_validation, batch_size=2048,
    pin_memory=True, num_workers=num_data_loading_workers
  )

  # mean과 std. 다시 구하기!
  imgs = torch.stack([img_t for img_t, _ in f_mnist_train], dim=3)
  print("Image Shape: ", imgs.shape)

  mean = imgs.view(1, -1).mean(dim=-1)
  std = imgs.view(1, -1).std(dim=-1)
  print("Mean: ", mean)
  print("Std.: ", std)

  f_mnist_transforms = nn.Sequential(
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=mean, std=std),
  )

  return train_data_loader, validation_data_loader, f_mnist_transforms


def get_augmented_fashion_mnist_test_data():
  data_path = "."

  f_mnist_test_images = datasets.FashionMNIST(data_path, train=False, download=True)
  f_mnist_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transforms.ToTensor())

  print("Num Test Samples: ", len(f_mnist_test))
  print("Sample Shape: ", f_mnist_test[0][0].shape)  # torch.Size([1, 28, 28])

  test_data_loader = DataLoader(dataset=f_mnist_test, batch_size=len(f_mnist_test))

  # get_fashion_mnist_data에서 구한 mean, std. 값 사용
  f_mnist_transforms = nn.Sequential(
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=0.2852, std=0.3501),
  )

  return f_mnist_test_images, test_data_loader, f_mnist_transforms


if __name__ == "__main__":
  config = {'batch_size': 2048,}
  wandb.init(mode="disabled", config=config)

  train_data_loader, validation_data_loader, f_mnist_transforms = get_augmented_fashion_mnist_data()
  print()
  f_mnist_test_images, test_data_loader, f_mnist_transforms = get_augmented_fashion_mnist_test_data()
