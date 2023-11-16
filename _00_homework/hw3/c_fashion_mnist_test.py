import numpy as np
import torch
import os

from matplotlib import pyplot as plt
from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent)  # BASE_PATH: /Users/yhhan/git/link_dl
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")

import sys
sys.path.append(BASE_PATH)

from _00_homework.hw3.b_fashion_mnist_train import get_vgg9_model
from _06_fcn_best_practice.d_tester import ClassificationTester
from _00_homework.hw3.a_fashion_mnist_data import get_fashion_mnist_test_data


def main():
  fashion_mnist_test_images, test_data_loader, fashion_mnist_transforms = get_fashion_mnist_test_data()

  test_model = get_vgg9_model()
  classification_tester = ClassificationTester(
    "cnn_fashion_mnist", test_model, test_data_loader, fashion_mnist_transforms, CHECKPOINT_FILE_PATH
  )
  classification_tester.test()

  img, label = fashion_mnist_test_images[0]
  print("     LABEL:", label)
  plt.imshow(img)
  plt.show()
  print()

  # torch.tensor(np.array(mnist_test_images[0][0])).unsqueeze(dim=0).unsqueeze(dim=0).shape: (1, 1, 28, 28)
  output = classification_tester.test_single(
    torch.tensor(np.array(fashion_mnist_test_images[0][0])).unsqueeze(dim=0).unsqueeze(dim=0)
  )
  print("PREDICTION:", output)
  print()

if __name__ == "__main__":
  main()
