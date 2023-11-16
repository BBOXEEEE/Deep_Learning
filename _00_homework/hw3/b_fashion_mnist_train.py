import torch
from torch import nn, optim
from datetime import datetime
import os
import wandb
from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")
if not os.path.isdir(CHECKPOINT_FILE_PATH):
  os.makedirs(os.path.join(CURRENT_FILE_PATH, "checkpoints"))

import sys
sys.path.append(BASE_PATH)

from _06_fcn_best_practice.c_trainer import ClassificationTrainer
from _00_homework.hw3.a_fashion_mnist_data import get_fashion_mnist_data
from _06_fcn_best_practice.e_arg_parser import get_parser


def get_sehyeon_cnn_model():
  class MyModel(nn.Module):
    def __init__(self, in_channels, n_output):
      super().__init__()

      self.model = nn.Sequential(
        # B x 1 x 28 x 28 --> B x 32 x (28 - 5 + 4 + 1) x (28 - 5 + 4 + 1) = B x 32 x 28 x 28
        nn.LazyConv2d(out_channels=32, kernel_size=5, stride=1, padding=2),
        #nn.LazyConv2d(out_channels=10, kernel_size=1),
        # B x 32 x 28 x 28 --> B x 32 x 14 x 14
        nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        nn.LazyBatchNorm2d(),
        nn.ReLU(),
        # B x 32 x 14 x 14 --> B x 64 x (14 - 3 + 2 + 1) x (14 - 3 + 2 + 1) = B x 64 x 14 x 14
        nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1, padding=1),
        #nn.LazyConv2d(out_channels=24, kernel_size=1),
        # B x 64 x 14 x 14 --> B x 64 x 7 x 7
        nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        nn.LazyBatchNorm2d(),
        nn.ReLU(),
        # B x 64 x 6 x 6 -> B x 128 x (6 - 3 + 1) x (14 - 3 + 2 + 1) = B x 128 x 6 x 6
        nn.LazyConv2d(out_channels=128, kernel_size=3, stride=1, padding=1),
        #nn.LazyConv2d(out_channels=32, kernel_size=1),
        # B x 128 x 6 x 6 --> B x 128 x 3 x 3
        nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        nn.LazyBatchNorm2d(),
        nn.ReLU(),
        nn.Flatten(),
        nn.Dropout(p=0.25),      # p: dropout probability
        nn.LazyLinear(256),
        nn.ReLU(),
        nn.Dropout(p=0.25),      # p: dropout probability
        nn.LazyLinear(128),
        nn.LazyBatchNorm1d(),
        nn.ReLU(),
        nn.LazyLinear(n_output),
      )

    def forward(self, x):
      x = self.model(x)
      return x

  # 1 * 28 * 28
  my_model = MyModel(in_channels=1, n_output=10)

  return my_model


def get_vgg9_model():
  class MyModel(nn.Module):
    def __init__(self, in_channels, n_output):
      super().__init__()

      self.model = nn.Sequential(
        # Block.1
        # B x 64 x 28 x 28
        nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.LazyBatchNorm2d(),
        nn.ReLU(),
        nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.LazyBatchNorm2d(),
        nn.ReLU(),
        # B x 64 x 14 x 14
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Block.2
        # B x 128 x 14 x 14
        nn.LazyConv2d(out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.LazyBatchNorm2d(),
        nn.ReLU(),
        nn.LazyConv2d(out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.LazyBatchNorm2d(),
        nn.ReLU(),
        # B x 128 x 7 x 7
        nn.MaxPool2d(kernel_size=2, stride=2),

        # FCN
        nn.Flatten(),
        # Input Features = 6272
        nn.LazyLinear(out_features=512),
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.LazyLinear(out_features=256),
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.LazyLinear(n_output)
      )

    def forward(self, x):
      x = self.model(x)
      return x

  # 1 * 28 * 28
  my_model = MyModel(in_channels=1, n_output=10)

  return my_model


def main(args):
  run_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

  config = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'validation_intervals': args.validation_intervals,
    'learning_rate': args.learning_rate,
    'early_stop_patience': args.early_stop_patience,
    'early_stop_delta': args.early_stop_delta,
    'weight_decay': args.weight_decay,
    'dropout_rate': args.dropout_rate,
  }

  # 프로젝트 정보 설정 및 wandb.init()
  project_name = "cnn_fashion_mnist"
  wandb.init(
    mode="online" if args.wandb else "disabled",
    project=project_name,
    notes="fashion_mnist experiment with cnn",
    tags=["cnn", "fashion_mnist"],
    name=run_time_str+"_with_vgg9",
    config=config
  )
  print(args)
  print(wandb.config)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Training on device {device}.")

  train_data_loader, validation_data_loader, mnist_transforms = get_fashion_mnist_data()
  model = get_vgg9_model()
  model.to(device)
  #wandb.watch(model)

  from torchinfo import summary
  summary(
      model=model, input_size=(1, 1, 28, 28),
      col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"]
  )

  # optimizer
  optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)

  classification_trainer = ClassificationTrainer(
    project_name, model, optimizer, train_data_loader, validation_data_loader, mnist_transforms,
    run_time_str, wandb, device, CHECKPOINT_FILE_PATH
  )
  classification_trainer.train_loop()

  wandb.finish()


if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()
  main(args)
  # python _01_code/_07_cnn/a_mnist_train_cnn.py --wandb -b 2048 -r 1e-3 -v 10
  # python _01_code/_07_cnn/a_mnist_train_cnn.py --no-wandb -b 2048 -r 1e-3 -v 10