import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from datetime import datetime
import wandb
import argparse
import pandas as pd

from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
print(BASE_PATH)
import sys
sys.path.append(BASE_PATH)

from _00_homework.hw2.a_titianic_dataset import get_preprocessed_dataset


def get_data():
    train_dataset, validation_dataset, test_dataset = get_preprocessed_dataset()
    print(len(train_dataset), len(validation_dataset), len(test_dataset))

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset))
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))

    return train_data_loader, validation_data_loader, test_data_loader


class MyModel(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_input, wandb.config.n_hidden_unit_list[0]),
            nn.ReLU(),
            nn.Linear(wandb.config.n_hidden_unit_list[0], wandb.config.n_hidden_unit_list[1]),
            nn.ReLU(),
            nn.Linear(wandb.config.n_hidden_unit_list[1], n_output),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def get_model_and_optimizer():
    my_model = MyModel(n_input=11, n_output=2)
    optimizer = optim.SGD(my_model.parameters(), lr=wandb.config.learning_rate)

    return my_model, optimizer


def training_loop(model, optimizer, train_data_loader, validation_data_loader):
    n_epochs = wandb.config.epochs
    loss_fn = nn.CrossEntropyLoss() # Use a built-in loss function
    next_print_epoch = 100

    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        num_trains = 0
        for train_batch in train_data_loader:
            output_train = model(train_batch['input'])
            loss = loss_fn(output_train, train_batch['target'])
            loss_train += loss.item()
            num_trains += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_validation = 0.0
        num_validations = 0
        with torch.no_grad():
            for validation_batch in validation_data_loader:
                output_validation = model(validation_batch['input'])
                loss = loss_fn(output_validation, validation_batch['target'])
                loss_validation += loss.item()
                num_validations += 1

        wandb.log({
            "Epoch": epoch,
            "Training loss": loss_train / num_trains,
            "Validation loss": loss_validation / num_validations
        })

        if epoch >= next_print_epoch:
            print(
                f"Epoch {epoch}, "
                f"Training loss {loss_train / num_trains:.4f}, "
                f"Validation loss {loss_validation / num_validations:.4f}"
            )
            next_print_epoch += 100


def test(test_data_loader, model):
    print("[TEST]")
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_data_loader))
        output_batch = model(batch['input'])
        #print(output_batch)
        prediction_batch = torch.argmax(output_batch, dim=1)
        labels = prediction_batch.numpy()

    passenger_ids = list(range(892, 892 + len(prediction_batch)))
    submission_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': labels
    })
    #print(submission_df)
    submission_df.to_csv('submission.csv', index=False)


def main(args):
    current_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': 1e-3,
        'n_hidden_unit_list': [30, 30],
    }

    wandb.init(
        mode="online" if args.wandb else "disabled",
        project="comparison_activation_function_ReLU",
        notes="My first wandb experiment",
        tags=["my_model", "titanic"],
        name=current_time_str,
        config=config
    )
    print(args)
    print(wandb.config)

    train_data_loader, validation_data_loader, test_data_loader = get_data()

    linear_model, optimizer = get_model_and_optimizer()

    wandb.watch(linear_model)

    print("#" * 50, 1)

    training_loop(
        model=linear_model,
        optimizer=optimizer,
        train_data_loader=train_data_loader,
        validation_data_loader=validation_data_loader
    )
    test(test_data_loader, linear_model)
    wandb.finish()


# https://docs.wandb.ai/guides/track/config
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=False, help="True or False"
    )

    parser.add_argument(
        "-b", "--batch_size", type=int, default=16, help="Batch size (int, default: 16)"
    )

    parser.add_argument(
        "-e", "--epochs", type=int, default=7_000, help="Number of training epochs (int, default:1_000)"
    )

    args = parser.parse_args()

    main(args)

