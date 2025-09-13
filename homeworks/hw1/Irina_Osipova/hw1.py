import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from os import path
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler


def check_path(file):
    if not path.exists(file):
        print(
            f"\033[33m{path.basename(__file__)} load_data: Внимание: файл {file} не найден. "
            "Он не будет обработан\033[0m"
        )
        return 0
    return 1


def load_data(train_csv, val_csv, test_csv):

    ss = StandardScaler()

    print("Загрузка тренировочных данных")
    if check_path(train_csv):
        train_df = pd.read_csv(train_csv)
        # target - колонка order0. Будем определять только одну звезду. Колнки order1, order2 - не используются.
        X_train = train_df.drop(columns=["order0", "order1", "order2"])
        y_train = train_df["order0"]
        X_train = ss.fit_transform(X_train.values)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.int64)
    else:
        return None, None, None, None, None

    print("Загрузка данных для валидации")
    if check_path(val_csv):
        val_df = pd.read_csv(val_csv)
        X_val = val_df.drop(columns=["order0", "order1", "order2"])
        y_val = val_df["order0"]
        X_val = ss.transform(X_val.values)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val.values, dtype=torch.int64)
    else:
        X_val, y_val = None, None

    print("Загрузка тестовых данных")
    if check_path(test_csv):
        test_df = pd.read_csv(test_csv)
        X_test = ss.transform(test_df.values)
        X_test = torch.tensor(X_test, dtype=torch.float32)
    else:
        X_test = None

    return X_train, y_train, X_val, y_val, X_test


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(360, 720)
        self.fc3 = nn.Linear(720, 120)
        self.fc4 = nn.Linear(120, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def init_model(lr):
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer


def evaluate(model, criterion=None, X=None, y=None):
    model.eval()
    with torch.no_grad():
        outputs = model(X)

        predictions = outputs.argmax(dim=1)
        if y is None:
            return predictions, None, None, None
        else:
            accuracy = accuracy_score(y, predictions)
            conf_matrix = confusion_matrix(y, predictions)
            loss = criterion(outputs, y)
            return predictions, accuracy, conf_matrix, loss


def train(
    model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size
):
    # train the model and validate it every epoch on X_val, y_val
    train_losses = []
    best_val_loss = float("inf")
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    if X_val is None:
        print(
            f"\033[33mВнимание: нет данных для валидации. Модель будет проверена только на обучающих данных\033[0m"
        )
        val_loss, val_acc = None, None
        f = False
    else:
        f = True

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, els = 0, 0
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            els += X_batch.size(0)
            train_losses.append(train_loss / els)

        train_loss /= X_train.size(0)

        if f:
            _, val_acc, _, val_loss = evaluate(model, criterion, X_val, y_val)
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), "model.pth")
                best_val_loss = val_loss

        print(
            f"Epoch {epoch}, Train Loss: {train_losses[-1]}, Val Loss: {val_loss}, Val Accuracy: {val_acc}"
        )

    if not f:
        torch.save(model.state_dict(), "model.pth")

    return model


def main(args):
    print("Загрузка данных...")
    X_train, y_train, X_val, y_val, X_test = load_data(
        args.train_csv, args.val_csv, args.test_csv
    )

    if X_train is None:
        print(
            f"\033[33mВнимание: нет данных для обучения. Проверьте данные и перезапустите программу\033[0m"
        )
        return

    print("Инициализация модели...")
    model, criterion, optimizer = init_model(args.lr)

    print("Обучение модели...")
    model = train(
        model,
        criterion,
        optimizer,
        X_train,
        y_train,
        X_val,
        y_val,
        args.num_epoches,
        args.batch_size,
    )

    print("Тестирование модели...")
    model.load_state_dict(torch.load("model.pth"))

    if X_val is not None:
        _, val_acc, _, val_loss = evaluate(model, criterion, X_val, y_val)
        print(f"Лучший результат: Val Loss: {val_loss}, Val Accuracy: {val_acc}")

    print("Модель сохранена в файл model.pth")

    if X_test is None:
        print(
            f"\033[33mВнимание: нет данных для тестирования. Программа завершила работу без этапа тестирования\033[0m"
        )
    else:
        y_pred, _, _, _ = evaluate(model=model, X=X_test)
        # dump predictions to 'submission.csv'
        pd.DataFrame(y_pred, columns=["order0"]).to_csv(
            args.out_csv,
            index=False,
        )
        print("Результаты успешно загружены в файл submission.csv")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", default="../data/train.csv")
    parser.add_argument("--val_csv", default="../data/val.csv")
    parser.add_argument("--test_csv", default="../data/test.csv")
    parser.add_argument("--out_csv", default="submission.csv")
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--num_epoches", default=15)

    args = parser.parse_args()
    main(args)
