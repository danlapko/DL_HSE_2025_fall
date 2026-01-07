import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings

num_epoches = 1024
batch_size = 2048

learning_rate = 1e-4

features = 360
num_classes = 3


warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
def create_features(X):
    X_new = X.copy()

    for body in range(3):
        for year in range(30):
            x = f'y{year}_b{body}_x'
            y = f'y{year}_b{body}_y'
            vx = f'y{year}_b{body}_vx'
            vy = f'y{year}_b{body}_vy'

            X_new[f'y{year}_b{body}_x_sq'] = X[vx]**2 - X[vy]**2
            # X_new[f'y{year}_b{body}_y_sq'] = X[vy]**2

            X_new[f'y{year}_b{body}_xvy_mult'] = X[x] * X[vy] - X[y] * X[vx]
            # X_new[f'y{year}_b{body}_yvx_mult'] = X[y] * X[vx]

            x_0 = (X[f'y{year}_b0_x'] + X[f'y{year}_b1_x'] + X[f'y{year}_b2_x']) / 3
            y_0 = (X[f'y{year}_b0_y'] + X[f'y{year}_b1_y'] + X[f'y{year}_b2_y']) / 3

            distance = np.sqrt((X[x] - x_0)**2 + (X[y] - y_0)**2)
            X_new[f'y{year}_b{body}_dist'] = distance
    return X_new

def load_data(train_csv, val_csv, test_csv):
    df_train = pd.read_csv(train_csv)
    X_train = df_train.drop(columns = ["order0", "order1", "order2"])
    X_train = create_features(X_train)
    global features
    features = X_train.shape[1]
    y_train = df_train["order0"]

    df_val = pd.read_csv(val_csv)
    X_val = df_val.drop(columns=["order0", "order1", "order2"])
    X_val = create_features(X_val)
    y_val = df_val["order0"]

    X_test = pd.read_csv(test_csv)
    X_test = create_features(X_test)

    return X_train, y_train, X_val, y_val, X_test #, y_test


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(features, 512)
        self.dropout1 = nn.Dropout(0.6)  #
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 64)
        self.dropout3 = nn.Dropout(0.4)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


def init_model():
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    return model, criterion, optimizer


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        test_outputs = model(X)
        predicted_classes = torch.argmax(test_outputs, dim=1)

        accuracy, conf_matrix = None, None
        if y is not None:
            accuracy = accuracy_score(y, predicted_classes)
            print(f'Test accuracy: {accuracy}')

            conf_matrix = confusion_matrix(y, predicted_classes)

    return predicted_classes, accuracy, conf_matrix


def train(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs):
    num_batches = (X_train.size(0) + batch_size - 1) // batch_size
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0 and epoch % 100 == 0:
                print(f'\t Train: Epoch {epoch}, train Loss: {loss.item()}')
            train_loss += loss.item()

        train_loss /= num_batches

        if epoch % 100 == 0:
            model.eval()
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            print(f'Val: Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss.item()}')

            train_pred = torch.argmax(model(X_train), dim=1)
            val_pred = torch.argmax(val_outputs, dim=1)
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    return model


def main(args):
    # Load data
    X_train, y_train, X_val, y_val, X_test = load_data(
        args.train_csv, args.val_csv, args.test_csv
    )

    global num_classes
    num_classes  = len(y_train.unique())

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    # Initialize model
    model, criterion, optimizer = init_model()

    # Train model
    train(model, criterion, optimizer, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, num_epoches)

    # Predict on test set
    evaluate(model, X_val_tensor, y_val)
    prediction, _, _  = evaluate(model, X_test_tensor, None)

    # dump predictions to 'submission.csv'
    df = pd.DataFrame({'order0': prediction.numpy()})
    df.to_csv(args.out_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_csv', default='homeworks/hw1/data/train.csv')
    parser.add_argument('--val_csv', default='homeworks/hw1/data/val.csv')
    parser.add_argument('--test_csv', default='homeworks/hw1/data/test.csv')
    parser.add_argument('--out_csv', default='homeworks/hw1/data/submission.csv')
    parser.add_argument('--lr', default=0)
    parser.add_argument('--batch_size', default=0)
    parser.add_argument('--num_epoches', default=0)

    args = parser.parse_args()
    main(args)
