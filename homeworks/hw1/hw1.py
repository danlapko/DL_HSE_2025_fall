import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def load_data(train_csv, val_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    feature_cols = [col for col in train_df.columns if col not in ['order0', 'order1', 'order2']]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['order0'].values.astype(np.int64)

    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df['order0'].values.astype(np.int64)

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = None

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.3):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def init_model(input_size, lr):
    model = MLP(input_size, [512, 256, 128, 64], 3, dropout=0.3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    return model, criterion, optimizer


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X)
        outputs = model(X_tensor)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.numpy()
        accuracy = accuracy_score(y, predictions)
        conf_matrix = confusion_matrix(y, predictions)
    return predictions, accuracy, conf_matrix


def train(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size):
    n_samples = len(X_train)

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = torch.tensor(X_train[batch_indices])
            y_batch = torch.tensor(y_train[batch_indices])

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        _, val_accuracy, _ = evaluate(model, X_val, y_val)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    return model


def main(args):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        args.train_csv, args.val_csv, args.test_csv
    )

    input_size = X_train.shape[1]

    model, criterion, optimizer = init_model(input_size, args.lr)

    model = train(model, criterion, optimizer, X_train, y_train, X_val, y_val,
                  args.num_epoches, args.batch_size)

    _, val_accuracy, val_conf_matrix = evaluate(model, X_val, y_val)
    print(f'\nFinal Validation Accuracy: {val_accuracy:.4f}')
    print('Confusion Matrix:')
    print(val_conf_matrix)

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test)
        outputs = model(X_test_tensor)
        _, test_predictions = torch.max(outputs, 1)
        test_predictions = test_predictions.numpy()

    submission_df = pd.DataFrame({'order0': test_predictions})
    submission_df.to_csv(args.out_csv, index=False)
    print(f'\nPredictions saved to {args.out_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_csv', default='data/train.csv')
    parser.add_argument('--val_csv', default='data/val.csv')
    parser.add_argument('--test_csv', default='data/test.csv')
    parser.add_argument('--out_csv', default='submission.csv')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epoches', type=int, default=100)

    args = parser.parse_args()
    main(args)
