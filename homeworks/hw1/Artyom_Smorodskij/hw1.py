import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(360, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def load_data(train_csv, val_csv, test_csv):

    y_train = train_csv['order0'].values
    X_train = train_csv.drop(['order0', 'order1', 'order2'], axis=1).values

    y_val = val_csv['order0'].values
    X_val = val_csv.drop(['order0', 'order1', 'order2'], axis=1).values

    X_test = test_csv.values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_val, y_val, X_test


def init_model(learning_rate):
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer


def evaluate(model, X, y):
    ### YOUR CODE HERE
    return predictions, accuracy, conf_matrix


def train(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size):
    best_accuracy = 0
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        perm = torch.randperm(len(X_train))

        for i in range(0, X_train.size(0), batch_size):
            optimizer.zero_grad()
            indices = perm[i:i + batch_size]
            X_batch = X_train[indices]
            y_batch = y_train[indices]

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            # if i % 100 == 0:
            #     print(f'\t Train: Epoch {epoch}, train Loss: {loss.item()}')
            train_loss += loss.item()

        train_loss /= X_train.size(0)

        model.eval()
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        _, val_pred = torch.max(val_outputs, 1)
        val_accuracy = accuracy_score(y_val.numpy(), val_pred.numpy())

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model.state_dict().copy()

        if epoch % 10 == 0:
            print(f'Val: Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss.item()}, Val Accuracy: {val_accuracy}')

    model.load_state_dict(best_model)

    return model


def main(args):
    ### YOUR CODE HERE

    # Load data
    train_csv = pd.read_csv(args.train_csv, delimiter=",")
    val_csv = pd.read_csv(args.val_csv, delimiter=",")
    test_csv = pd.read_csv(args.test_csv, delimiter=",")

    X_train, y_train, X_val, y_val, X_test = load_data(train_csv, val_csv, test_csv)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Initialize model
    model, criterion, optimizer = init_model(args.lr)

    # Train model
    model = train(model, criterion, optimizer, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, args.num_epoches, args.batch_size)

    # Predict on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, test_predictions = torch.max(test_outputs, 1)
        test_predictions = test_predictions.numpy()

    # dump predictions to 'submission.csv'
    output = pd.DataFrame({'order0': test_predictions})
    output.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_csv', default='../data/train.csv')
    parser.add_argument('--val_csv', default='../data/val.csv')
    parser.add_argument('--test_csv', default='../data/test.csv')
    parser.add_argument('--out_csv', default='../data/submission.csv')
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--num_epoches', default=200)

    args = parser.parse_args()
    main(args)
