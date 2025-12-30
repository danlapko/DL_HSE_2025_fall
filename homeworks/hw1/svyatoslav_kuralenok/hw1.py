import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super().__init__()

        self.model = nn.Sequential()
        self.model.add_module(f"fc{0}", nn.Linear(in_features, hidden_features))
        self.model.add_module(f"relu{0}", nn.ReLU())
        self.model.add_module(f"fc{1}", nn.Linear(hidden_features, hidden_features // 4))
        self.model.add_module(f"relu{1}", nn.ReLU())
        self.model.add_module(f"fc{2}", nn.Linear(hidden_features // 4, out_features))
        self.model.add_module(f"softmax{2}", nn.Softmax(dim=-1))

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
        return torch.argmax(logits, dim=1, keepdim=True)


def normalize_tensor_data(tensor: torch.Tensor, with_mean: bool, with_variance: bool):
    eps=1e-8
    mean = torch.mean(tensor, dim=0, keepdim=True)
    std = torch.std(tensor, dim=0, keepdim=True)
    if with_mean:
        tensor -= mean
    if with_variance:
        tensor /= std + eps
    return tensor


def load_data(train_csv, val_csv, test_csv):
    ### YOUR CODE HERE
    ### TODO: add velocity/accele vector length?

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    X_train = torch.tensor(train_df.iloc[:, :-3].values, dtype=torch.float32)
    X_train = normalize_tensor_data(X_train, True, True)
    y_train = torch.tensor(train_df[f"order0"].values, dtype=torch.long)

    X_val = torch.tensor(val_df.iloc[:, :-3].values, dtype=torch.float32)
    X_val = normalize_tensor_data(X_val, True, True)
    y_val = torch.tensor(val_df[f"order0"].values, dtype=torch.long)

    X_test = torch.tensor(test_df.values, dtype=torch.float32)
    X_test = normalize_tensor_data(X_test, True, True)
    y_test = torch.tensor(np.empty((0, 1)), dtype=torch.long)

    return (
        X_train.to(device),
        y_train.to(device),
        X_val.to(device),
        y_val.to(device),
        X_test.to(device),
        y_test.to(device),
    )


def init_model(lr, in_features, hidden_features):
    ### YOUR CODE HERE

    model = MLP(in_features=in_features, hidden_features=hidden_features, out_features=3)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    return model, criterion, optimizer


def evaluate(model, X, y):
    ### YOUR CODE HERE

    predictions = model.predict(X).cpu().detach()
    accuracy = accuracy_score(y.cpu(), predictions)
    conf_matrix = confusion_matrix(y.cpu(), predictions)

    return predictions, accuracy, conf_matrix


def train(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size):
    ### YOUR CODE HERE: train the model and validate it every epoch on X_val, y_val

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    best_epoch = 0
    best_accuracy = 0
    best_model_state = None
    for epoch in range(0, epochs):
        model.train()
        train_loss = 0
        for _, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_size
        train_loss /= X_train.shape[0]
        if epoch % 1 == 0:
            _, accuracy, _ = evaluate(model, X_val, y_val)
            print(f'Epoch {epoch}:\n\tTrain Loss: {np.exp(train_loss)}\n\tValidate accuracy: {accuracy}')
            if accuracy > best_accuracy:
                best_epoch = epoch
                best_accuracy = accuracy
                best_model_state = model.state_dict()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f"Best epoch: {best_epoch}; best accuracy on validate dataset: {best_accuracy}")

    return model


def main(args):
    ### YOUR CODE HERE

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.train_csv, args.val_csv, args.test_csv)

    # Initialize model
    model, criterion, optimizer = init_model(lr=args.lr, in_features=X_train.shape[1], hidden_features=1024)

    # Train model
    model = train(model, criterion, optimizer, X_train, y_train, X_val, y_val, args.num_epoches, args.batch_size)

    # Predict on test set
    y_test = model.predict(X_test)

    # dump predictions to 'submission.csv'
    df = pd.DataFrame(y_test.cpu().detach().numpy(), columns=['order0'])
    df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", default="homeworks/hw1/data/train.csv")
    parser.add_argument("--val_csv", default="homeworks/hw1/data/val.csv")
    parser.add_argument("--test_csv", default="homeworks/hw1/data/test.csv")
    parser.add_argument("--out_csv", default="homeworks/hw1/svyatoslav_kuralenok/submission.csv")
    parser.add_argument("--lr", default=3e-4)
    parser.add_argument("--batch_size", default=512)
    parser.add_argument("--num_epoches", default=200)

    args = parser.parse_args()
    main(args)
