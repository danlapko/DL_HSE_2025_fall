import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import logging
from torch.utils.data import TensorDataset, DataLoader
import os


logging.basicConfig(
    filename='training_tmp.log',  
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)


def load_data(train_csv, val_csv, test_csv):
    scaler = StandardScaler()
    train_df = pd.read_csv(train_csv).drop(columns=['order1', 'order2'])
    train_df_np = scaler.fit_transform(train_df.iloc[:, :-1])
    X_train = torch.tensor(train_df_np, dtype=torch.float32)
    y_train = torch.tensor(train_df.iloc[:, -1].values, dtype=torch.long)

    val_df = pd.read_csv(val_csv).drop(columns=['order1', 'order2'])
    val_df_np = scaler.transform(val_df.iloc[:, :-1])
    X_val = torch.tensor(val_df_np, dtype=torch.float32)
    y_val = torch.tensor(val_df.iloc[:, -1].values, dtype=torch.long)

    test_df = pd.read_csv(test_csv)
    test_df = scaler.transform(test_df)
    X_test = torch.tensor(test_df, dtype=torch.float32)
    return X_train, y_train, X_val, y_val, X_test


def init_model(lr=None):
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(360, 128)
            self.dropout1 = nn.Dropout(0.3)
            self.fc2 = nn.Linear(128, 64)
            self.dropout2 = nn.Dropout(0.3)
            self.fc3 = nn.Linear(64, 3)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout1(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout2(x)
            x = self.fc3(x)
            return x
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer


def evaluate(model, X, y=None):
    accuracy, conf_matrix = None, None
    model.eval()
    with torch.no_grad():
        test_outputs = model(X)
        predictions = torch.argmax(test_outputs, dim=1).cpu().numpy()
    if y is not None:
        accuracy = accuracy_score(y.cpu().numpy(), predictions)
        conf_matrix = confusion_matrix(y.cpu().numpy(), predictions)
    return predictions, accuracy, conf_matrix


def train(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size, r_s):
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    best_state = None
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                logging.info(f'Epoch {epoch}, Batch {i}, Train Loss: {loss.item():.4f}')

            train_loss += loss.item() * X_batch.size(0)
        train_loss /= X_train.size(0)
        _, accuracy, conf_matrix = evaluate(model, X_val, y_val)
        logging.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {accuracy:.4f}, Val Conf Matrix:\n{conf_matrix}')
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    logging.info(f'Best_val_acc: {best_val_acc:.4f}')
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)

    old_name = 'training_tmp.log'
    new_name = f'training_{best_val_acc:.4f}_{r_s}.log'
    os.rename(old_name, new_name)
    return model, best_val_acc


def main(args):
    r_s = 40
    torch.manual_seed(r_s)
    np.random.seed(r_s)
    X_train, y_train, X_val, y_val, X_test = load_data(args.train_csv, args.val_csv, args.test_csv)
    model, criterion, optimizer = init_model(args.lr)
    train_model, best_val_acc = train(model, criterion, optimizer, X_train, y_train, X_val, y_val, args.num_epoches, args.batch_size, r_s)
    torch.save(train_model.state_dict(), f'mlp_weights_{best_val_acc:.4f}_{r_s}.pth')
    predictions, _, _ = evaluate(train_model, X_test)
    df = pd.DataFrame(predictions, columns=["order0"])  

    submission_file = f'submission_{best_val_acc:.4f}_{r_s}.csv'
    df.to_csv(submission_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_csv', default='homeworks/hw1/data/train.csv')
    parser.add_argument('--val_csv', default='homeworks/hw1/data/val.csv')
    parser.add_argument('--test_csv', default='homeworks/hw1/data/test.csv')
    parser.add_argument('--out_csv', default='homeworks/hw1/lisa_simakova/submission.csv')
    parser.add_argument('--lr', default=0.0003)
    parser.add_argument('--batch_size', default=512)
    parser.add_argument('--num_epoches', default=100)

    args = parser.parse_args()
    main(args)
