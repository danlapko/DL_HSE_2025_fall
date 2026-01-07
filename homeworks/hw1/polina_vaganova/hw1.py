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
    
    # Добавим новые признаки    
    def add_features(df):
        new_features = []
        
        for year in range(30):
            for body in range(3):
                x_col = f'y{year}_b{body}_x'
                y_col = f'y{year}_b{body}_y'
                vx_col = f'y{year}_b{body}_vx'
                vy_col = f'y{year}_b{body}_vy'
                
                if all(col in df.columns for col in [x_col, y_col, vx_col, vy_col]):
                    # Расстояние от центра (более тяжелые звезды ближе к центру)
                    dist_feature = pd.DataFrame({f'y{year}_b{body}_dist': np.sqrt(df[x_col]**2 + df[y_col]**2)})
                    new_features.append(dist_feature)
                    
                    # Скорость (более тяжелые звезды движутся медленнее)
                    speed_feature = pd.DataFrame({f'y{year}_b{body}_speed': np.sqrt(df[vx_col]**2 + df[vy_col]**2)})
                    new_features.append(speed_feature)
        
        for body in range(3):
            for feature in ['_x', '_y', '_vx', '_vy']:
                series = []
                valid_cols = []
                for year in range(30):
                    col_name = f'y{year}_b{body}{feature}'
                    if col_name in df.columns:
                        series.append(df[col_name])
                        valid_cols.append(col_name)
                
                if len(series) > 0:
                    series_df = pd.DataFrame(series).T  
                    # Скользящее среднее и стандартное отклонение с окном 10 лет
                    rolling_mean = series_df.T.rolling(window=10, min_periods=1).mean().T
                    rolling_std = series_df.T.rolling(window=10, min_periods=1).std().T
                    
                    for i in [9, 19, 29]:  
                        if i < rolling_mean.shape[1]:
                            mean_feature = pd.DataFrame({f'b{body}{feature}_rolling_mean_{i}': rolling_mean.iloc[:, i]})
                            std_feature = pd.DataFrame({f'b{body}{feature}_rolling_std_{i}': rolling_std.iloc[:, i].fillna(0)})
                            new_features.append(mean_feature)
                            new_features.append(std_feature)
        
        all_new_features = pd.concat(new_features, axis=1)
        return pd.concat([df, all_new_features], axis=1)
    
    train_df = add_features(train_df)
    val_df = add_features(val_df)
    test_df = add_features(test_df)
    
    train_cols_to_drop = []
    val_cols_to_drop = []
    test_cols_to_drop = []
    
    y_train = train_df['order0'].values
    for col in ['order0', 'order1', 'order2']:
        if col in train_df.columns:
            train_cols_to_drop.append(col)

    y_val = val_df['order0'].values
    for col in ['order0', 'order1', 'order2']:
        if col in val_df.columns:
            val_cols_to_drop.append(col)

    X_train = train_df.drop(columns=train_cols_to_drop).values
    X_val = val_df.drop(columns=val_cols_to_drop).values
    X_test = test_df.drop(columns=test_cols_to_drop).values
    y_test = None
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3) 
        )
    
    def forward(self, x):
        return self.layers(x)


def init_model(input_dim, lr=0.001):
    model = MLP(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    return model, criterion, optimizer


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        outputs = model(X_tensor)
        predictions = torch.argmax(outputs, dim=1).numpy()
    
    if y is not None:
        accuracy = accuracy_score(y, predictions)
    else:
        accuracy = None
    
    return predictions, accuracy


def train(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs):
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            _, val_accuracy = evaluate(model, X_val, y_val)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {val_accuracy:.4f}')
    
    return model


def main(args):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        args.train_csv, args.val_csv, args.test_csv
    )
    
    input_dim = X_train.shape[1]
    model, criterion, optimizer = init_model(input_dim, args.lr)
    
    model = train(model, criterion, optimizer, X_train, y_train, X_val, y_val, args.num_epoches)
    
    _, val_accuracy = evaluate(model, X_val, y_val)
    print(f'Final Validation Accuracy: {val_accuracy:.4f}')
    
    test_predictions, _ = evaluate(model, X_test, None)
    
    submission = pd.DataFrame({
        'order0': test_predictions
    })

    submission.to_csv(args.out_csv, index=False)
    print(f"Predictions saved to {args.out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', default='../data/train.csv')
    parser.add_argument('--val_csv', default='../data/val.csv')
    parser.add_argument('--test_csv', default='../data/test.csv')
    parser.add_argument('--out_csv', default='../data/submission.csv')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epoches', type=int, default=200)
    args = parser.parse_args()
    main(args)