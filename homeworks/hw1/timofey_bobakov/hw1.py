import argparse

import pandas as pd
import torch
import torch.nn as nn
import random
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import os
os.chdir(r'C:\Users\Asus\1codes\DL\DL_HSE_2025_fall')


def load_data(type, train_csv, val_csv, test_csv):
    # Load data
    test_df = pd.read_csv(test_csv)
    val_df = pd.read_csv(val_csv)
    train_df = val_df if type == "fast" else pd.read_csv(train_csv)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)               #SHUFFLE train data
    
    print("\nloaded data\n")

    # Function to add std, max, and median features to any dataset using PyTorch
    def add_features(data):
        # Convert to tensor if not already
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        all_new_features = []
        
        # Process each of the 3 body types
        for body_type in range(3):
            start_idx = body_type * 4  # 0, 4, 8
            
            # Get coordinates for this body type (every 12 columns starting from start_idx)
            x_coords = data[:, start_idx::12]        # x: 0,12,24... or 4,16,28... or 8,20,32...
            y_coords = data[:, start_idx+1::12]      # y: 1,13,25... or 5,17,29... or 9,21,33...
            vx_coords = data[:, start_idx+2::12]     # vx: 2,14,26... or 6,18,30... or 10,22,34...
            vy_coords = data[:, start_idx+3::12]     # vy: 3,15,27... or 7,19,31... or 11,23,35...
            
            # Calculate features for this body type
            stdev_x = torch.std(x_coords, dim=1, unbiased=False)
            stdev_y = torch.std(y_coords, dim=1, unbiased=False)
            max_vx = torch.max(vx_coords, dim=1).values
            max_vy = torch.max(vy_coords, dim=1).values
            median_vx = torch.median(vx_coords, dim=1).values
            median_vy = torch.median(vy_coords, dim=1).values
            
            # Stack features for this body type
            body_features = torch.stack([
                stdev_x, stdev_y, 
                max_vx, max_vy,
                median_vx, median_vy
            ], dim=1)
            
            all_new_features.append(body_features)
        
        # Concatenate features from all 3 body types
        all_new_features_tensor = torch.cat(all_new_features, dim=1)
        
        # Concatenate original data with all new features
        return torch.cat([data, all_new_features_tensor], dim=1)


    # Process all datasets
    datasets = {}
    for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        X_original = df[df.columns[0:-3]].values if name != 'test' else df[df.columns[0:]].values
        X_with_features = add_features(X_original)
        
        datasets[f'X_{name}'] = X_with_features
        
        if name in ['train', 'val']:
            # Convert labels to torch tensors
            datasets[f'y_{name}'] = torch.tensor(df['order0'].values, dtype=torch.long)


    return datasets['X_train'], datasets['y_train'], datasets['X_val'], datasets['y_val'], datasets['X_test']


class MLP(nn.Module):
    def __init__(self, A, B, dropout_rate=0.15):
        super(MLP, self).__init__()
        self.bn0 = nn.BatchNorm1d(378)
        self.fc1 = nn.Linear(378, A)
        self.bn1 = nn.BatchNorm1d(A) 
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(A, B)
        self.bn2 = nn.BatchNorm1d(B)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(B, 3)

    def forward(self, x):
        x = self.bn0(x)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def init_model(lr, A, B):
    model = MLP(A, B)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer



def evaluate(model, X, y):

    model.eval()
    with torch.no_grad():
        outputs = model(X)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y).float().mean().item() #item returns a py- int
        
        # Create confusion matrix
        conf_matrix = torch.zeros(3, 3, dtype=torch.int32)
        for true_label, pred_label in zip(y, predictions):
            conf_matrix[true_label, pred_label] += 1
            
    return predictions, accuracy, conf_matrix


def train(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size):
    """
    Train the model with early stopping based on validation loss.
    Stops training if validation loss doesn't decrease by at least 0.01 
    for 30 consecutive epochs compared to the minimum validation loss encountered.
    """
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_weights = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Training loop
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]
            
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 250 == 0 and epoch % 5 == 0:
                print(f"\t Train: Epoch {epoch}, train Loss: {loss.item()}")
            
            train_loss += loss.item() * batch_size

        train_loss /= X_train.size(0)
        

        ### Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).squeeze()
            val_loss = criterion(val_outputs, y_val)
            current_val_loss = val_loss.item()
        
        # Print training progress
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {current_val_loss:.4f}")
        
            # Early stopping logic
            if current_val_loss < best_val_loss - 0.02: 
                best_val_loss = current_val_loss
                epochs_without_improvement = 0
                # Save the best model weights
                best_model_weights = model.state_dict().copy()
                print(f"✓ New best validation loss: {best_val_loss:.4f}")
            else:
                epochs_without_improvement += 5
        
            # Check for early stopping condition
            if epochs_without_improvement >= 50:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                print(f"Best validation loss: {best_val_loss:.4f}")
                print(f"No improvement for {epochs_without_improvement} epochs")
                break
        
    # Load the best model weights before returning
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print("Loaded best model weights from training")
    
    return model

def main(args):
    
    train_csv = args.train_csv
    val_csv = args.val_csv
    test_csv = args.test_csv
    out_csv = args.out_csv
    learning_rate = args.lr
    batch_size = args.batch_size
    epoches = args.num_epoches
    
    
    epoches = 1000              
    batch_size = 1024
    learning_rate = 0.001
    A = 256
    B = 32

    X_train, y_train, X_val, y_val, X_test = load_data("ok", train_csv, val_csv, test_csv)

    # Initialize model
    model, criterion, optimizer = init_model(learning_rate, A, B)
    # Train model
    model = train(model, criterion, optimizer, X_train, y_train, X_val, y_val, epoches, batch_size)
    

    predictions, accuracy, _ = evaluate(model, X_val, y_val)
    print(f"Val_accuracy = {accuracy}\n")
    # Predict on test set

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1)

    # dump predictions to 'submission.csv'
    predictions_df = pd.DataFrame({'order0': predictions.numpy()})
    predictions_df.to_csv(out_csv, index=False)
    print("Predictions saved to submission.csv")

    # Verify the format
    print("\nFirst 10 predictions:")
    print(predictions_df.head(10))
    




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
