import torch

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


class ShallowLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, scaler=None):
        super(ShallowLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.scaler = scaler

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

    def predict(self, x):
        if self.scaler is not None:
            x = self.scaler.transform(x)
        x = self.forward(x)
        if self.scaler is not None:
            x = self.scaler.inverse_transform(x)
        return x

class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 1e-3):
        self.patience = patience
        self.delta = delta
        self.best_score = np.inf
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model, model_path: str = None):
        if val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0
            if model_path is not None:
                torch.save(model, model_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_data(data: pd.DataFrame, lookback=1, lookahead=1):
    feature_columns = ['ws_x', 'ws_y', 'ti', 'rho']
    target_columns = ['ws_x', 'ws_y']
    features = []
    targets = []
    for i in trange(lookback, data.shape[0] - lookahead):
        features.append(data.iloc[i - lookback:i][feature_columns].T.values.flatten().tolist())
        targets.append(data.iloc[i:i + lookahead][target_columns].T.values.flatten().tolist())

    return torch.tensor(features, requires_grad=True), torch.tensor(targets)

def main():
    # Load data
    print("Loading data...")
    data = pd.read_pickle('./data/Horns_rev1.pkl')

    # Preprocess
    print("Preprocessing data...")
    data['ws_x'] = data.apply(lambda x: x.ws * np.cos(np.pi * x.wd / 180), axis=1)
    data['ws_y'] = data.apply(lambda x: x.ws * np.sin(np.pi * x.wd / 180), axis=1)

    # Split into train-validation-test
    print("Splitting data...")
    samples_in_year = 365 * 24 * 6
    train_data = data.iloc[-3 * samples_in_year:-samples_in_year]
    test_data = data.iloc[-samples_in_year:]
    train_data, val_data = train_test_split(train_data, test_size=0.2, shuffle=False)

    # Scale features
    print("Scaling data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)

    train_dataset = pd.DataFrame(scaler.transform(train_data), columns=train_data.columns)
    val_dataset = pd.DataFrame(scaler.transform(val_data), columns=val_data.columns)

    print("Building datasets...")
    x_train, y_train = build_data(train_dataset, lookback=6 * 24 * 5, lookahead=6 * 24 * 2)
    x_val, y_val = build_data(val_dataset, lookback=6 * 24 * 5, lookahead=6 * 24 * 2)

    train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=4096, shuffle=True)

    # Define model, optimizer and loss function
    print("Defining model...")
    model = ShallowLSTM(6 * 24 * 5 * 4, 6 * 24 * 5, 6 * 24 * 2 * 2, scaler=scaler)
    optimizer = Adam(model.parameters())
    loss_function = nn.MSELoss()
    early_stopping = EarlyStopping(patience=5, delta=0.001)

    # Picking device
    device = pick_device()
    print(f"Using device: {device}...")
    model = model.to(device)

    # Run training
    n_epochs = 100
    print(f"Training model for {n_epochs} epochs...")
    for epoch in trange(n_epochs):
        print(f" - Starting epoch #{epoch}")
        model.train()
        for i, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            y_hat = model.forward(x)
            loss = loss_function(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f" - Epoch #{epoch}: batch {i}/{len(train_dataloader)} MSE {loss:.4f}")

        model.eval()
        with torch.no_grad():
            y_hat = model.forward(x_train.to(device))
            train_loss = loss_function(y_hat, y_train.to(device)).item()
            y_hat = model.forward(x_val.to(device))
            val_loss = loss_function(y_hat, y_val.to(device)).item()
        print(f"Epoch #{epoch}: train MSE {train_loss:.4f}, validation MSE {val_loss:.4f}")

        early_stopping(val_loss, model, model_path=f'./models/ShallowLSTM_epoch{epoch}_vloss{val_loss:.4f}.pth')
        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == "__main__":
    main()
