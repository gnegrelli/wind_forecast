import pickle
import torch

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


class ShallowLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ShallowLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = x.reshape(2, x.shape[0] / 2)
        return x

def build_data(data: pd.DataFrame, lookback=1, lookahead=1):
    feature_columns = ['ws_x', 'ws_y', 'ti', 'rho']
    target_columns = ['ws_x', 'ws_y']
    features = []
    targets = []
    for i in range(lookback, data.shape[0] - lookahead - 1):
        features.append(data.iloc[i - lookback:i + 1][feature_columns])
        targets.append(data.iloc[i + 1:i + 1 + lookahead][target_columns])

    return torch.tensor(features, requires_grad=True), torch.tensor(targets)

def main():
   # Load data
   with  open('data/Horns_rev1.pkl', 'rb') as file:
     data = pickle.load(file)

   # Preprocess
   data['ws_x'] = data.apply(lambda x: x.ws * np.cos(np.pi * x.wd / 180), axis=1)
   data['ws_y'] = data.apply(lambda x: x.ws * np.sin(np.pi * x.wd / 180), axis=1)

   # Split into train-validation-test
   train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=False)

   # Scale features
   scaler = MinMaxScaler(feature_range=(0, 1))
   scaler.fit(train_data)

   train_dataset = pd.DataFrame(scaler.transform(train_data), columns=train_data.columns)
   test_dataset = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)

   x_train, y_train = build_data(train_dataset, lookback=6 * 24 * 5, lookahead=6 * 24 * 2)
   x_test, y_test = build_data(train_dataset, lookback=6 * 24 * 5, lookahead=6 * 24 * 2)

   train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=2048, shuffle=True)
   test_dataloader = DataLoader(TensorDataset(x_test, y_test), batch_size=256, shuffle=False)

   # Define model, optimizer and loss function
   model = ShallowLSTM(6 * 24 * 5, 6 * 24 * 5, 6 * 24 * 2 * 2)
   optimizer = Adam(model.parameters())
   loss = nn.MSELoss()
   
   # Run training
   n_epochs = 100
   for epoch in range(n_epochs):
       model.train()
       for x, y in train_dataloader:
           y_hat = model.forward(x)
           loss = loss(y_hat, y)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

       model.eval()
       with torch.no_grad():
           y_hat = model.forward(x_train)
           train_loss = loss(y_hat, y_train)
           y_hat = model.forward(x_test)
           test_loss = loss(y_hat, y_test)
       print(f"Epoch #{epoch}: train MSE {train_loss:.4f}, test MSE {test_loss:.4f}")


if __name__ == "__main__":
   main()
 