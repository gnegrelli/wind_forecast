import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch import nn, tensor, no_grad
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


class ShallowLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ShallowLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

def build_data(data: pd.DataFrame, lookback=1, lookahead=1):
    feature_columns = ['ws_x', 'ws_y', 'ti', 'rho']
    target_columns = ['ws_x', 'ws_y']
    features = []
    targets = []
    for i in trange(lookback, data.shape[0] - lookahead):
        features.append(data.iloc[i - lookback:i][feature_columns].T.values.flatten().tolist())
        targets.append(data.iloc[i:i + lookahead][target_columns].T.values.flatten().tolist())

    return tensor(features, requires_grad=True), tensor(targets)

def main():
   # Load data
   print("Loading data...")
   with  open('data/Horns_rev1.pkl', 'rb') as file:
     data = pickle.load(file)
     data = data[int(-1 * 365 * 24 * 6):]

   # Preprocess
   print("Preprocessing data...")
   data['ws_x'] = data.apply(lambda x: x.ws * np.cos(np.pi * x.wd / 180), axis=1)
   data['ws_y'] = data.apply(lambda x: x.ws * np.sin(np.pi * x.wd / 180), axis=1)

   # Split into train-validation-test
   print("Splitting data...")
   train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

   # Scale features
   print("Scaling data...")
   scaler = MinMaxScaler(feature_range=(0, 1))
   scaler.fit(train_data)

   train_dataset = pd.DataFrame(scaler.transform(train_data), columns=train_data.columns)
   test_dataset = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)

   print("Building datasets...")
   x_train, y_train = build_data(train_dataset, lookback=6 * 24 * 5, lookahead=6 * 24 * 2)
   x_test, y_test = build_data(test_dataset, lookback=6 * 24 * 5, lookahead=6 * 24 * 2)

   train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=2048, shuffle=True)

   # Define model, optimizer and loss function
   print("Defining model...")
   model = ShallowLSTM(6 * 24 * 5 * 4, 6 * 24 * 5, 6 * 24 * 2 * 2)
   optimizer = Adam(model.parameters())
   loss_function = nn.MSELoss()
   previous_loss = np.inf
   
   # Run training
   n_epochs = 100
   print(f"Training model for {n_epochs} epochs...")
   for epoch in trange(n_epochs):
       print(f" - Starting epoch #{epoch}")
       model.train()
       for x, y in train_dataloader:
           y_hat = model.forward(x)
           loss = loss_function(y_hat, y)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           print(f" - Epoch #{epoch}: batch MSE {loss:.4f}")

       model.eval()
       with no_grad():
           y_hat = model.forward(x_train)
           train_loss = loss_function(y_hat, y_train)
           y_hat = model.forward(x_test)
           test_loss = loss_function(y_hat, y_test)
       print(f"Epoch #{epoch}: train MSE {train_loss:.4f}, test MSE {test_loss:.4f}")
       if test_loss < previous_loss:
           print("Saving model...")
           previous_loss = test_loss
           with open(f'models/ShallowLSTM_epoch{epoch}_loss{test_loss:.4f}.pkl', 'wb') as file:
               pickle.dump(model.state_dict(), file)


if __name__ == "__main__":
   main()
 