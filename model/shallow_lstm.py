import torch 

from torch import nn


class ShallowLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, scaler=None):
        super(ShallowLSTM, self).__init__()
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, input_size)
        self.scaler = scaler

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        x = self.linear(self.relu(x[:, -1:, :]))
        return x, hidden

    def predict(self, x):
        
        # Transform input
        if self.scaler is not None:
            x = torch.tensor(self.scaler.scale_, device=x.device, dtype=x.dtype) * x + torch.tensor(self.scaler.min_, device=x.device, dtype=x.dtype)

        input_sequence_len = x.shape[1]
        output = torch.zeros(x.shape[0], self.sequence_length, x.shape[2], device=x.device)

        with torch.no_grad():
            hidden = None
        
            for i in range(self.sequence_length):
                # Predict next time step
                z, hidden = self.forward(x[:, -input_sequence_len:, :], hidden)

                # Concatenate prediction to inputs
                x = torch.cat((x, z), dim=1)

                # Store transformed value into output tensor
                if self.scaler is not None:
                    z = (z - torch.tensor(self.scaler.min_, device=z.device, dtype=z.dtype))/torch.tensor(self.scaler.scale_, device=z.device, dtype=z.dtype)
                output[:, i, :] = z.squeeze()

        # Return only values for ws_x and ws_y
        return output[:, :, :2]
