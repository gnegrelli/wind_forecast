import torch 

from torch import nn


class ShallowLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, scaler=None):
        super(ShallowLSTM, self).__init__()
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.relu = nn.ReLU
        self.linear = nn.Linear(hidden_size, input_size)
        self.scaler = scaler

    def forward(self, x):
        x = self.relu(self.lstm(x))
        x = self.linear(x)
        return x

    def predict(self, x):
        
        # Transform input
        if self.scaler is not None:
            x = self.scaler.transform(x)

        input_sequence_len = x.size[1]
        output = torch.zeros(x.size[0], self.sequence_length, x.size[2])
        
        for i in range(self.sequence_length):
            # Predict next time step
            z = self.forward(x[:, -input_sequence_len:, :])

            # Concatenate prediction to inputs
            x = torch.cat((x, z), dim=1)

            # Store transformed value into output tensor
            output[:, i, :] = self.scaler.inverse_transform(z)

        # Return only values for ws_x and ws_y
        return output[:, :, :2]
