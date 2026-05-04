import torch

from torch import nn


class EncoderDecoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            batch_size: int,
            sequence_length: int,
            num_layers: int = 1,
            scaler=None
    ):
        super(EncoderDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.encoder_lstm = nn.LSTM(self.input_size, hidden_size, num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(self.output_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.scaler = scaler

    def forward(self, x):
        # batch_size = min(self.batch_size, x.size(0))
        batch_size = x.size(0)
        _, hidden = self.encoder_lstm(x)
        input_t = torch.zeros(batch_size, 1, self.output_size, dtype=torch.float32)
        output_tensor = torch.zeros(batch_size, self.sequence_length, self.output_size)
        for t in range(self.sequence_length):
            output_t, hidden = self.decoder_lstm(input_t, hidden)
            output_t = self.linear(output_t)
            input_t = output_t
            output_tensor[:, t] = output_t.squeeze()
        return output_tensor

    def predict(self, x):
        if self.scaler is not None:
            x = self.scaler.transform(x)
        x = self.forward(x)
        if self.scaler is not None:
            x = self.scaler.inverse_transform(x)
        return x
