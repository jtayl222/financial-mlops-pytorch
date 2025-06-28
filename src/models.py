import torch
import torch.nn as nn

class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=1, dropout_prob=0.2):
        super(StockPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # batch_first=True means input/output tensors are (batch_size, sequence_length, features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid() # For binary classification output between 0 and 1

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Detach hidden states for each batch to prevent backpropagation through entire history
        # out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out, _ = self.lstm(x, (h0, c0))

        # Take output from the last time step for classification
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out) # Apply sigmoid for binary classification probability
        return out