import torch.nn as nn


class Summarizer(nn.Module):
    def __init__(self, input_dim, hidden_size=32, num_layers=1):
        super(Summarizer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.hidden = None

    def forward(self, observation, hidden=None):
        if len(observation.shape) == 2:
            observation = observation.unsqueeze(1)
        lstm_out, hidden = self.lstm(observation, hidden)
        return lstm_out, hidden
    
    def reset_hidden(self):
        self.hidden = None