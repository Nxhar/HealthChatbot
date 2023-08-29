import torch.nn as nn

class FwdNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=input_size,out_features=hidden_size)
        self.layer_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.layer_3 = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.to(self.layer_1.weight.dtype)

        output = self.layer_1(x)
        output = self.relu(output)

        output = self.layer_2(output)
        output = self.relu(output)

        output = self.layer_3(output)

        return output
    
    # no activation or softmax
