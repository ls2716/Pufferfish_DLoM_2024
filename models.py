"""Implement an fully connected neural network model for the MNIST dataset.

Similarily, implement a low rank version of the network using the low rank layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import SVD function


class FullRankNet(nn.Module):
    """Full rank network for MNIST dataset.
    """

    def __init__(self, no_hidden_layers=5, hidden_size=512):
        super(FullRankNet, self).__init__()
        self.input_layer = nn.Linear(28*28, hidden_size)
        self.hidden_layers = nn.ModuleList()
        for i in range(no_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, 10)
        self.no_hidden_layers = no_hidden_layers

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 28*28)
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        # Return the logits
        return x


class LowRankLayer(nn.Module):
    """Low rank layer for the neural network.
    """

    def __init__(self, input_size, output_size, low_rank):
        super(LowRankLayer, self).__init__()
        self.u_layer = nn.Linear(input_size, low_rank, bias=False)
        self.v_layer = nn.Linear(low_rank, output_size, bias=True)

    def forward(self, x):
        u = self.u_layer(x)
        v = self.v_layer(u)
        return v


class LowRankNet(nn.Module):
    """Low rank network for MNIST dataset.
    """

    def __init__(self, full_rank_net, low_rank=10, first_layer_to_low=0):
        super(LowRankNet, self).__init__()
        # Copy the input layer from the full rank network
        # Get the hidden dimension of the full rank network
        hidden_size = full_rank_net.input_layer.weight.size(0)
        # Create the input layer
        self.input_layer = nn.Linear(28*28, hidden_size)
        # Create the hidden layers
        # Get the number of hidden layers in the full rank network
        no_hidden_layers = len(full_rank_net.hidden_layers)
        self.hidden_layers = nn.ModuleList()
        for i in range(first_layer_to_low):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        for i in range(first_layer_to_low, no_hidden_layers):
            # Create the low rank layer
            self.hidden_layers.append(LowRankLayer(
                hidden_size, hidden_size, low_rank))
        self.output_layer = nn.Linear(hidden_size, 10)
        self.no_hidden_layers = no_hidden_layers
        self.first_layer_to_low = first_layer_to_low
        self.low_rank = low_rank

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x


def convert_to_low_rank(full_rank_net, low_rank_net):
    """Convert a full rank network to a low rank network.
    """
    low_rank = low_rank_net.low_rank
    first_layer_to_low = low_rank_net.first_layer_to_low
    no_hidden_layers = low_rank_net.no_hidden_layers

    with torch.no_grad():
        # Copy the input layer weights
        low_rank_net.input_layer.weight.data[:] = \
            full_rank_net.input_layer.weight.data[:]
        # Copy the biases of the input layer
        low_rank_net.input_layer.bias.data[:] = \
            full_rank_net.input_layer.bias.data[:]

        # Copy the output layer weights
        low_rank_net.output_layer.weight.data[:] = \
            full_rank_net.output_layer.weight.data[:]
        # Copy the biases of the output layer
        low_rank_net.output_layer.bias.data[:] = \
            full_rank_net.output_layer.bias.data[:]

        for i in range(first_layer_to_low):
            # Copy the weights of the unchanged layers
            low_rank_net.hidden_layers[i].weight.data[:] = \
                full_rank_net.hidden_layers[i].weight.data[:]
            # Copy the biases of the unchanged layers
            low_rank_net.hidden_layers[i].bias.data[:] = \
                full_rank_net.hidden_layers[i].bias.data[:]

        for i in range(first_layer_to_low, no_hidden_layers):
            # Get the matrix
            A = full_rank_net.hidden_layers[i].weight.data
            # Perform SVD
            U, S, Vh = torch.linalg.svd(A)
            # Compute new U
            U_low_rank = U[:, :low_rank] @ torch.sqrt(torch.diag(S[:low_rank]))
            # Compute new Vh
            Vh_low_rank = torch.sqrt(torch.diag(
                S[:low_rank])) @ Vh[:low_rank, :]
            # Copy the weights of the low rank layer
            print(low_rank_net.hidden_layers[i].u_layer.weight.data.shape)
            low_rank_net.hidden_layers[i].u_layer.weight.data[:] = U_low_rank.T[:]
            low_rank_net.hidden_layers[i].v_layer.weight.data[:] = Vh_low_rank.T[:]
            # Copy the biases of the low rank layer
            low_rank_net.hidden_layers[i].v_layer.bias.data[:] = \
                full_rank_net.hidden_layers[i].bias.data[:]


if __name__ == "__main__":
    full_rank_net = FullRankNet()
    low_rank_net = LowRankNet(
        full_rank_net, low_rank=258, first_layer_to_low=2)
    convert_to_low_rank(full_rank_net, low_rank_net)

    # Test the networks
    x = torch.randn(5, 28, 28)
    logits_full_rank = full_rank_net(x)
    logits_low_rank = low_rank_net(x)
    print(logits_full_rank[0])
    print(logits_low_rank[0])
