import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from src.noisy import NoisyLinear


class RainbowDQN(nn.Module):
    """Q model implementation of Rainbow agent.

    Attributes
    ----------
    in_dim : int
        Number of elements expected in the input vector.
    out_dim : int
        Number of elements at the output layer corresponding 
        to the total number of actions as defined by
        the environment.
    atom_size : int 
        Number of atoms to use for the categorical DQN algorithm.
    support : torch.Tensor
        The support vector used in the distribution projection.
    """

    def __init__(self, in_dim, out_dim, atom_size, support):
        super(RainbowDQN, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = nn.Sequential(OrderedDict([
            ('linear1', init(nn.Linear(in_dim, 256))),
            ('relu1', nn.ReLU()),
            ('linear2', init(nn.Linear(256, 256))),
            ('relu2', nn.ReLU()),
            ('linear3', init(nn.Linear(256, 128))),
            ('relu3', nn.ReLU())
        ]))

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(128, 128, 0.5)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size, 0.5)

        # set value layer
        self.value_hidden_layer = NoisyLinear(128, 128, 0.5)
        self.value_layer = NoisyLinear(128, atom_size, 0.5)

    def forward(self, x):
        """Forward propagation of input.

        Attributes
        ----------
        x : torch.Tensor
            Input tensor to be propagated to the model.
        """
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q

    def dist(self, x):
        """Get distribution for atoms.

        Attributes
        ----------
        x : torch.Tensor
            Input tensor to be propagated to the model.
        """
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size)
        value = self.value_layer(val_hid).view(
            -1, 1, self.atom_size)

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)

        return dist

    def reset_noise(self):
        """Reset all noisy layers.
        """

        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()


def init_(module, weight_init, bias_init, gain=1):
    """Initializes a linear layer.

    Parameters
    ----------
    module : torch.nn.Module
        Module found on the model defined.
    weight_init : torch.Tensor
        Manner of given module's weights initialization.
    bias_init : torch.Tensor
        Manner of given module's biases initialization.
    gain : int, optional
        The recommended gain value for the given
        nonlinearity function, by default 1

    Returns
    -------
    torch.nn.Module
        Initialized module.
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def init(module):
    """Initializes model.

    Parameters
    ----------
    module : torch.nn.Module
        The defined model.

    Returns
    -------
    torch.nn.Module
        Initialized model.        
    """
    return init_(module,
                 nn.init.orthogonal_,
                 lambda x: nn.init.constant_(x, 0),
                 nn.init.calculate_gain('relu'))
