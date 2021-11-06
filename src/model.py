
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from collections import OrderedDict

from src.noisy import NoisyLinear


class RainbowDQN(nn.Module):
    def __init__(self, out_dim, atom_size, support, features_layer_path, stats_layer_path):
        """Initialization."""
        super(RainbowDQN, self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set feature layer
        self.feature_layer = self.get_feature_layer()
        
        # set stats extractor layer
        self.stats_layer = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(6, 256)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(256, 256)),
            ('relu2', nn.ReLU()),
            ('linear3', nn.Linear(256, 256)),
            ('relu3', nn.ReLU()),
            ('out', nn.Linear(256, 6))
        ]))

        # set merge and predict layer
        self.driver_layer = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(12, 128)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(128, 128)),
            ('relu2', nn.ReLU()),
            ('linear3', nn.Linear(128, 128)),
            ('relu3', nn.ReLU())
        ]))

        # load pretrained layers
        chkpt = torch.load(features_layer_path)
        features_state_dict = chkpt.get('model_state_dict')
        self.feature_layer.load_state_dict(features_state_dict)

        chkpt = torch.load(stats_layer_path)
        stats_state_dict = chkpt.get('model_state_dict')
        self.stats_layer.load_state_dict(stats_state_dict)

        # freeze pretrained layers
        self.feature_layer.requires_grad = False
        self.stats_layer.requires_grad = False

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(128, 128, 0.5)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size, 0.5)

        # set value layer
        self.value_hidden_layer = NoisyLinear(128, 128, 0.5)
        self.value_layer = NoisyLinear(128, atom_size, 0.5)

    def forward(self, features, stats):
        """Forward method implementation."""
        dist = self.dist(features, stats)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, features, stats):
        """Get distribution for atoms."""
        features = self.feature_layer(features)
        stats = self.stats_layer(stats)
        merged = self.driver_layer(torch.cat((features, stats), 1))
        
        adv_hid = F.relu(self.advantage_hidden_layer(merged))
        val_hid = F.relu(self.value_hidden_layer(merged))
        
        advantage = self.advantage_layer(adv_hid).view(-1, self.out_dim, self.atom_size)
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()
    
    def get_feature_layer(self, hidden_neurons:int = 2048, n_classes:int = 6):
        model = models.resnet152(progress=True, pretrained=False)
        model.fc = torch.nn.Linear(hidden_neurons, n_classes)
        return model
