import math

import torch
import torch.nn as nn

from models.metamodule import MetaModule, MetaSequential, MetaBatchLinear


class Sine(nn.Module):
    def __init__(self, w0=30.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0*x)


class MetaSirenLayer(MetaModule):
    """
    Single layer of SIREN; uses SIREN-style init. scheme.
    """
    def __init__(self, dim_in, dim_out, w0=30., c=6., is_first=False, is_final=False):
        super().__init__()
        # Encapsulates MetaLinear and activation.
        self.linear = MetaBatchLinear(dim_in, dim_out)
        self.activation = nn.Identity() if is_final else Sine(w0)
        # Initializes according to SIREN init.
        self.init_(c=c, w0=w0, is_first=is_first)

    def init_(self, c, w0, is_first):
        dim_in = self.linear.weight.size(1)
        w_std = 1/dim_in if is_first else (math.sqrt(c/dim_in)/w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def forward(self, x, params=None):
        return self.activation(self.linear(x, self.get_subdict(params, 'linear')))


class MetaReLULayer(MetaModule):
    """
    Single layer of SIREN; uses SIREN-style init. scheme.
    """
    def __init__(self, dim_in, dim_out, w0=30., c=6., is_first=False, is_final=False):
        super().__init__()
        # Encapsulates MetaLinear and activation.
        self.linear = MetaBatchLinear(dim_in, dim_out)
        self.activation = nn.Identity() if is_final else nn.ReLU()
        # Initializes according to SIREN init.
        self.init_(c=c, w0=w0, is_first=is_first)

    def init_(self, c, w0, is_first):
        dim_in = self.linear.weight.size(1)
        w_std = 1/dim_in if is_first else (math.sqrt(c/dim_in)/w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def forward(self, x, params=None):
        return self.activation(self.linear(x, self.get_subdict(params, 'linear')))


class MetaSiren(MetaModule):
    """
    SIREN as a meta-network.
    """
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers=4, w0=30., w0_initial=30.,
                 data_type='img', data_size=(3, 178, 178)):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.w0 = w0
        layers = []
        for ind in range(num_layers-1):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden
            layers.append(MetaSirenLayer(dim_in=layer_dim_in, dim_out=dim_hidden,
                                         w0=layer_w0, is_first=is_first))
        layers.append(MetaSirenLayer(dim_in=dim_hidden, dim_out=dim_out,
                                     w0=w0, is_final=True))
        self.layers = MetaSequential(*layers)

    def forward(self, x, params=None):
        return self.layers(x, params=self.get_subdict(params, 'layers')) + 0.5


class MetaSirenPenultimate(MetaModule):
    """
    SIREN as a meta-network.
    """
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers=4, w0=30., w0_initial=30.,
                 data_type='img', data_size=(3, 178, 178)):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.w0 = w0
        layers = []
        for ind in range(num_layers-1):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden
            layers.append(MetaSirenLayer(dim_in=layer_dim_in, dim_out=dim_hidden,
                                         w0=layer_w0, is_first=is_first))
        self.layers = MetaSequential(*layers)
        self.last_layer = MetaSirenLayer(dim_in=dim_hidden, dim_out=dim_out,
                                         w0=w0, is_final=True)

    def forward(self, x, params=None, get_features=False):
        feature = self.layers(x, params=self.get_subdict(params, 'layers'))
        out = self.last_layer(feature, params=self.get_subdict(params, 'last_layer')) + 0.5

        if get_features:
            return out, feature
        else:
            return out
