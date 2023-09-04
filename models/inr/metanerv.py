import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from collections import OrderedDict
from models.metamodule import MetaModule, MetaSequential, MetaBatchLinear
import math


small_config = {
    "lower_width": 32,
    "num_blocks": 1,
    "embed_base": 1.25,
    "embed_levels": 40,
    "stem_dim": 512,
    "stem_num": 1,
    "reduction": 2,
    "fc_hw_dim": "4_4_32",
    "expansion": 4,
    "strides": [4,2,2,2],
    "conv_type": "conv",
    "b": 1,
    "norm": None,
    "act": "gelu",
    "bias": True
}

large_config = {
    "lower_width": 32,
    "num_blocks": 1,
    "embed_base": 1.25,
    "embed_levels": 40,
    "stem_dim": 512,
    "stem_num": 1,
    "reduction": 2,
    "fc_hw_dim": "4_4_32",
    "expansion": 4,
    "strides": [4,2,2,2,2],
    "conv_type": "conv",
    "b": 1,
    "norm": None,
    "act": "gelu",
    "bias": True
}

class MetaUpSampler(MetaModule):
    def __init__(self, data_type, decoder, upscale_factor=None):
        super(MetaUpSampler, self).__init__()
        self.data_type = data_type
        self.decoder = decoder
        if upscale_factor != None:
            self.upscale_factor = upscale_factor

    def forward(self, x, params=None):

        if self.data_type == 'video' and self.decoder == "nerv":
            b, t, c, h, w = x.size()
            x = rearrange(x, "b t c h w -> (b t) c h w")
            x = F.pixel_shuffle(x, self.upscale_factor)
            # x = F.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear')
            x = rearrange(x, "(b t) c h w -> b t c h w", b=b, t=t)
            return x
        else:
            raise NotImplementedError()


class MetaGeLULayer(MetaModule):
    """
    Single layer of SIREN; uses SIREN-style init. scheme.
    """
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        # Encapsulates MetaLinear and activation.
        if bias == True:
            self.linear = MetaBatchLinear(dim_in, dim_out, bias=True)
        else:
            self.linear = MetaBatchLinear(dim_in, dim_out)

        self.activation = nn.GELU()

    def forward(self, x, params=None):
        return self.activation(self.linear(x, self.get_subdict(params, 'linear')))


class MetaBatchConv2DLayer(nn.Conv2d, MetaModule):

    __doc__ = nn.Conv2d.__doc__


    def forward(self, x, params):

        assert len(x.shape) == 5, "batchconv2d expects a 5d [{}] tensor".format(x.shape)

        if params is None:
            params = OrderedDict(self.named_parameters())
            for name, param in params.items():
                params[name] = param[None, ...].repeat((x.size(0),) + (1,) * len(param.shape))


        bias = params.get('bias', None)
        weight = params['weight']


        if bias is None:
            assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
        else:
            assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
                0], "dim=0 of bias must be equal in size to dim=0 of weight"

        b_i, b_j, c, h, w = x.shape
        b_i, out_channels, in_channels, kernel_height_size, kernel_width_size = weight.shape

        out = x.permute([1, 0, 2, 3, 4]).contiguous().view(b_j, b_i * c, h, w)
        weight = weight.view(b_i * out_channels, in_channels, kernel_height_size, kernel_width_size)

        out = F.conv2d(out, weight=weight, bias=None, stride=self.stride, dilation=self.dilation, groups=b_i,
                       padding=self.padding)


        out = out.view(b_j, b_i, out_channels, out.shape[-2], out.shape[-1])

        out = out.permute([1, 0, 2, 3, 4])

        if bias is not None:
            out = out + bias.unsqueeze(1).unsqueeze(3).unsqueeze(3)

        return out


class MetaNeRVBlock(MetaModule):
    def __init__(self, is_last = False, **kargs):
        super().__init__()
        ngf, new_ngf, stride = kargs['ngf'], kargs['new_ngf'], kargs['stride']
        self.is_last = is_last
        if self.is_last :
            self.conv = MetaBatchConv2DLayer(in_channels=ngf, out_channels=new_ngf, kernel_size=1, stride=1)
        else:
            self.conv = MetaBatchConv2DLayer(in_channels=ngf, out_channels=new_ngf * stride * stride, kernel_size=3, stride=1, padding=1)
            self.up_scale = MetaUpSampler(data_type="video", decoder="nerv", upscale_factor=stride)
            
            self.norm = nn.Identity()
            self.act = nn.GELU()

    def forward(self, x, params=None):
        x = self.conv(x, self.get_subdict(params, 'conv'))
        if self.is_last == True:
            return x
        out = self.up_scale(x)
        return self.act(self.norm(out))


class MetaNeRV(MetaModule):    
    """
    NERV as a meta-network.
    """
    def __init__(self, config="small", 
                 data_type='img', data_size=(3, 178, 178)):
        super().__init__()
        if config == "small":
            config = small_config
            self.config = small_config
        else:
            config = large_config
            self.config = large_config
        
        self.lbase = config["embed_base"]
        self.levels = config["embed_levels"]
        
        stem_dim = config["stem_dim"]
        stem_num = config["stem_num"]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in config['fc_hw_dim'].split('_')]
        mlp_dim_list = [self.levels*2] + [stem_dim] * stem_num + [self.fc_h *self.fc_w *self.fc_dim]
        
        stems = []
        for ind in range(len(mlp_dim_list)-1):
            stems.append(MetaGeLULayer(dim_in=mlp_dim_list[ind], dim_out=mlp_dim_list[ind+1], ))

        self.stems = MetaSequential(*stems)
        
        # BUILD CONV LAYERS
        layers = []

        ngf = self.fc_dim
        for i, stride in enumerate(config['strides']):
            if i == 0:
                # expand channel width at first stage
                new_ngf = int(ngf * config['expansion'])
            else:
                # change the channel width for each stage
                new_ngf = max(ngf // (1 if stride == 1 else config['reduction']), config['lower_width'])

            for j in range(config['num_blocks']):
                layers.append(MetaNeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=1 if j else stride))
                ngf = new_ngf
    
        head_layers = MetaNeRVBlock(ngf=ngf, new_ngf=3, stride=1, is_last=True)
        layers.append(head_layers)

        self.layers = MetaSequential(*layers)
        
    def posenc(self, x):
        pe_list = []
        for i in range(self.levels):
            temp_value = x * self.lbase **(i) * math.pi
            pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
        return torch.stack(pe_list, 1)

    def forward(self, input, params=None):

        b, t = input.shape
        x = self.posenc(input.view(-1)) # [b, t, d]
        x = rearrange(x, "(b t) d -> b t d", b=b, t=t)

        x = self.stems(x, params=self.get_subdict(params, 'stems')) # [b, t, self.fc_h *self.fc_w *self.fc_dim]
        x = x.view(b, t, self.fc_dim, self.fc_h, self.fc_w)

        x = self.layers(x, params=self.get_subdict(params, 'layers'))

        # [0, 1] range, b, t, c, h, w
        output = (torch.tanh(x) + 1) * 0.5

        output = rearrange(output, 'b t c h w -> b (t h w) c')
        return  output


class MetaNeRVPenultimate(MetaModule):
    """
    NERV as a meta-network.
    """

    def __init__(self, config="small",
                 data_type='img', data_size=(3, 178, 178)):
        super().__init__()
        if config == "small":
            config = small_config
            self.config = small_config
        else:
            config = large_config
            self.config = large_config

        self.lbase = config["embed_base"]
        self.levels = config["embed_levels"]

        stem_dim = config["stem_dim"]
        stem_num = config["stem_num"]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in config['fc_hw_dim'].split('_')]
        mlp_dim_list = [self.levels * 2] + [stem_dim] * stem_num + [self.fc_h * self.fc_w * self.fc_dim]

        stems = []
        for ind in range(len(mlp_dim_list) - 1):
            stems.append(MetaGeLULayer(dim_in=mlp_dim_list[ind], dim_out=mlp_dim_list[ind + 1], ))

        self.stems = MetaSequential(*stems)

        # BUILD CONV LAYERS
        layers = []

        ngf = self.fc_dim
        for i, stride in enumerate(config['strides']):
            if i == 0:
                # expand channel width at first stage
                new_ngf = int(ngf * config['expansion'])
            else:
                # change the channel width for each stage
                new_ngf = max(ngf // (1 if stride == 1 else config['reduction']), config['lower_width'])

            for j in range(config['num_blocks']):
                layers.append(MetaNeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=1 if j else stride))
                ngf = new_ngf

        # head_layers = MetaNeRVBlock(ngf=ngf, new_ngf=3, stride=1, is_last=True)
        # layers.append(head_layers)

        self.layers = MetaSequential(*layers)
        self.last_layers = MetaNeRVBlock(ngf=ngf, new_ngf=3, stride=1, is_last=True)

    def posenc(self, x):
        pe_list = []
        for i in range(self.levels):
            temp_value = x * self.lbase ** (i) * math.pi
            pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
        return torch.stack(pe_list, 1)

    def forward(self, input, params=None, get_features=False):

        b, t = input.shape
        x = self.posenc(input.view(-1))  # [b, t, d]
        x = rearrange(x, "(b t) d -> b t d", b=b, t=t)

        x = self.stems(x, params=self.get_subdict(params, 'stems'))  # [b, t, self.fc_h *self.fc_w *self.fc_dim]
        x = x.view(b, t, self.fc_dim, self.fc_h, self.fc_w)

        feature = self.layers(x, params=self.get_subdict(params, 'layers'))
        x = self.last_layers(feature, params=self.get_subdict(params, 'last_layers'))

        # [0, 1] range, b, t, c, h, w
        output = (torch.tanh(x) + 1) * 0.5
        output = rearrange(output, 'b t c h w -> b (t h w) c')

        if get_features:
            feature = rearrange(feature, 'b t c h w -> b (t h w) c')
            x = 1 - torch.tanh(x) ** 2
            x = rearrange(x, 'b t c h w -> b (t h w) c')
            return output, feature, x

        return output
