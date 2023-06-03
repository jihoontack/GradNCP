from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def exists(val):
    return val is not None


class MetaWrapper(nn.Module):
    def __init__(self, P, decoder):
        super().__init__()
        self.P = P
        self.decoder = decoder
        self.data_type = P.data_type
        self.sampled_coord = None
        self.sampled_index = None
        self.gradncp_coord = None
        self.gradncp_index = None

        if self.data_type == 'img':
            self.width = P.data_size[1]
            self.height = P.data_size[2]

            mgrid = self.shape_to_coords((self.width, self.height))
            mgrid = rearrange(mgrid, 'h w c -> (h w) c')

        else:
            raise NotImplementedError()

        self.register_buffer('grid', mgrid)

    def shape_to_coords(self, spatial_shape):
        coords = []
        for i in range(len(spatial_shape)):
            coords.append(torch.linspace(-1.0, 1.0, spatial_shape[i]))
        return torch.stack(torch.meshgrid(*coords), dim=-1)

    def get_batch_params(self, params, batch_size):
        if params is None:
            params = OrderedDict()
            for name, param in self.decoder.meta_named_parameters():
                params[name] = param[None, ...].repeat((batch_size,) + (1,) * len(param.shape))
        return params

    def coord_init(self):
        self.sampled_coord = None
        self.sampled_index = None
        self.gradncp_coord = None
        self.gradncp_index = None

    def get_batch_coords(self, inputs=None, params=None):
        if inputs is None and params is None:
            meta_batch_size = 1
        elif inputs is None:
            meta_batch_size = list(params.values())[0].size(0)
        else:
            meta_batch_size = inputs.size(0)

        # batch of coordinates
        if self.sampled_coord is None and self.gradncp_coord is None:
            coords = self.grid
        elif self.gradncp_coord is not None:
            return self.gradncp_coord, meta_batch_size
        else:
            coords = self.sampled_coord
        coords = coords.clone().detach()[None, ...].repeat((meta_batch_size,) + (1,) * len(coords.shape))
        return coords, meta_batch_size

    def forward(self, inputs, params=None):
        if self.data_type in ['img']:
            return self.forward_image(inputs, params)
        else:
            raise NotImplementedError()

    def sample(self, sample_type, task_data, params):
        if sample_type == 'random':
            self.random_sample()
        elif sample_type == 'gradncp':
            self.gradncp(task_data, params)
        else:
            raise NotImplementedError()

    def gradncp(self, inputs, params):
        ratio = self.P.data_ratio
        inputs = inputs[0]
        meta_batch_size = inputs.size(0)
        coords = self.grid
        coords = coords.clone().detach()[None, ...].repeat((meta_batch_size,) + (1,) * len(coords.shape))

        with torch.no_grad():
            out, feature = self.decoder(coords, params=params, get_features=True)
            if self.data_type in ['img']:
                out = rearrange(out, 'b hw c -> b c hw')
                feature = rearrange(feature, 'b hw f -> b f hw')
                inputs = rearrange(inputs, 'b c h w -> b c (h w)')
            else:
                raise NotImplementedError()

            error = inputs - out # b c (hw)
            gradient = -1 * feature.unsqueeze(dim=1) * error.unsqueeze(dim=2) # b c f hw
            gradient_bias = -1 * error.unsqueeze(dim=2) # b c hw
            gradient = torch.cat([gradient, gradient_bias], dim=2)
            gradient = rearrange(gradient, 'b c f hw -> b (c f) hw')
            gradient_norm = torch.norm(gradient, dim=1)  # b hw
            coords_len = gradient_norm.size(1)

        # coords b hw dim_in
        self.gradncp_index = torch.sort(
            gradient_norm, dim=1, descending=True
        )[1][:, :int(coords_len * ratio)]  # b int(hw * ratio)

        self.gradncp_coord = torch.gather(
            coords, 1, self.gradncp_index.unsqueeze(dim=2).repeat(1, 1, self.P.dim_in)
        )
        self.gradncp_index = self.gradncp_index.unsqueeze(dim=1).repeat(1, self.P.dim_out, 1)

    def random_sample(self):
        coord_size = self.grid.size(0)  # shape (h * w, c)
        perm = torch.randperm(coord_size)
        self.sampled_index = perm[:int(self.P.data_ratio * coord_size)]
        self.sampled_coord = self.grid[self.sampled_index]
        return self.sampled_coord

    def forward_image(self, inputs=None, params=None):
        if exists(inputs):
            inputs = inputs[0]

        coords, meta_batch_size = self.get_batch_coords(inputs, params)

        out = self.decoder(coords, params=params)
        out = rearrange(out, 'b hw c -> b c hw')

        if exists(inputs):
            if self.sampled_coord is None and self.gradncp_coord is None:
                return F.mse_loss(
                    inputs.view(meta_batch_size, -1), out.view(meta_batch_size, -1), reduce=False
                ).mean(dim=1)
            elif self.gradncp_coord is not None:
                inputs = rearrange(inputs, 'b c h w -> b c (h w)')
                inputs = torch.gather(inputs, 2, self.gradncp_index)
                return F.mse_loss(
                    inputs.view(meta_batch_size, -1), out.view(meta_batch_size, -1), reduce=False
                ).mean(dim=1)
            else:
                inputs = rearrange(inputs, 'b c h w -> b c (h w)')[:, :, self.sampled_index]
                return F.mse_loss(
                    inputs.view(meta_batch_size, -1), out.view(meta_batch_size, -1), reduce=False
                ).mean(dim=1)

        out = rearrange(out, 'b c (h w) -> b c h w', h=self.height, w=self.width)
        return out
