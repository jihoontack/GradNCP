import math
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

        elif self.data_type == 'manifold':
            self.width = P.data_size[1]
            self.height = P.data_size[2]
            mgrid = self.shape_to_shpher_coords(P.data_size[-2:])
            mgrid = rearrange(mgrid, 'h w c -> (h w) c')

        elif self.data_type == 'audio':
            self.length = P.data_size[-1]

            mgrid = self.shape_to_coords([self.length])
            mgrid = 50 * mgrid

        elif self.data_type == 'video':
            self.n_frames = P.data_size[1]
            self.width = P.data_size[2]
            self.height = P.data_size[3]
            self.ratio_wh = self.width / self.height

            if P.decoder == "nerv":
                mgrid = torch.linspace(0.0, 1.0, self.n_frames)
                P.dim_in = 1

            else:
                mgrid = self.shape_to_coords((self.n_frames, self.width, self.height))
                mgrid = rearrange(mgrid, 't h w c -> (t h w) c')

        else:
            raise NotImplementedError()

        self.register_buffer('grid', mgrid)

    def shape_to_coords(self, spatial_shape):
        coords = []
        for i in range(len(spatial_shape)):
            coords.append(torch.linspace(-1.0, 1.0, spatial_shape[i]))
        return torch.stack(torch.meshgrid(*coords), dim=-1)

    def shape_to_shpher_coords(self, spatial_shape):
        num_lats, num_lons = spatial_shape
        # Uniformly spaced latitudes and longitudes corresponding to ERA5 grids
        latitude = torch.linspace(90.0, -90.0, num_lats)
        longitude = torch.linspace(0.0, 360.0 - (360.0 / num_lons), num_lons)
        # Create a grid of latitude and longitude values (num_lats, num_lons)
        longitude_grid, latitude_grid = torch.meshgrid(longitude, latitude)
        longitude_grid, latitude_grid = longitude_grid.t(), latitude_grid.t()
        # Create coordinate tensor
        # Spherical coordinates have 3 dimensions
        coordinates = torch.zeros(latitude_grid.shape + (3,))
        long_rad = math.pi * longitude_grid / 180.0
        lat_rad = math.pi * latitude_grid / 180.0
        coordinates[..., 0] = torch.cos(lat_rad) * torch.cos(long_rad)
        coordinates[..., 1] = torch.cos(lat_rad) * torch.sin(long_rad)
        coordinates[..., 2] = torch.sin(lat_rad)
        return coordinates

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
        if self.data_type in ['img', 'manifold']:
            return self.forward_image(inputs, params)
        elif self.data_type == 'audio':
            return self.forward_audio(inputs, params)
        elif self.data_type == 'video':
            return self.forward_video(inputs, params)
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
            if self.P.decoder == "nerv":
                out, feature, x = self.decoder(coords, params=params, get_features=True)
            else:
                out, feature = self.decoder(coords, params=params, get_features=True)

            if self.data_type in ['img', 'manifold']:
                out = rearrange(out, 'b hw c -> b c hw')
                feature = rearrange(feature, 'b hw f -> b f hw')
                inputs = rearrange(inputs, 'b c h w -> b c (h w)')
            elif self.data_type == 'audio':
                out = rearrange(out, 'b l c -> b c l')
                feature = rearrange(feature, 'b l f -> b f l')
            elif self.P.data_type == "video":
                if self.P.decoder == "nerv":
                    out = rearrange(out, 'b (t h w) c  -> b (c h w) t',
                                    h=self.P.data_size[2], w=self.P.data_size[3])
                    feature = rearrange(feature, 'b (t h w) c  -> b (c h w) t',
                                        h=self.P.data_size[2], w=self.P.data_size[3])
                    x = rearrange(x, 'b (t h w) c  -> b (c h w) t',
                                  h=self.P.data_size[2], w=self.P.data_size[3])
                    inputs = rearrange(inputs, 'b t c h w -> b (c h w) t')
                else:
                    out = rearrange(out, 'b thw c -> b c thw')
                    feature = rearrange(feature, 'b thw f -> b f thw')
                    inputs = rearrange(inputs, 'b t c h w -> b c (t h w)')
            else:
                raise NotImplementedError()

            error = inputs - out # b c (hw)
            if self.P.decoder == "nerv":
                error = error * x  # b c (hw)
                error_norm = torch.norm(error, dim=1)  # b hw
                feature_norm = torch.norm(feature, dim=1)  # b hw
                gradient_norm = error_norm * feature_norm # approximate with upper bound
            else:
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

        if self.P.decoder == "nerv":
            width = self.P.data_size[2]
            height = self.P.data_size[3]
            self.gradncp_coord = torch.gather(
                coords, 1, self.gradncp_index
            )  # b int(t*ratio)
            self.gradncp_index = self.gradncp_index.unsqueeze(dim=1).repeat(1, self.P.dim_out * width * height, 1)
            # b c*h*w int(t * ratio)
        else:
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

    def forward_audio(self, inputs=None, params=None):
        if exists(inputs):
            inputs = inputs[0]
        coords, meta_batch_size = self.get_batch_coords(inputs, params)

        out = self.decoder(coords, params=params)
        out = rearrange(out, 'b l c -> b c l')

        if exists(inputs):
            if self.sampled_coord is None and self.gradncp_coord is None:
                return F.mse_loss(
                    inputs.view(meta_batch_size, -1), out.view(meta_batch_size, -1), reduce=False
                ).mean(dim=1)
            elif self.gradncp_coord is not None:
                inputs = torch.gather(inputs, 2, self.gradncp_index)
                return F.mse_loss(
                    inputs.view(meta_batch_size, -1), out.view(meta_batch_size, -1), reduce=False
                ).mean(dim=1)
            else:
                inputs = inputs[:, :, self.sampled_index]
                return F.mse_loss(
                    inputs.view(meta_batch_size, -1), out.view(meta_batch_size, -1), reduce=False
                ).mean(dim=1)

        return out

    def forward_video(self, inputs=None, params=None):
        if exists(inputs):
            inputs = inputs[0]

        coords, meta_batch_size = self.get_batch_coords(inputs, params)

        out = self.decoder(coords, params=params)
        out = rearrange(out, 'b thw c -> b c thw').contiguous()

        if exists(inputs):
            if self.sampled_coord is None and self.gradncp_coord is None:
                inputs = rearrange(inputs, 'b t c h w -> b c (t h w)').contiguous()
                return F.mse_loss(
                    inputs.view(meta_batch_size, -1), out.view(meta_batch_size, -1), reduce=False
                ).mean(dim=1)
            elif self.gradncp_coord is not None:
                if self.P.decoder == "nerv":
                    b,t,c,h,w = inputs.shape
                    inputs = rearrange(inputs, 'b t c h w -> b (c h w) t')
                    inputs = torch.gather(inputs, 2, self.gradncp_index)
                    out = rearrange(out, 'b c (t h w) -> b (c h w) t', c=c, h=h, w=w)
                else:
                    inputs = rearrange(inputs, 'b t c h w -> b c (t h w)')
                    inputs = torch.gather(inputs, 2, self.gradncp_index)

                return F.mse_loss(
                    inputs.view(meta_batch_size, -1), out.view(meta_batch_size, -1), reduce=False
                ).mean(dim=1)
            else:
                if self.P.decoder == "nerv":
                    raise NotImplementedError()
                inputs = rearrange(inputs, 'b t c h w -> b c (t h w)')[:, :, self.sampled_index]
                return F.mse_loss(
                    inputs.view(meta_batch_size, -1), out.view(meta_batch_size, -1), reduce=False
                ).mean(dim=1)

        out = rearrange(out, 'b c (t h w) -> b t c h w', t=self.n_frames, h=self.height, w=self.width)
        return out
