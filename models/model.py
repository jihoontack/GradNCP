import torch

from models.inr.metasiren import MetaSiren, MetaSirenPenultimate
from models.wrapper import MetaWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_inr(P):
    if P.decoder == 'siren':
        if P.sample_type in ['gradncp']:
            model = MetaSirenPenultimate(P.dim_in, P.dim_hidden, P.dim_out, P.num_layers,
                                         w0=P.w0, w0_initial=P.w0, data_size=P.data_size, data_type=P.data_type)
        else:
            model = MetaSiren(P.dim_in, P.dim_hidden, P.dim_out, P.num_layers,
                              w0=P.w0, w0_initial=P.w0, data_size=P.data_size, data_type=P.data_type)
    else:
        raise ValueError("no such model exists, mate.")

    return model


def get_model(P):
    decoder = get_inr(P)

    if P.data_type in ['img']:
        return MetaWrapper(P, decoder)
    else:
        raise NotImplementedError()
