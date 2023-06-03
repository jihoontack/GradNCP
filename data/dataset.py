import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import Dataset

DATA_PATH = '/data'


class ImgDataset(Dataset):
    def __init__(self, data, sdf=False):
        self.data = data
        self.sdf = sdf

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if not self.sdf:
            x = x[0]
        return {
            'imgs': x,
        }


def get_dataset(P, dataset, only_test=False):
    """
    Load dataloaders for an image dataset, center-cropped to a resolution.
    """
    val_set = None
    P.data_size = None

    if dataset == 'celeba':
        T_base = T.Compose([
            T.Resize(178),
            T.CenterCrop(178),
            T.ToTensor()
        ])
        train_set = ImgDataset(
            datasets.CelebA(DATA_PATH, split='train',
                            target_type='attr', transform=T_base)
        )
        test_set = ImgDataset(
            datasets.CelebA(DATA_PATH, split='test',
                            target_type='attr', transform=T_base)
        )
        P.data_type = 'img'
        P.dim_in, P.dim_out = 2, 3
        P.data_size = (3, 178, 178)

    else:
        raise NotImplementedError()

    P.train_set = train_set

    if only_test:
        return test_set

    val_set = test_set if val_set is None else val_set
    return train_set, val_set
