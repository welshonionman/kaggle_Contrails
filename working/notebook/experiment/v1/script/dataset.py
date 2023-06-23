import os
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A


class fastnumpyio:
    def load(file):
        file = open(file, "rb")
        header = file.read(128)
        descr = str(header[19:25], 'utf-8').replace("'", "").replace(" ", "")
        shape = tuple(int(num) for num in str(header[60:120], 'utf-8').replace(', }', '').replace('(', '').replace(')', '').split(','))
        datasize = np.lib.format.descr_to_dtype(descr).itemsize
        for dimension in shape:
            datasize *= dimension
        return np.ndarray(shape, dtype=descr, buffer=file.read(datasize))


def get_band_images(idx: str, parrent_folder: str, band: str, CFG) -> np.array:
    return fastnumpyio.load(os.path.join(CFG.dataset_path, parrent_folder, idx, f'band_{band}.npy'))


def normalize_range(data, bounds):
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def get_ash_color_images(idx: str, parrent_folder: str, CFG, get_mask_frame_only=False) -> np.array:
    _T11_BOUNDS = (243, 303)
    _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
    _TDIFF_BOUNDS = (-4, 2)

    band11 = get_band_images(idx, parrent_folder, '11', CFG)
    band14 = get_band_images(idx, parrent_folder, '14', CFG)
    band15 = get_band_images(idx, parrent_folder, '15', CFG)

    if get_mask_frame_only:
        band11 = band11[:, :, 4]
        band14 = band14[:, :, 4]
        band15 = band15[:, :, 4]

    r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
    g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(band14, _T11_BOUNDS)
    false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
    return false_color


def get_mask_image(idx: str, parrent_folder: str, CFG) -> np.array:
    return np.load(os.path.join(CFG.dataset_path, parrent_folder, idx, 'human_pixel_masks.npy'))


class ContrailsDataset(Dataset):
    def __init__(self, df, transform, mode='train'):
        self.df = df
        self.transform = A.Compose(transform)
        self.mode = mode

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = row.path
        record_id = row.record_id
        npy = fastnumpyio.load(str(path))

        if self.mode == 'train':
            image = npy[..., :-1]
            label = npy[..., -1]
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']
            label = np.expand_dims(label, 0)
            image = torch.tensor(image)
            label = torch.tensor(label)
            return image.float(), label.float()

        if self.mode == 'test':
            image = npy
            data = self.transform(image=image)
            image = data['image']
            image = torch.tensor(image)
            return image.float(), record_id

    def __len__(self):
        return len(self.df)


def show_dataset(idx, dataset):
    image, label = dataset[idx]
    times = image.shape[0]//3
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(40, 18))
    axes = axes.flatten()
    fig.tight_layout(pad=0.1)

    for time in range(times):

        axes[time].imshow(image[time::times, :, :].permute(1, 2, 0).to(torch.float))
        axes[time].axis('off')

    plt.figure()
    plt.imshow(label.permute(1, 2, 0).to(torch.float))
    plt.axis('off')

    plt.figure()
    plt.hist(image.flatten(), bins=100)
