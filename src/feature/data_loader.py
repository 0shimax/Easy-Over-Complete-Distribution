from pathlib import Path
import numpy
from skimage import io
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image

from feature.utils import ImageTransform


class WBCDataset(Dataset):
    def __init__(self, image_labels, root_dir,
                 subset="Dataset1", transform=ImageTransform()):
        super().__init__()
        self.image_labels = image_labels
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_name, label = self.image_labels[idx]
        image_path = Path(self.root_dir, self.subset, "{0:03}.bmp".format(img_name))
        image = io.imread(image_path)

        if self.transform:
            image = Image.fromarray(numpy.uint8(image))
            image = self.transform(image)
        return image, torch.LongTensor([label])


def loader(dataset, batch_size,  shuffle=True):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4)
    return loader