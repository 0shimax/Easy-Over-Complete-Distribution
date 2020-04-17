import io
from torchvision import transforms
import torch
from pathlib import Path
from PIL import Image


class ImageTransform(object):
    def __init__(self):
        pass

    def __call__(self, x):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform = transforms.Compose(
            # [transforms.Resize(256),
            #  transforms.CenterCrop(224),
            [transforms.Resize(76),
             transforms.CenterCrop(64),
             transforms.ToTensor(),
             normalize,
             ])
        return transform(x)


def one_hot(labels, class_size): 
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets