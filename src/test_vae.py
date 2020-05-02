import os
from pathlib import Path
import numpy 
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import pandas

from feature.data_loader import WBCDataset, loader
from model.cvae import CVAE


torch.manual_seed(555)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def test(args, mnasnet, model, data_loader, show_image_on_board=True):
    writer = SummaryWriter()
    images = []
    labels = []
    weights = []    
    with torch.no_grad():
        for i, (image, label) in enumerate(data_loader):
            image = image.to(device)
            images.append(image.squeeze(0).numpy())
            label = label.squeeze(0).numpy()[0]
            labels.append(label)

            x = mnasnet(image)
            recon_x = model(x)
            weights.append(x.squeeze(0).numpy())

    weights = torch.FloatTensor(weights)
    writer.add_embedding(weights, metadata=labels)
    print("done")


def main(args):
    mnasnet = models.mnasnet1_0(pretrained=True).to(device).eval()
    model = CVAE(1000, 128, 128, args.n_class, 128).to(device).eval()
    if Path(args.resume_model).exists():
        print("load regressor model:", args.resume_model)
        model.load_state_dict(torch.load(args.resume_model))

    image_label = pandas.read_csv(
        Path(args.data_root, 
             args.metadata_file_name.format(args.subset))
    ).sample(frac=1, random_state=551) #[250:]
    image_label["class"] = image_label["class"] - 1
    dataset = WBCDataset(image_label.values, args.data_root, subset=args.subset)
    data_loader = loader(dataset, 1, False)
    test(args, mnasnet, model, data_loader)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-class', type=int, default=5, help='number of class')
    parser.add_argument('--resume-model', default='./result/wbc/model_phase1_emb.pth', help='path to trained model')
    parser.add_argument('--data-root', default="./data/segmentation_WBC-master")
    parser.add_argument('--metadata-file-name', default="Class_Labels_of_{}.csv")
    parser.add_argument('--subset', default="Dataset1")
    parser.add_argument('--out-dir', default='./result/wbc', help='folder to output data and model checkpoints')
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True),

    main(args)
