from pathlib import Path
import argparse
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pandas

from model.cvae import CVAE
from feature.data_loader import WBCDataset, loader
from feature.utils import one_hot
from optimizer.radam import RAdam


torch.manual_seed(555)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def train(args, mnasnet1_0, model, optimizer, data_loader):
    train_loss = 0
    for epoch in range(args.epoch):
        for batch_idx, (data, cond) in enumerate(data_loader):
            data, cond = data.to(device), one_hot(cond, args.n_class).to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                x = mnasnet1_0(data)
            recon_batch = model(x, cond)
            loss = model.loss(recon_batch, x.detach())
            
            loss.backward()
            train_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_loader.dataset)))
    torch.save(model.state_dict(),
               '{}/model_phase1.pth'.format(args.out_dir))

def main(args):
    mnasnet1_0 = models.mnasnet1_0(pretrained=True).to(device).eval()
    model = CVAE(1000, 128, args.n_class*2, args.n_class).to(device)

    image_label = pandas.read_csv(
        Path(args.data_root, 
             args.metadata_file_name.format(args.subset))
    ).sample(frac=1, random_state=551)[:250]
    image_label["class"] = image_label["class"] - 1
    dataset = WBCDataset(image_label.values, args.data_root, subset=args.subset)

    data_loader = loader(dataset, args.batch_size, True)
    optimizer = RAdam(model.parameters(), weight_decay=1e-3)
    train(args, mnasnet1_0, model, optimizer, data_loader)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-class', type=int, default=5, help='number of class')
    parser.add_argument('--data-root', default="./data/segmentation_WBC-master")
    parser.add_argument('--metadata-file-name', default="Class_Labels_of_{}.csv")
    parser.add_argument('--subset', default="Dataset1")
    parser.add_argument('--epoch', default=50, help='number of epoch')
    parser.add_argument('--batch-size', default=16, help='number of batch')
    parser.add_argument('--out-dir', default='./result/wbc', help='folder to output data and model checkpoints')
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True),

    main(args)