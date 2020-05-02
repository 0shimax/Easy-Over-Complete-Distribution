from pathlib import Path
import argparse
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pandas
import random
random.seed(555)

from feature.phase2_dataloader import WBCDataset, loader
from feature.utils import one_hot
from model.regressor import Regressor
from model.cvae import CVAE
from optimizer.radam import RAdam


torch.manual_seed(555)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def train(args, mnasnet, cvae_model, model, optimizer, data_loader):
    train_loss = 0
    for epoch in range(args.epoch):
        for batch_idx, (anchor, positive, negative, label, neg_label) in enumerate(data_loader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            label = label.to(device)
            neg_label = neg_label.to(device)

            with torch.no_grad():
                anchor = mnasnet(anchor)
                positive = mnasnet(positive)
                negative = mnasnet(negative)

            if random.random()>0.9:
                anchor = cvae_model(anchor)
                positive = cvae_model(positive)
                negative = cvae_model(negative)

            optimizer.zero_grad()
            
            anc_cl, pos_cl, neg_cl = model(anchor, positive, negative)
            loss = model.loss(anc_cl, pos_cl, neg_cl, label, label, neg_label)
            
            loss.backward()
            train_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * args.batch_size, len(data_loader.dataset),
                    100. * batch_idx / len(data_loader), loss.item() / args.batch_size))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_loader.dataset)))
    torch.save(model.state_dict(),
               '{}/model_phase2_vanira.pth'.format(args.out_dir))


def main(args):
    mnasnet = models.mnasnet1_0(pretrained=True).to(device).eval()
    cvae_model = CVAE(1000, 128, 128, args.n_class, 128).to(device).eval()
    if Path(args.resume_model).exists():
        print("load model:", args.resume_model)
        cvae_model.load_state_dict(torch.load(args.resume_model))

    model = Regressor().to(device)

    image_label = pandas.read_csv(
        Path(args.data_root, 
             args.metadata_file_name.format(args.subset))
    ).sample(frac=1, random_state=551)[:250]
    image_label["class"] = image_label["class"] - 1

    dataset = WBCDataset(args.n_class, image_label[:250].values, args.data_root, 
                         subset=args.subset, train=True)
    data_loader = loader(dataset, args.batch_size, True)
    optimizer = RAdam(model.parameters(), weight_decay=1e-3)
    train(args, mnasnet, cvae_model, model, optimizer, data_loader)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-class', type=int, default=5, help='number of class')
    parser.add_argument('--data-root', default="./data/segmentation_WBC-master")
    parser.add_argument('--metadata-file-name', default="Class_Labels_of_{}.csv")
    parser.add_argument('--subset', default="Dataset1")
    parser.add_argument('--resume-model', default='./result/wbc/model_phase1_emb.pth', help='path to trained model')
    parser.add_argument('--epoch', default=50, help='number of epoch')
    parser.add_argument('--batch-size', default=16, help='number of batch')
    parser.add_argument('--out-dir', default='./result/wbc', help='folder to output data and model checkpoints')
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True),

    main(args)