import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def distance(x, y):
    return (x-y).pow(2).sum(dim=1).unsqueeze(dim=1)


def hinge_loss(d_near, d_far, margin=1.):
    loss = d_near - d_far + margin
    return torch.clamp(loss, min=0.)
    

class Regressor(nn.Module):
    def __init__(self, x_dim=1000, n_class=5, h_dim=128):
        super().__init__()
        self.fc = nn.Linear(x_dim, h_dim)
        self.cl = nn.Linear(h_dim, n_class)

    def predict(self, x):
        self.emb = F.relu(self.fc(x))
        return self.cl(self.emb), self.emb

    def forward(self, anchor, positive, negative):
        anc_cl, anc_emb = self.predict(anchor)
        pos_cl, pos_emb = self.predict(positive)
        neg_cl, neg_emb = self.predict(negative)
        self.d1 = distance(anc_emb, pos_emb)
        self.d2 = distance(anc_emb, neg_emb)
        return anc_cl, pos_cl, neg_cl
        
    def loss(self, y_anc, y_pos, y_neg, t_anc, t_pos, t_neg):
        y = torch.cat([y_anc, y_pos, y_neg], 0)
        c = torch.cat([t_anc, t_pos, t_neg], 0)
        CL = F.cross_entropy(y, c.squeeze(1))
        return hinge_loss(self.d1, self.d2).mean() + CL