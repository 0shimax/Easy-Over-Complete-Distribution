import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embeder(nn.Module):
    def __init__(self, x_dim, n_class, emb_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(x_dim, max(x_dim//2, emb_dim))
        self.fc2 = nn.Linear(max(x_dim//2, emb_dim), emb_dim)
        self.centers = torch.nn.Parameter(torch.ones([n_class, emb_dim]).to(device), requires_grad=True)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return out


class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim1, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim1)
        self.fc31 = nn.Linear(h_dim1, z_dim)
        self.fc32 = nn.Linear(h_dim1, z_dim)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var).to(device)
        eps = torch.randn_like(std).to(device)
        return eps.mul(std).add(mu) # return z sample

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        self.mu, self.log_var = self.fc31(h), self.fc32(h)
        z = self.sampling(self.mu, self.log_var)
        return z


class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim1, z_dim, emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim + emb_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim1)
        self.fc3 = nn.Linear(h_dim1, x_dim)

    def forward(self, z, emb):
        concat_input = torch.cat([z, emb], 1)
        h = F.relu(self.fc1(concat_input))
        h = F.relu(self.fc2(h))
        return F.sigmoid(self.fc3(h))


class CVAE(nn.Module):
    def __init__(self, x_dim, h_dim1, z_dim, n_class=5, emb_dim=128):
        super().__init__()
        self.embeder = Embeder(x_dim, n_class, emb_dim)
        self.encoder = Encoder(x_dim, h_dim1, z_dim)
        self.decoder = Decoder(x_dim, h_dim1, z_dim, emb_dim)
        self.class_eye = torch.eye(n_class).to(device)
    
    def forward(self, x):
        self.emb = self.embeder(x)
        z = self.encoder(x)
        return self.decoder(z, self.emb)

    def loss(self, recon_x, x, c, _lanmbda=.01):
        BCE = F.binary_cross_entropy(recon_x, x)
        KLD = -0.5 * torch.sum(1 + self.encoder.log_var - self.encoder.mu.pow(2) - self.encoder.log_var.exp())
        EMV = (self.embeder(recon_x)-self.emb).pow(2).sum(dim=1).mean()
        # CL = self._center_loss(self.emb, self.embeder.centers[c].squeeze(1))
        # OL = self._orthogonality_loss()
        # return BCE + KLD + _lanmbda*(EMV+CL+OL)
        return BCE + KLD + _lanmbda*EMV

    def _center_loss(self, predicts, targets):
        return (0.5*(predicts - targets).pow(2).sum(dim=1)).mean()

    def _orthogonality_loss(self):
        # reference:
        # http://www.cerfacs.fr/algor/reports/2003/TR_PA_03_25.pdf
        return (self.class_eye-torch.mm(self.embeder.centers, torch.t(self.embeder.centers))).pow(2).sum().pow(0.5)