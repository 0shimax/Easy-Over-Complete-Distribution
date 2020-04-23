import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    def __init__(self, x_dim, h_dim1, z_dim, c_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim + c_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim1)
        self.fc3 = nn.Linear(h_dim1, x_dim)

    def forward(self, z, c):
        concat_input = torch.cat([z, c], 1)
        h = F.relu(self.fc1(concat_input))
        h = F.relu(self.fc2(h))
        return F.sigmoid(self.fc3(h))


class CVAE(nn.Module):
    def __init__(self, x_dim, h_dim1, z_dim, c_dim):
        super().__init__()
        # # encoder part
        # self.fc1 = nn.Linear(x_dim, h_dim1)
        # self.fc2 = nn.Linear(h_dim1, h_dim1)
        # self.fc31 = nn.Linear(h_dim1, z_dim)
        # self.fc32 = nn.Linear(h_dim1, z_dim)
        # # decoder part
        # self.fc4 = nn.Linear(z_dim + c_dim, h_dim1)
        # self.fc5 = nn.Linear(h_dim1, h_dim1)
        # self.fc6 = nn.Linear(h_dim1, x_dim)
        self.encoder = Encoder(x_dim, h_dim1, z_dim)
        self.decoder = Decoder(x_dim, h_dim1, z_dim, c_dim)

    # def encoder(self, x):
    #     h = F.relu(self.fc1(x))
    #     h = F.relu(self.fc2(h))
    #     return self.fc31(h), self.fc32(h)
    
    # def sampling(self, mu, log_var):
    #     std = torch.exp(0.5*log_var).to(device)
    #     eps = torch.randn_like(std).to(device)
    #     return eps.mul(std).add(mu) # return z sample
    
    # def decoder(self, c):
    #     concat_input = torch.cat([self.z, c], 1)
    #     h = F.relu(self.fc4(concat_input))
    #     h = F.relu(self.fc5(h))
    #     return F.sigmoid(self.fc6(h))
    
    def forward(self, x, c):
        z = self.encoder(x)
        return self.decoder(z, c)

    def loss(self, recon_x, x):
        BCE = F.binary_cross_entropy(recon_x, x)
        KLD = -0.5 * torch.sum(1 + self.encoder.log_var - self.encoder.mu.pow(2) - self.encoder.log_var.exp())
        return BCE + KLD
