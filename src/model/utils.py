import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CenterCalculator(object):
    def __init__(self, n_class, centers=None):
        if centers is None:
            self.center = torch.zeros([n_class, n_class]).to(device)
        else:
            self.center = centers
        self.n_class = n_class

    def calculate_centers(self, batch_vectors, targets):
        counter = {c:0 for c in range(n_class)}
        for i, t in enumerate(targets):
            counter[t] += 1
            self.center[t] = self.center[t] + batch_vectors[i]
        
        for k, v in counter.items():
            self.center[k] = self.center[k]/(v+1)