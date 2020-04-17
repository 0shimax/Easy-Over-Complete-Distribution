from sklearn.metrics import confusion_matrix, classification_report
from scipy import spatial
import collections
import numpy
import torch
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_result(gt, predicted):
    print("confusion_matrix: \n", confusion_matrix(gt, predicted))
    print("\n report: \n", classification_report(gt, predicted))


def cossim(image_vec, cat_matrix):
    cos_sim = []
    cos_sim = cat_matrix.dot(image_vec)
    idx = numpy.argmax(cos_sim)
    return idx


def val(args, model, data_loader, emb_dim=128):
    model.eval()
    labels = []
    label_idxs = []
    center = numpy.zeros([args.n_class, emb_dim])
    with torch.no_grad():
        for i, (image, cat) in enumerate(data_loader):
            image = image.to(device)
            label_idxs.append(cat.item())
            cat = cat.to(device)

            image_embedded_vec = model.predict(x=image, category=None)
            vec = F.softmax(image_embedded_vec, dim=1).squeeze(0).numpy()
            center[cat.item()] += vec
    cnt = collections.Counter(label_idxs)
    print(cnt)
    for c, v in cnt.items():
        center[c] /= v
    return center
