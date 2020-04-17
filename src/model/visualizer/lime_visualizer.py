from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.segmentation import mark_boundaries
from lime import lime_image


explainer = lime_image.LimeImageExplainer()


def normalize(x):
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

def visualize(args, image, predict_fn, img_idx, label_id, gt_id, num_samples=1000):
    data = image.transpose(2,1).transpose(3,2).squeeze(0).numpy()
    explanation = explainer.explain_instance(data, predict_fn, top_labels=5, hide_color=0, num_samples=num_samples)
    temp, mask = explanation.get_image_and_mask(int(gt_id), positive_only=False, num_features=1000, hide_rest=True)
    out = mark_boundaries(temp / 2 + 0.5, mask)
    out = normalize(out)
    plt.imsave(Path(args.out_dir, "visualized", "testimg{}_gt{}_pred{}.jpeg".format(img_idx, gt_id, label_id)), out)
    plt.imsave(Path(args.out_dir, "visualized", "testimg{}.jpeg".format(img_idx)), normalize(data))
