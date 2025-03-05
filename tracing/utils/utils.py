import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from scipy.stats import chi2


def manual_seed(seed, fix_cudnn=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if fix_cudnn:
        torch.backends.cudnn.deterministic = True  # noqa
        torch.backends.cudnn.benchmark = False  # noqa


def spcor(x, y):
    n = len(x)
    with torch.no_grad():
        r = 1 - torch.sum(6 * torch.square(x - y)) / (n * (n**2 - 1))

    return r


def pdists(x, y):
    x = x.to("cuda")
    y = y.to("cuda")

    with torch.no_grad():
        xsum = torch.sum(torch.square(x), axis=-1)
        ysum = torch.sum(torch.square(y), axis=-1)

        dists = xsum.view(-1, 1) + ysum.view(1, -1) - 2 * x @ y.T

    return dists.cpu()


def cossim(x, y):
    x = x.to("cuda")
    y = y.to("cuda")

    with torch.no_grad():
        similarities = (
            x
            @ y.T
            / (
                torch.linalg.norm(x, axis=-1).view(-1, 1)
                * torch.linalg.norm(y, axis=-1).view(1, -1)
            )
        )

    return similarities.cpu()


def fisher(p):
    count = 0
    chi_2 = 0
    for pvalue in p:
        if not np.isnan(pvalue):
            chi_2 -= 2 * np.log(pvalue)
            count += 1

    return chi2.sf(chi_2, df=2 * count)


def normalize_mc_midpoint(mid, base, ft):
    slope = ft - base
    mid -= slope * 0.5
    mid -= base
    return mid


def normalize_trace(trace, alphas):
    slope = trace[-1] - trace[0]
    start = trace[0]
    for i in range(len(trace)):
        trace[i] -= slope * alphas[i]
        trace[i] -= start
    return trace


def output_hook(m, inp, op, name, feats):
    feats[name] = op.detach()


def get_submodule(module, submodule_string):
    attributes = submodule_string.split(".")
    for attr in attributes:
        module = getattr(module, attr)
    return module


def plot_trace(losses, alphas, normalize, model_a_name, model_b_name, plot_path):

    plt.figure(figsize=(8, 6))
    if normalize:
        losses = normalize_trace(losses, alphas)
    plt.plot(alphas, losses, "o-")

    plt.xlabel("Alpha")
    plt.ylabel("Loss")
    plt.title(f"{model_a_name} (Left) vs {model_b_name} (Right)")
    plot_filename = f"{plot_path}.png"

    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()
