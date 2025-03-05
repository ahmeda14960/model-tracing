import torch
from collections import defaultdict
import scipy
import numpy as np
from scipy.stats import chi2

from scipy.optimize import linear_sum_assignment as LAP

from tracing.utils.utils import cossim
from tracing.utils.evaluate import evaluate


def statistic(base_model, ft_model, dataloader):
    return csh_sp_dataloader(base_model, ft_model, dataloader)


def hook(m, inp, op, feats, name):
    feats[name].append(op.detach().cpu())


def hook_in(m, inp, op, feats, name):
    feats[name].append(inp[0].detach().cpu())


def csh_sp_dataloader_block(base_model, ft_model, dataloader, i):

    feats = defaultdict(list)

    base_hook = lambda *args: hook(*args, feats, "base")
    base_model.model.layers[i].mlp.down_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args: hook(*args, feats, "ft")
    ft_model.model.layers[i].mlp.down_proj.register_forward_hook(ft_hook)

    evaluate(base_model, dataloader)
    evaluate(ft_model, dataloader)

    base_mat = torch.vstack(feats["base"])
    ft_mat = torch.vstack(feats["ft"])

    base_mat = base_mat.view(-1, base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1, ft_mat.shape[-1]).T

    matched = torch.argmax(cossim(base_mat, ft_mat), axis=-1)
    orig = torch.arange(len(matched))

    cor, pvalue = scipy.stats.spearmanr(matched.tolist(), orig.tolist())
    return pvalue


def csh_sp_dataloader(base_model, ft_model, dataloader, n_blocks=32):

    chi_squared = 0
    feats = defaultdict(list)

    base_hooks = {}
    ft_hooks = {}

    for i in range(n_blocks):

        layer = str(i)

        base_hooks[layer] = lambda m, inp, op, layer=layer, feats=feats: hook(
            m, inp, op, feats, "base_" + layer
        )
        base_model.model.layers[i].mlp.up_proj.register_forward_hook(base_hooks[layer])

        ft_hooks[layer] = lambda m, inp, op, layer=layer, feats=feats: hook(
            m, inp, op, feats, "ft_" + layer
        )
        ft_model.model.layers[i].mlp.up_proj.register_forward_hook(ft_hooks[layer])

    evaluate(base_model, dataloader)
    evaluate(ft_model, dataloader)

    p_values = []
    count = 0

    for i in range(n_blocks):

        base_mat = torch.vstack(feats["base_" + str(i)])
        ft_mat = torch.vstack(feats["ft_" + str(i)])

        base_mat = torch.reshape(
            base_mat, (base_mat.shape[0] * base_mat.shape[1], base_mat.shape[2])
        )
        ft_mat = torch.reshape(ft_mat, (ft_mat.shape[0] * ft_mat.shape[1], ft_mat.shape[2]))

        base_mat = base_mat.T
        ft_mat = ft_mat.T

        matched = LAP(
            cossim(base_mat.type(torch.float64), ft_mat.type(torch.float64)), maximize=True
        )
        matched = matched[1]
        orig = torch.arange(len(matched))

        cor, temp = scipy.stats.spearmanr(matched.tolist(), orig.tolist())

        if not np.isnan(temp):
            chi_squared -= 2 * np.log(temp)
            count += 1
            print(i, temp)
        p_values.append(temp)

    p_value = chi2.sf(chi_squared, df=2 * count)

    return p_value, p_values
