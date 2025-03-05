import torch
from collections import defaultdict
import scipy
import numpy as np

from tracing.utils.evaluate import evaluate
from tracing.utils.llama.matching import match_wmats


def hook_in(m, inp, op, feats, name):
    feats[name].append(inp[0].detach().cpu())


def hook_out(m, inp, op, feats, name):
    feats[name].append(op.detach().cpu())


def statistic(base_model, ft_model, dataloader, n_blocks=32):

    stats = []

    for i in range(n_blocks):

        gate_match = mlp_matching_gate(base_model, ft_model, dataloader, i=i)
        up_match = mlp_matching_up(base_model, ft_model, dataloader, i=i)

        cor, pvalue = scipy.stats.spearmanr(gate_match.tolist(), up_match.tolist())
        print(i, pvalue, len(gate_match))
        stats.append(pvalue)

    return stats


def statistic_layer(base_model, ft_model, dataloader, i=0):
    gate_perm = mlp_matching_gate(base_model, ft_model, dataloader, i=i)
    up_perm = mlp_matching_up(base_model, ft_model, dataloader, i=i)
    cor, pvalue = scipy.stats.spearmanr(gate_perm.tolist(), up_perm.tolist())
    return pvalue


def mlp_matching_gate(base_model, ft_model, dataloader, i=0):
    feats = defaultdict(list)

    base_hook = lambda *args: hook_out(*args, feats, "base")
    base_handle = base_model.model.layers[i].mlp.gate_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args: hook_out(*args, feats, "ft")
    ft_handle = ft_model.model.layers[i].mlp.gate_proj.register_forward_hook(ft_hook)

    evaluate(base_model, dataloader)
    evaluate(ft_model, dataloader)

    base_mat = torch.vstack(feats["base"])
    ft_mat = torch.vstack(feats["ft"])

    base_mat.to("cuda")
    ft_mat.to("cuda")

    base_mat = base_mat.view(-1, base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1, ft_mat.shape[-1]).T

    base_handle.remove()
    ft_handle.remove()

    perm = match_wmats(base_mat, ft_mat)

    return perm


def mlp_matching_up(base_model, ft_model, dataloader, i=0):
    feats = defaultdict(list)

    base_hook = lambda *args: hook_out(*args, feats, "base")
    base_handle = base_model.model.layers[i].mlp.up_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args: hook_out(*args, feats, "ft")
    ft_handle = ft_model.model.layers[i].mlp.up_proj.register_forward_hook(ft_hook)

    evaluate(base_model, dataloader)
    evaluate(ft_model, dataloader)

    base_mat = torch.vstack(feats["base"])
    ft_mat = torch.vstack(feats["ft"])

    base_mat.to("cuda")
    ft_mat.to("cuda")

    base_mat = base_mat.view(-1, base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1, ft_mat.shape[-1]).T

    base_handle.remove()
    ft_handle.remove()

    perm = match_wmats(base_mat, ft_mat)

    return perm


def mlp_layers(base_model, ft_model, dataloader, i, j):

    gate_match = mlp_matching_gate(base_model, ft_model, dataloader, i, j)
    up_match = mlp_matching_up(base_model, ft_model, dataloader, i, j)

    cor, pvalue = scipy.stats.spearmanr(gate_match.tolist(), up_match.tolist())

    return pvalue


def statistic_all(model_1, model_2, dataloader):
    model_1_matched = np.zeros(model_1.config.num_hidden_layers)
    model_2_matched = np.zeros(model_2.config.num_hidden_layers)

    for i in range(model_1.config.num_hidden_layers):
        for j in range(model_2.config.num_hidden_layers):
            if model_1_matched[i] == 1 or model_2_matched[j] == 1:
                continue
            stat = mlp_layers(model_1, model_2, dataloader, i, j)
            print(i, j, stat)
            if stat < 0.000001:
                model_1_matched[i] = 1
                model_2_matched[j] = 1
                break
