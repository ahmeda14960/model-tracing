"""
Implementation of activation matching algorithms for comparing neural network models.

This module provides functions for matching neurons between two models based on
their activation patterns, helping identify corresponding functional units despite
permutation differences.
"""

import torch
from collections import defaultdict
import scipy
import numpy as np

from tracing.utils.evaluate import evaluate
from tracing.utils.llama.matching import match_wmats


def hook_in(m, inp, op, feats, name):
    """
    Forward hook to capture input activations to model layers.

    Args:
        m: Module being hooked
        inp: Input to the module (tuple)
        op: Output from the module
        feats: Dictionary to store activations
        name: Key to store the activations under
    """
    feats[name].append(inp[0].detach().cpu())


def hook_out(m, inp, op, feats, name):
    """
    Forward hook to capture output activations from model layers.

    Args:
        m: Module being hooked
        inp: Input to the module
        op: Output from the module
        feats: Dictionary to store activations
        name: Key to store the activations under
    """
    feats[name].append(op.detach().cpu())


def statistic(base_model, ft_model, dataloader, n_blocks=32):
    """
    Compute neuron matching statistics across all transformer blocks.

    For each block, compares the gate and up projections to determine if
    the permutation patterns are consistent, which would indicate functionally
    corresponding neurons.

    Args:
        base_model: Base model to compare
        ft_model: Fine-tuned or target model to compare against the base model
        dataloader: DataLoader providing input data for activation collection
        n_blocks: Number of transformer blocks to analyze (default: 32)

    Returns:
        list: Spearman correlation p-values for each block
    """
    stats = []

    for i in range(n_blocks):
        gate_match = mlp_matching_gate(base_model, ft_model, dataloader, i=i)
        up_match = mlp_matching_up(base_model, ft_model, dataloader, i=i)

        cor, pvalue = scipy.stats.spearmanr(gate_match.tolist(), up_match.tolist())
        print(i, pvalue, len(gate_match))
        stats.append(pvalue)

    return stats


def statistic_layer(base_model, ft_model, dataloader, i=0):
    """
    Compute neuron matching statistics for a specific transformer block.

    Args:
        base_model: Base model to compare
        ft_model: Fine-tuned or target model to compare against the base model
        dataloader: DataLoader providing input data for activation collection
        i: Block index to analyze (default: 0)

    Returns:
        float: Spearman correlation p-value for the specified block
    """
    gate_perm = mlp_matching_gate(base_model, ft_model, dataloader, i=i)
    up_perm = mlp_matching_up(base_model, ft_model, dataloader, i=i)
    cor, pvalue = scipy.stats.spearmanr(gate_perm.tolist(), up_perm.tolist())
    return pvalue


def mlp_matching_gate(base_model, ft_model, dataloader, i=0):
    """
    Match neurons between models by comparing gate projection activations.

    Collects activations from the gate projection layer for both models
    and computes a permutation that would align corresponding neurons.

    Args:
        base_model: Base model to compare
        ft_model: Fine-tuned or target model to compare against the base model
        dataloader: DataLoader providing input data for activation collection
        i: Block index to analyze (default: 0)

    Returns:
        torch.Tensor: Permutation indices that match neurons between models
    """
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
    """
    Match neurons between models by comparing up projection activations.

    Collects activations from the up projection layer for both models
    and computes a permutation that would align corresponding neurons.

    Args:
        base_model: Base model to compare
        ft_model: Fine-tuned or target model to compare against the base model
        dataloader: DataLoader providing input data for activation collection
        i: Block index to analyze (default: 0)

    Returns:
        torch.Tensor: Permutation indices that match neurons between models
    """
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
    """
    Compare gate and up projections between specific layers of two models.

    Useful for comparing non-corresponding layers to find functional similarities.

    Args:
        base_model: Base model to compare
        ft_model: Fine-tuned or target model to compare against the base model
        dataloader: DataLoader providing input data for activation collection
        i: Layer index in the base model
        j: Layer index in the fine-tuned model

    Returns:
        float: Spearman correlation p-value between gate and up projections
    """
    gate_match = mlp_matching_gate(base_model, ft_model, dataloader, i, j)
    up_match = mlp_matching_up(base_model, ft_model, dataloader, i, j)

    cor, pvalue = scipy.stats.spearmanr(gate_match.tolist(), up_match.tolist())

    return pvalue


def statistic_all(model_1, model_2, dataloader):
    """
    Perform comprehensive layer matching between two models.

    Tests all combinations of layers between the models to identify corresponding
    functional units, regardless of their position in the network architecture.

    Args:
        model_1: First model to compare
        model_2: Second model to compare
        dataloader: DataLoader providing input data for activation collection

    Returns:
        None: Prints matching results during execution
    """
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
