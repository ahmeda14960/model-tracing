"""
Implementation of Cosine Similarity of Weights (CSW) test for comparing neural network models.

This module provides functions to test whether two models have similar weight matrices
using cosine similarity and statistical tests to quantify the similarity.
"""

import torch

from tracing.utils.utils import cossim, fisher
import scipy
import numpy as np
from scipy.stats import chi2

from scipy.optimize import linear_sum_assignment as LAP


def statistic(base_model, ft_model):
    """
    Compute Cosine Similarity of Weights statistic between two models.

    Args:
        base_model: Base model to compare
        ft_model: Fine-tuned or target model to compare against the base model

    Returns:
        tuple: (aggregate_p_value, p_values_per_layer) from the CSW test
    """
    return csw_sp(base_model, ft_model)


def csw_sp_layer(base_model, ft_model, layer_name):
    """
    Calculate Cosine Similarity of Weights for a specific layer.

    Uses linear assignment to find optimal matching between neurons in the layer
    and calculates Spearman correlation to quantify similarity.

    Args:
        base_model: Base model to compare
        ft_model: Fine-tuned or target model to compare against the base model
        layer_name: Name of the layer in the model's state dict to analyze

    Returns:
        float: p-value indicating the statistical similarity of weight matrices
    """
    base_mat = base_model.state_dict()[layer_name]
    ft_mat = ft_model.state_dict()[layer_name]

    matched = LAP(cossim(base_mat.type(torch.float64), ft_mat.type(torch.float64)), maximize=True)
    matched = matched[1]
    orig = torch.arange(len(matched))

    cor, pvalue = scipy.stats.spearmanr(matched.tolist(), orig.tolist())
    return pvalue


def csw_sp(model1, model2):
    """
    Apply CSW test across all MLP up-projection layers in the models.

    Performs Fisher's method to combine p-values from individual layer tests
    into an aggregate statistic.

    Args:
        model1: First model to compare
        model2: Second model to compare

    Returns:
        tuple: (aggregate_p_value, list_of_p_values_per_layer)
    """
    chi_squared = 0
    num_layers = 0

    p_values = []

    for name1, name2 in zip(list(model1.state_dict().keys()), list(model2.state_dict().keys())):
        if name1 != name2:
            raise ValueError(f"Model parameter names do not match: {name1} != {name2}")
        elif "mlp.up_proj" not in name1:
            continue

        pvalue = csw_sp_layer(model1, model2, name1)
        if not np.isnan(pvalue):
            chi_squared -= 2 * np.log(pvalue)
            num_layers += 1
        p_values.append(pvalue)

        print(name1, pvalue)

    aggregate_pvalue = chi2.sf(chi_squared, df=2 * num_layers)
    return aggregate_pvalue, p_values


def csw_sp_pair(base_model, ft_model, layer_name_base, layer_name_ft):
    """
    Calculate Cosine Similarity of Weights between two specific layers.

    Similar to csw_sp_layer but allows comparing layers with different names.

    Args:
        base_model: Base model to compare
        ft_model: Fine-tuned or target model to compare against the base model
        layer_name_base: Name of the layer in the base model's state dict
        layer_name_ft: Name of the layer in the fine-tuned model's state dict

    Returns:
        float: p-value indicating the statistical similarity of weight matrices
    """
    base_mat = base_model.state_dict()[layer_name_base]
    ft_mat = ft_model.state_dict()[layer_name_ft]

    matched = LAP(cossim(base_mat.type(torch.float64), ft_mat.type(torch.float64)), maximize=True)
    matched = matched[1]
    orig = torch.arange(len(matched))

    cor, pvalue = scipy.stats.spearmanr(matched.tolist(), orig.tolist())
    return pvalue


def statistic_all(base_model, ft_model):
    """
    Compute comprehensive pairwise comparisons between all compatible layers.

    Tests every possible layer pairing between models that have compatible shapes,
    useful for exploring model structure similarities without assumptions.

    Args:
        base_model: Base model to compare
        ft_model: Fine-tuned or target model to compare against the base model

    Returns:
        float: Aggregate p-value from Fisher's method combining all layer comparisons
    """
    base_model.to("cpu")
    ft_model.to("cpu")

    weights_base = base_model.state_dict()
    weights_ft = ft_model.state_dict()

    shapes_base = {}
    shapes_ft = {}

    for name1 in list(weights_base.keys()):
        shapes_base[name1] = weights_base[name1].shape
    for name2 in list(weights_ft.keys()):
        shapes_ft[name2] = weights_ft[name2].shape

    pvalues = []

    for name1 in list(weights_base.keys()):
        for name2 in list(weights_ft.keys()):
            if shapes_base[name1] == shapes_ft[name2] and len(shapes_base[name1]) != 1:
                pval = csw_sp_pair(base_model, ft_model, name1, name2)
                print(name1, name2, pval)
                pvalues.append(pval)

    print(pvalues)

    res = 0

    if len(pvalues) == 0:
        res = 999
    else:
        res = fisher(pvalues)

    print(res)
    return res
