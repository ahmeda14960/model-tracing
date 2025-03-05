"""
Implementation of L2 distance metrics for comparing neural network model weights.

This module provides functions to calculate the L2 (Euclidean) distance between
corresponding weight tensors of two models, providing a measure of parameter space
similarity.
"""

import torch


def statistic(base_model, ft_model):
    """
    Compute the average L2 distance between weights of two models.

    Args:
        base_model: Base model to compare
        ft_model: Fine-tuned or target model to compare against the base model

    Returns:
        float: Average L2 distance across all comparable parameters
    """
    return calculate_l2_distance(base_model, ft_model)


def calculate_l2_distance(model1, model2):
    """
    Calculate the average L2 distance between corresponding parameters of two models.

    For each parameter tensor in the models, computes the Euclidean distance between
    them and returns the average across all parameters. Handles potential shape
    mismatches in embedding or output layers.

    Args:
        model1: First model to compare
        model2: Second model to compare

    Returns:
        float: Average L2 distance across all comparable parameters

    Raises:
        ValueError: If parameter names don't match or if there are shape mismatches
                   in parameters other than embedding or output layers
    """
    total_squared_diff = 0
    num_layers = 0

    all_layers = []

    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if name1 != name2:
            raise ValueError(f"Model parameter names do not match: {name1} != {name2}")
        elif param1.shape != param2.shape:
            if name1 == "model.embed_tokens.weight" or name1 == "lm_head.weight":
                print(
                    f"Skipping {name1} because of shape mismatch: {param1.shape} != {param2.shape}"
                )
                continue
            raise ValueError(
                f"Model parameter shapes do not match for {name1}: {param1.shape} != {param2.shape}"
            )

        l2_diff = torch.sum((param1 - param2) ** 2) ** 0.5
        total_squared_diff += l2_diff.item()
        all_layers.append(l2_diff.item())
        num_layers += 1

    avg_l2_distance = total_squared_diff / num_layers
    return avg_l2_distance
