import torch
import numpy as np

from tracing.perm.permute import permute_model


def main(base_model, ft_model, test_stat, num_perm, emb_dim=4096, mlp_dim=11008):

    unperm_stat = test_stat(base_model, ft_model)
    print(unperm_stat)

    perm_stats = []

    for i in range(num_perm):

        mlp_permutation = torch.randperm(mlp_dim)
        emb_permutation = torch.randperm(emb_dim)

        permute_model(ft_model, mlp_permutation, emb_permutation)

        perm_stat = test_stat(base_model, ft_model)

        perm_stats.append(perm_stat)
        print(i, perm_stat)

    print(perm_stats)
    exact = p_value_exact(unperm_stat, perm_stats.copy())
    approx = p_value_approx(unperm_stat, perm_stats.copy())

    print(exact, approx)

    return exact, approx, unperm_stat, perm_stats


def p_value_exact(unpermuted, permuted):
    count = 0
    for a in permuted:
        if a < unpermuted:
            count += 1
    return round((count + 1) / (len(permuted) + 1), 2)


def p_value_approx(unpermuted, permuted):
    mean = sum(permuted) / len(permuted)
    std = np.std(permuted)
    zscore = (unpermuted - mean) / std
    return zscore
