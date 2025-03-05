import torch
from tracing.perm.permute import permute_model
from scripts.perm.main import p_value_exact, p_value_approx


def statistic(base_model, ft_model, mc_stat, l2_stat, num_perm, emb_dim=4096, mlp_dim=11008):

    unperm_stat_mc = mc_stat(base_model, ft_model)
    unperm_stat_l2 = l2_stat(base_model, ft_model)

    print(unperm_stat_mc, unperm_stat_l2)

    perm_stats_mc = []
    perm_stats_l2 = []

    for i in range(num_perm):
        mlp_permutation = torch.randperm(mlp_dim)
        emb_permutation = torch.randperm(emb_dim)

        permute_model(ft_model, mlp_permutation, emb_permutation)

        perm_stat_mc = mc_stat(base_model, ft_model)
        perm_stat_l2 = l2_stat(base_model, ft_model)

        perm_stats_mc.append(perm_stat_mc)
        perm_stats_l2.append(perm_stat_l2)

        print(i, perm_stat_mc, perm_stat_l2)

    exact_mc = p_value_exact(unperm_stat_mc, perm_stats_mc.copy())
    approx_mc = p_value_approx(unperm_stat_mc, perm_stats_mc.copy())

    exact_l2 = p_value_exact(unperm_stat_l2, perm_stats_l2.copy())
    approx_l2 = p_value_approx(unperm_stat_l2, perm_stats_l2.copy())

    print(exact_mc, approx_mc)
    print(exact_l2, approx_l2)

    return (
        exact_mc,
        approx_mc,
        exact_l2,
        approx_l2,
        unperm_stat_mc,
        unperm_stat_l2,
        perm_stats_mc,
        perm_stats_l2,
    )
