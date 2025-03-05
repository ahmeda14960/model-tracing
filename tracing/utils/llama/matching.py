VOCAB_SIZE = 32000

import torch
from scipy.optimize import linear_sum_assignment as LAP

from ..utils import pdists
from .model import permute_model, permute_transformer_block


def match_wmats(wmat0, wmat1):
    dists = pdists(wmat0, wmat1).type(torch.float64)
    perm = LAP(dists)[1]

    return perm  # wmat1[perm] should match wmat0


def match_mlp(base_model, ft_model, i=0):
    base_wmat = base_model.state_dict()["model.layers." + str(i) + ".mlp.gate_proj.weight"]
    ft_wmat = ft_model.state_dict()["model.layers." + str(i) + ".mlp.gate_proj.weight"]

    perm = match_wmats(base_wmat, ft_wmat)

    return perm


def match_emb(base_model, ft_model, i="inp"):
    if i == "inp":
        weight_id = "model.embed_tokens.weight"
    if i == "out":
        weight_id = "lm_head.weight"

    base_wmat = base_model.state_dict()[weight_id][:VOCAB_SIZE].T
    ft_wmat = ft_model.state_dict()[weight_id][:VOCAB_SIZE].T

    perm = match_wmats(base_wmat, ft_wmat)
    return perm


def align_model(base_model, ft_model, tmp_model, n_blocks=32):
    emb_perm = match_emb(base_model, ft_model)
    permute_model(ft_model, tmp_model, torch.arange(11008), emb_perm)

    for i in range(n_blocks):
        mlp_perm = match_mlp(base_model, tmp_model, i=i)
        permute_transformer_block(tmp_model, i, tmp_model, mlp_perm, torch.arange(4096))
