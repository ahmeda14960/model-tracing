import copy


def permute_model(model, tmp_model, mlp_permutation, emb_permutation, n_blocks=32):
    permute_embedding_layer(model, tmp_model, emb_permutation)
    for i in range(n_blocks):
        permute_transformer_block(tmp_model, i, tmp_model, mlp_permutation, emb_permutation)
    permute_output_layer(tmp_model, tmp_model, emb_permutation)


def permute_transformer_block(model, i, tmp_model, mlp_permutation, emb_permutation):
    weights = model.state_dict()

    weights["model.layers." + str(i) + ".self_attn.q_proj.weight"] = weights[
        "model.layers." + str(i) + ".self_attn.q_proj.weight"
    ][:, emb_permutation]
    weights["model.layers." + str(i) + ".self_attn.k_proj.weight"] = weights[
        "model.layers." + str(i) + ".self_attn.k_proj.weight"
    ][:, emb_permutation]
    weights["model.layers." + str(i) + ".self_attn.v_proj.weight"] = weights[
        "model.layers." + str(i) + ".self_attn.v_proj.weight"
    ][:, emb_permutation]
    weights["model.layers." + str(i) + ".self_attn.o_proj.weight"] = weights[
        "model.layers." + str(i) + ".self_attn.o_proj.weight"
    ][emb_permutation]

    weights["model.layers." + str(i) + ".mlp.gate_proj.weight"] = weights[
        "model.layers." + str(i) + ".mlp.gate_proj.weight"
    ][mlp_permutation]
    weights["model.layers." + str(i) + ".mlp.up_proj.weight"] = weights[
        "model.layers." + str(i) + ".mlp.up_proj.weight"
    ][mlp_permutation]
    weights["model.layers." + str(i) + ".mlp.down_proj.weight"] = weights[
        "model.layers." + str(i) + ".mlp.down_proj.weight"
    ][:, mlp_permutation]

    weights["model.layers." + str(i) + ".mlp.gate_proj.weight"] = weights[
        "model.layers." + str(i) + ".mlp.gate_proj.weight"
    ][:, emb_permutation]
    weights["model.layers." + str(i) + ".mlp.up_proj.weight"] = weights[
        "model.layers." + str(i) + ".mlp.up_proj.weight"
    ][:, emb_permutation]
    weights["model.layers." + str(i) + ".mlp.down_proj.weight"] = weights[
        "model.layers." + str(i) + ".mlp.down_proj.weight"
    ][emb_permutation]

    tmp_model.load_state_dict(weights)


def permute_embedding_layer(model, tmp_model, emb_permutation):
    weights = model.state_dict()

    weights["model.embed_tokens.weight"] = weights["model.embed_tokens.weight"][:, emb_permutation]
    tmp_model.load_state_dict(weights)


def permute_output_layer(model, tmp_model, emb_permutation):
    weights = model.state_dict()

    weights["lm_head.weight"] = weights["lm_head.weight"][:, emb_permutation]
    tmp_model.load_state_dict(weights)


def permute_mlp_block(model, i, tmp_model, mlp_permutation):
    weights = model.state_dict()

    weights["model.layers." + str(i) + ".mlp.gate_proj.weight"] = weights[
        "model.layers." + str(i) + ".mlp.gate_proj.weight"
    ][mlp_permutation]
    weights["model.layers." + str(i) + ".mlp.up_proj.weight"] = weights[
        "model.layers." + str(i) + ".mlp.up_proj.weight"
    ][mlp_permutation]
    weights["model.layers." + str(i) + ".mlp.down_proj.weight"] = weights[
        "model.layers." + str(i) + ".mlp.down_proj.weight"
    ][:, mlp_permutation]

    tmp_model.load_state_dict(weights)


def avg_mlp_block(model0, model1, i, tmp_model, alpha=0.5):
    weights0 = model0.state_dict()
    weights1 = model1.state_dict()

    weights0["model.layers." + str(i) + ".mlp.gate_proj.weight"] = (
        alpha * weights0["model.layers." + str(i) + ".mlp.gate_proj.weight"]
        + (1 - alpha) * weights1["model.layers." + str(i) + ".mlp.gate_proj.weight"]
    )
    weights0["model.layers." + str(i) + ".mlp.up_proj.weight"] = (
        alpha * weights0["model.layers." + str(i) + ".mlp.up_proj.weight"]
        + (1 - alpha) * weights1["model.layers." + str(i) + ".mlp.up_proj.weight"]
    )
    weights0["model.layers." + str(i) + ".mlp.down_proj.weight"] = (
        alpha * weights0["model.layers." + str(i) + ".mlp.down_proj.weight"]
        + (1 - alpha) * weights1["model.layers." + str(i) + ".mlp.down_proj.weight"]
    )

    tmp_model.load_state_dict(weights0)


def avg_transformer_block(model0, model1, i, tmp_model, alpha=0.5, attn=True):
    weights0 = model0.state_dict()
    weights1 = model1.state_dict()

    if attn is True:
        weights0["model.layers." + str(i) + ".self_attn.q_proj.weight"] = (
            alpha * weights0["model.layers." + str(i) + ".self_attn.q_proj.weight"]
            + (1 - alpha) * weights1["model.layers." + str(i) + ".self_attn.q_proj.weight"]
        )
        weights0["model.layers." + str(i) + ".self_attn.k_proj.weight"] = (
            alpha * weights0["model.layers." + str(i) + ".self_attn.k_proj.weight"]
            + (1 - alpha) * weights1["model.layers." + str(i) + ".self_attn.k_proj.weight"]
        )
        weights0["model.layers." + str(i) + ".self_attn.v_proj.weight"] = (
            alpha * weights0["model.layers." + str(i) + ".self_attn.v_proj.weight"]
            + (1 - alpha) * weights1["model.layers." + str(i) + ".self_attn.v_proj.weight"]
        )
        weights0["model.layers." + str(i) + ".self_attn.o_proj.weight"] = (
            alpha * weights0["model.layers." + str(i) + ".self_attn.o_proj.weight"]
            + (1 - alpha) * weights1["model.layers." + str(i) + ".self_attn.o_proj.weight"]
        )

    weights0["model.layers." + str(i) + ".mlp.gate_proj.weight"] = (
        alpha * weights0["model.layers." + str(i) + ".mlp.gate_proj.weight"]
        + (1 - alpha) * weights1["model.layers." + str(i) + ".mlp.gate_proj.weight"]
    )
    weights0["model.layers." + str(i) + ".mlp.up_proj.weight"] = (
        alpha * weights0["model.layers." + str(i) + ".mlp.up_proj.weight"]
        + (1 - alpha) * weights1["model.layers." + str(i) + ".mlp.up_proj.weight"]
    )
    weights0["model.layers." + str(i) + ".mlp.down_proj.weight"] = (
        alpha * weights0["model.layers." + str(i) + ".mlp.down_proj.weight"]
        + (1 - alpha) * weights1["model.layers." + str(i) + ".mlp.down_proj.weight"]
    )

    tmp_model.load_state_dict(weights0)


def avg_embedding_layer(model0, model1, tmp_model, alpha=0.5):
    weights0 = model0.state_dict()
    weights1 = model1.state_dict()

    weights0["model.embed_tokens.weight"] = (
        alpha * weights0["model.embed_tokens.weight"]
        + (1 - alpha) * weights1["model.embed_tokens.weight"]
    )

    tmp_model.load_state_dict(weights0)


def avg_output_layer(model0, model1, tmp_model, alpha=0.5):
    weights0 = model0.state_dict()
    weights1 = model1.state_dict()

    weights0["lm_head.weight"] = (
        alpha * weights0["lm_head.weight"] + (1 - alpha) * weights1["lm_head.weight"]
    )
    weights0["model.norm.weight"] = (
        alpha * weights0["model.norm.weight"] + (1 - alpha) * weights1["model.norm.weight"]
    )

    tmp_model.load_state_dict(weights0)


def avg_model(model0, model1, tmp_model, alpha=0.5, n_blocks=32, attn=True, emb=True):
    model1 = copy.deepcopy(model1)

    if emb is True:
        avg_embedding_layer(model0, model1, tmp_model, alpha=alpha)
    else:
        tmp_model.load_state_dict(model0.state_dict())
    for i in range(n_blocks):
        avg_transformer_block(tmp_model, model1, i, tmp_model, alpha=alpha, attn=attn)
    if emb is True:
        avg_output_layer(tmp_model, model1, tmp_model, alpha=alpha)


def get_mlp_weights(model, i):
    return model.state_dict()["model.layers." + str(i) + ".intermediate.dense.weight"]


def get_emb_weights(model):
    return model.state_dict()["model.embed_tokens.weight"]
