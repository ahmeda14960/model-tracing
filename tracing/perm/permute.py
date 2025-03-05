"""
Module for permuting weights in a Llama model architecture.
This enables exploring model connectivity and representation properties
by applying consistent neuron permutations throughout the network.
"""


def permute_model(model, mlp_permutation, emb_permutation, n_blocks=32):
    """
    Apply permutations to a Llama model's weights to maintain functional equivalence.

    Args:
        model: The Llama model to permute
        mlp_permutation: Permutation indices for MLP hidden dimensions
        emb_permutation: Permutation indices for embedding dimensions
        n_blocks: Number of transformer blocks in the model (default: 32)

    Returns:
        None: Modifies the model in-place
    """
    permute_embedding_layer(model, emb_permutation)
    permute_transformer_blocks(model, mlp_permutation, emb_permutation)
    permute_output_layer(model, emb_permutation)


def permute_transformer_blocks(model, mlp_permutation, emb_permutation):
    """
    Apply permutations to transformer block weights in a Llama model.

    Permutes attention layers, MLP layers, and normalization layers according to
    the provided permutation indices to maintain functional equivalence.

    Args:
        model: The Llama model to permute
        mlp_permutation: Permutation indices for MLP hidden dimensions
        emb_permutation: Permutation indices for embedding dimensions

    Returns:
        None: Modifies the model in-place
    """
    weights = model.state_dict()

    # Permuting the Self attention layers
    for key in weights:
        if "self_attn" not in key:
            continue

        if "o_proj" in key:
            weights[key] = weights[key][emb_permutation]
        else:
            weights[key] = weights[key][:, emb_permutation]

    # Permuting the mlp projection layers
    for key in weights:
        if "mlp" not in key:
            continue
        if len(weights[key].shape) != 2:
            continue

        dim_0 = weights[key].size(0)
        dim_1 = weights[key].size(1)

        if dim_0 == len(mlp_permutation):
            weights[key] = weights[key][mlp_permutation]
        elif dim_1 == len(mlp_permutation):
            weights[key] = weights[key][:, mlp_permutation]

        if dim_0 == len(emb_permutation):
            weights[key] = weights[key][emb_permutation]
        elif dim_1 == len(emb_permutation):
            weights[key] = weights[key][:, emb_permutation]

    # input_layernorm, post_attention_layernorm
    for key in weights:
        if "model.layers" not in key:
            continue
        if len(weights[key].shape) != 1 or len(weights[key]) != len(emb_permutation):
            continue

        weights[key] = weights[key][emb_permutation]

    model.load_state_dict(weights)


def permute_embedding_layer(model, emb_permutation):
    """
    Apply permutation to embedding layer weights in a Llama model.

    Args:
        model: The Llama model to permute
        emb_permutation: Permutation indices for embedding dimensions

    Returns:
        None: Modifies the model in-place
    """
    weights = model.state_dict()

    weights["model.embed_tokens.weight"] = weights["model.embed_tokens.weight"][:, emb_permutation]
    model.load_state_dict(weights)


def permute_output_layer(model, emb_permutation):
    """
    Apply permutation to output layer weights in a Llama model.

    Permutes the language model head and final normalization layer.

    Args:
        model: The Llama model to permute
        emb_permutation: Permutation indices for embedding dimensions

    Returns:
        None: Modifies the model in-place
    """
    weights = model.state_dict()

    weights["lm_head.weight"] = weights["lm_head.weight"][:, emb_permutation]
    weights["model.norm.weight"] = weights["model.norm.weight"][emb_permutation]
    model.load_state_dict(weights)
