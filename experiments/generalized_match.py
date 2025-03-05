"""
This file contains the code to do generalize \phi_MATCH.
Given two models, it first retrains the MLPs or other FFN of the models, then runs \phi_MATCH on the distilled MLPs.

May need to be modified depending on model architecture (this code was used for GPT-architecture).
"""

MLP_SIZE = 3072
MLP_SIZE_2 = 3072
EMB_SIZE = 768
EMB_SIZE_2 = 768
N_BLOCKS = 12

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    GPT2LMHeadModel,
    OpenAIGPTLMHeadModel,
)
import scipy
from collections import defaultdict

import numpy as np

from tracing.utils.evaluate import prepare_hf_dataset, prepare_hf_dataloader, evaluate
from tracing.utils.evaluate import (
    prepare_hf_dataset,
    prepare_hf_dataloader,
)

from tracing.utils.utils import manual_seed
from tracing.utils.llama.matching import match_wmats

manual_seed(0)


# architecture of MLP trained from scratch can be different from original
# eg, uncomment to get a 2-hidden layer MLP (original has just 1 hidden layer)
class CustomLlamaMLP(nn.Module):
    """
    Custom MLP module for Llama-style architecture with SwiGLU activation.

    This implementation allows for flexible architecture changes when training
    replacement MLPs for model distillation and analysis.

    Args:
        config: Model configuration containing embedding dimensions
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.intermediate_size = 4 * config.n_embd

        self.gate_proj1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj1 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = nn.SiLU()

    def forward(self, x):
        """
        Forward pass implementing SwiGLU activation function.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            torch.Tensor: Output tensor after MLP transformation
        """
        down_proj = self.down_proj1(self.act_fn(self.gate_proj1(x)) * self.up_proj1(x))
        return down_proj


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


def mlp_layers(base_model_gate, base_model_up, ft_model_gate, ft_model_up, dataloader, i, j):
    """
    Compare gate and up projections between separate model components.

    Tests whether separately trained gate and up projection models have
    consistent permutation patterns, which would indicate functionally
    corresponding neurons.

    Args:
        base_model_gate: First model with gate projection weights
        base_model_up: First model with up projection weights
        ft_model_gate: Second model with gate projection weights
        ft_model_up: Second model with up projection weights
        dataloader: DataLoader providing input data for activation collection
        i: Layer index in the first model
        j: Layer index in the second model

    Returns:
        float: Pearson correlation p-value between gate and up projection permutations
    """
    gate_match = mlp_matching(base_model_gate, ft_model_gate, dataloader, i, j)
    up_match = mlp_matching(base_model_up, ft_model_up, dataloader, i, j)

    print(gate_match, up_match, i, j)

    cor, pvalue = scipy.stats.pearsonr(gate_match.tolist(), up_match.tolist())
    return pvalue


def mlp_matching(base_model, ft_model, dataloader, i, j):
    """
    Match neurons between models by comparing activations.

    Collects activations from the feed-forward layer for both models
    and computes a permutation that would align corresponding neurons.

    Args:
        base_model: Base model to compare
        ft_model: Target model to compare against the base model
        dataloader: DataLoader providing input data for activation collection
        i: Layer index in the base model
        j: Layer index in the target model

    Returns:
        torch.Tensor: Permutation indices that match neurons between models
    """
    feats = defaultdict(list)

    base_hook = lambda *args: hook_out(*args, feats, "base")
    base_handle = base_model.transformer.h[i].mlp.c_fc.register_forward_hook(base_hook)

    ft_hook = lambda *args: hook_out(*args, feats, "ft")
    ft_handle = ft_model.transformer.h[i].mlp.c_fc.register_forward_hook(ft_hook)

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


def run(i):
    """
    Run the generalized MATCH algorithm to compare models with distilled components.

    This function:
    1. Loads two different GPT-2 models
    2. Trains custom MLPs to replicate the behavior of the original model MLPs
    3. Optionally applies random rotations to test invariance to representation changes
    4. Creates separate models for gate and up projections
    5. Compares the models using the MATCH algorithm

    The process demonstrates that functionally equivalent components can be identified
    even after representation changes, by examining activation patterns.

    Args:
        i: Layer index to focus on for the analysis

    Returns:
        None: Prints p-value results from the neuron matching
    """
    train_losses = []

    model_id_2 = "manupande21/GPT2_PMC"
    model_id_1 = "openai-community/gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_id_1, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_id_1, torch_dtype=torch.bfloat16)

    base_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    dataset_wikitext = prepare_hf_dataset("dlwh/wikitext_103_detokenized", 512, base_tokenizer)
    dataloader_wikitext = prepare_hf_dataloader(dataset_wikitext, 1)

    config = AutoConfig.from_pretrained(model_id_1)

    i = 0  # layer to retrain
    bsz = 5000  # batch size
    T = 10000  # gradient steps
    width_fac = 1.0  # easier to get loss down for wider MLPs when retraining

    # Train the first custom MLP
    mlp = CustomLlamaMLP(config).bfloat16()

    mlp.to("cuda")
    model.transformer.h[i].mlp.to("cuda")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
    print(f"Training MLP {model_id_1}")

    # Random rotation matrix to test invariance to representation changes
    A = torch.randn(size=(EMB_SIZE, EMB_SIZE), device="cuda").bfloat16() / np.sqrt(
        EMB_SIZE
    )  # rotate outputs (just for kicks / sanity check)

    # Distillation training loop for first model
    for t in range(T):
        X_batch = torch.randn(size=(bsz, EMB_SIZE), dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            Y_batch = model.transformer.h[i].mlp(X_batch)
            Y_batch = Y_batch @ A.T  # Apply rotation to outputs

        Y_h = mlp(X_batch)

        optimizer.zero_grad()
        loss = criterion(Y_h, Y_batch)

        loss.backward()
        optimizer.step()

        if t % 1000 == 0:
            print(f"train loss: {loss.item()}")
            train_losses.append(loss.item())

    # Create separate models for gate and up projections
    config = AutoConfig.from_pretrained(model_id_1)
    config.intermediate_size = int(width_fac * MLP_SIZE)

    model_retrained_1_gate = OpenAIGPTLMHeadModel(config).bfloat16()
    model_retrained_1_up = OpenAIGPTLMHeadModel(config).bfloat16()
    model.to("cpu")
    mlp.to("cpu")

    # Loading retrained weights to model_retrained
    weights_1_gate = model.state_dict()
    weights_1_up = model.state_dict()

    weights_1_gate[f"transformer.h.{i}.mlp.c_fc.weight"] = mlp.gate_proj1.weight.T
    weights_1_up[f"transformer.h.{i}.mlp.c_fc.weight"] = mlp.up_proj1.weight.T
    model_retrained_1_gate.load_state_dict(weights_1_gate)
    model_retrained_1_up.load_state_dict(weights_1_up)

    # Retraining / distilling second model
    model = AutoModelForCausalLM.from_pretrained(model_id_2, torch_dtype=torch.bfloat16)

    config = AutoConfig.from_pretrained(model_id_2)
    mlp = CustomLlamaMLP(config).bfloat16()

    mlp.to("cuda")
    model.transformer.h[i].mlp.to("cuda")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

    print(f"Training MLP {model_id_2}")

    # Different random rotation matrix for second model
    A = torch.randn(size=(EMB_SIZE_2, EMB_SIZE_2), device="cuda").bfloat16() / np.sqrt(
        EMB_SIZE_2
    )  # rotate outputs (just for kicks / sanity check)

    # Distillation training loop for second model
    for t in range(T):
        X_batch = torch.randn(size=(bsz, EMB_SIZE_2), dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            Y_batch = model.transformer.h[i].mlp(X_batch)
            Y_batch = Y_batch @ A.T  # Apply rotation to outputs

        Y_h = mlp(X_batch)

        optimizer.zero_grad()
        loss = criterion(Y_h, Y_batch)

        loss.backward()
        optimizer.step()

        if t % 1000 == 0:
            print(f"train loss: {loss.item()}")
            train_losses.append(loss.item())

    # Create separate models for gate and up projections
    config = AutoConfig.from_pretrained(model_id_2)
    config.intermediate_size = int(width_fac * MLP_SIZE_2)

    model_retrained_2_gate = GPT2LMHeadModel(config).bfloat16()
    model_retrained_2_up = GPT2LMHeadModel(config).bfloat16()
    model.to("cpu")
    mlp.to("cpu")

    weights_2_gate = model.state_dict()
    weights_2_up = model.state_dict()

    weights_2_gate[f"transformer.h.{i}.mlp.c_fc.weight"] = mlp.gate_proj1.weight.T
    weights_2_up[f"transformer.h.{i}.mlp.c_fc.weight"] = mlp.up_proj1.weight.T

    model_retrained_2_gate.load_state_dict(weights_2_gate)
    model_retrained_2_up.load_state_dict(weights_2_up)

    # Run MATCH algorithm to compare the models
    print(
        mlp_layers(
            model_retrained_1_gate,
            model_retrained_1_up,
            model_retrained_2_gate,
            model_retrained_2_up,
            dataloader,
            0,
            0,
        )
    )


if __name__ == "__main__":
    for i in range(0, 10):
        run(i)
