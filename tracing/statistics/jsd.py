"""
Implementation of Jensen-Shannon Divergence (JSD) for comparing language model outputs.

This module provides functions to compute the Jensen-Shannon Divergence between
probability distributions output by two language models, measuring their similarity
in output space rather than parameter space.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from tracing.utils.evaluate import (
    prepare_hf_dataset,
    prepare_hf_dataloader,
)


def statistic(base_model, ft_model, dataloader, device="cuda"):
    """
    Compute Jensen-Shannon Divergence between outputs of two language models.

    Args:
        base_model: Base model to compare
        ft_model: Fine-tuned or target model to compare against the base model
        dataloader: DataLoader providing input data for model evaluation
        device: Device to run the computation on (default: "cuda")

    Returns:
        float: Sum of Jensen-Shannon Divergence values across all batches
    """
    return compute_jsd(base_model, ft_model, dataloader, device)


def statistic_stable(base_model, ft_model, dataloader, device="cuda"):
    """
    Compute numerically stable Jensen-Shannon Divergence between outputs of two models.

    This version handles potential numerical issues better than the standard version.

    Args:
        base_model: Base model to compare
        ft_model: Fine-tuned or target model to compare against the base model
        dataloader: DataLoader providing input data for model evaluation
        device: Device to run the computation on (default: "cuda")

    Returns:
        float: Sum of Jensen-Shannon Divergence values across all batches
    """
    return compute_jsd_stable(base_model, ft_model, dataloader, device)


def compute_jsd(base_model, ft_model, dataloader, device="cuda"):
    """
    Compute Jensen-Shannon Divergence between two models using softmax outputs.

    Processes each batch in the dataloader and computes JSD between the models'
    probability distributions over vocabulary tokens. Handles potential vocabulary
    size differences by truncating to a common size (32000 tokens).

    Args:
        base_model: Base model to compare
        ft_model: Fine-tuned or target model to compare against the base model
        dataloader: DataLoader providing input data for model evaluation
        device: Device to run the computation on (default: "cuda")

    Returns:
        float: Sum of Jensen-Shannon Divergence values across all batches
    """
    jsds = []

    base_model.to(device)
    ft_model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs_base = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            outputs_ft = ft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            logits_base = outputs_base.logits.squeeze()
            logits_ft = outputs_ft.logits.squeeze()

            softmax_base = torch.softmax(logits_base, dim=-1)
            softmax_ft = torch.softmax(logits_ft, dim=-1)

            # Truncate the softmax outputs to the first 32000 dimensions
            softmax_base = softmax_base[:, :32000]
            softmax_ft = softmax_ft[:, :32000]

            m = 0.5 * (softmax_base + softmax_ft)
            jsd = 0.5 * (F.kl_div(m.log(), softmax_base) + F.kl_div(m.log(), softmax_ft))

            jsds.append(jsd.item())

    base_model.to("cpu")
    ft_model.to("cpu")
    return sum(jsds)


def compute_jsd_stable(base_model, ft_model, dataloader, device="cuda"):
    """
    Compute numerically stable Jensen-Shannon Divergence between two models.

    A more robust implementation that:
    1. Handles vocabulary size mismatches by truncating to the minimum size
    2. Uses log-space calculations to avoid numerical underflow
    3. Computes JSD directly from log probabilities for better stability

    Args:
        base_model: Base model to compare
        ft_model: Fine-tuned or target model to compare against the base model
        dataloader: DataLoader providing input data for model evaluation
        device: Device to run the computation on (default: "cuda")

    Returns:
        float: Sum of Jensen-Shannon Divergence values across all batches
    """
    jsds = []

    base_model.to(device)
    ft_model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs_base = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            outputs_ft = ft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            logits_base = outputs_base.logits.squeeze()
            logits_ft = outputs_ft.logits.squeeze()

            # Determine the minimum vocabulary size between the two models
            min_vocab_size = min(logits_base.size(-1), logits_ft.size(-1))

            # Truncate the logits to the minimum vocabulary size
            logits_base = logits_base[..., :min_vocab_size]
            logits_ft = logits_ft[..., :min_vocab_size]

            log_probs_base = F.log_softmax(logits_base, dim=-1)
            log_probs_ft = F.log_softmax(logits_ft, dim=-1)

            m = 0.5 * (log_probs_base.exp() + log_probs_ft.exp())
            log_m = m.log()

            kl_div_base_m = (log_probs_base - log_m).sum(dim=-1)
            kl_div_ft_m = (log_probs_ft - log_m).sum(dim=-1)

            jsd = 0.5 * (kl_div_base_m + kl_div_ft_m).mean()
            jsds.append(jsd.item())

    base_model.to("cpu")
    ft_model.to("cpu")

    return sum(jsds)


if __name__ == "__main__":

    base_model_name = "LLM360/Amber"  # 'openlm-research/open_llama_7b' # 'lmsys/vicuna-7b-v1.5'
    ft_model_name = "LLM360/AmberChat"  # 'openlm-research/open_llama_7b_v2' # 'LLM360/Amber' # "lmsys/vicuna-7b-v1.1"

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    ft_model = AutoModelForCausalLM.from_pretrained(ft_model_name, torch_dtype=torch.bfloat16)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    # dataset = load_generated_datasets(base_model_name, ft_model_name, 512, base_tokenizer, ["text"])
    # dataloader = prepare_hf_dataloader(dataset, 1)

    dataset = prepare_hf_dataset("dlwh/wikitext_103_detokenized", 512, base_tokenizer)
    dataloader = prepare_hf_dataloader(dataset, 1)

    print(statistic(base_model, ft_model, dataloader))
