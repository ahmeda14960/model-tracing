
# LLM Model Tracing
This repository investigates model tracing in large language models (LLMs).

Specifically, given a base LLM and a fine-tuned LLM, this code provides functionality to:

- Permute the weights of one model (either MLP or embedding weights).
- Align the weights of the fine-tuned model to the base model using the Hungarian algorithm.
- Evaluate the effect of weight permutation and alignment on different statistics:
    - Mode connectivity
    - Cosine similarity
    - Embedding similarity
- Evaluate the perplexity of the base and fine-tuned models on a given dataset.

## Requirements

Install the necessary packages using:

```bash
pip install -r requirements.txt
```

For development, install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

### Code Formatting with pre-commit

This repository uses pre-commit hooks to ensure code quality and consistency.

1. Install pre-commit:

```bash
pip install pre-commit
```

2. Set up the pre-commit hooks:

```bash
pre-commit install
```

3. (Optional) Run pre-commit on all files:

```bash
pre-commit run --all-files
```

Pre-commit will automatically run on staged files when you commit changes, applying:
- Black for code formatting
- Ruff for linting and fixing common issues
- nbQA for notebook formatting
- Various file checks (trailing whitespace, YAML validity, etc.)

## Usage

The repository provides three main scripts:

- `main.py`: Executes the main experiment pipeline for model tracing.
- `launch.py`: Launches multiple experiments in parallel using slurm.

### `main.py`

This script performs the following steps:

1. Loads the base and fine-tuned LLMs.
2. Optionally permutes the weights of the fine-tuned model.
3. Calculates the selected statistic for the non-aligned models.
4. Optionally aligns the weights of the fine-tuned model to the base model.
5. Calculates the selected statistic for the aligned models.
6. Optionally evaluates the perplexity of the base and fine-tuned models.
7. Saves the results to a pickle file.

The script accepts various command-line arguments:

- `--base_model_id`: HuggingFace model ID for the base model.
- `--ft_model_id`: HuggingFace model ID for the fine-tuned model.
- `--permute`: Whether to permute the weights of the fine-tuned model.
- `--align`: Whether to align the weights of the fine-tuned model to the base model.
- `--dataset_id`: HuggingFace dataset ID for perplexity evaluation.
- `--stat`: Statistic to calculate (options: "mode", "cos", "emb").
  - csu: cosine similarity of weights statistic (on MLP up projection matrices) w/ Spearman correlation
    - csu_all: csu on all pairs of parameters with equal shape
  -  csh: cosine similarity of MLP activations statistic w/ Spearman correlation
  -  match: unconstrained statistic (match) with permutation matching of MLP activations
     - match_all: unconstrained statistic (match) on all pairs of MLP block activations
- `--attn`: Whether to consider attention weights in the "mode" statistic.
- `--emb`: Whether to consider embedding weights in the "mode" statistic.
- `--eval`: Whether to evaluate perplexity.
- `--save`: Path to save the results pickle file.

Example usage:

```bash
python main.py --base_model_id meta-llama/Llama-2-7b-hf --ft_model_id lmsys/vicuna-7b-v1.5 --stat csu --save results.p
```

```bash
python main.py --base_model_id meta-llama/Llama-2-7b-hf --ft_model_id lmsys/vicuna-7b-v1.5 --permute --align --dataset wikitext --stat match --attn --save results.p
```

### `launch.py`

This script launches multiple experiments in parallel using slurm. It reads model IDs from a YAML file and runs `main.py` for each pair of base and fine-tuned models. Use the flag --flat all (defaulted) to run on all pairs of models from a YAML (see config/llama7b.yaml); or, --flat split to run on all pairs of a 'base' model with a 'finetuned' model (see config/llama7b_split.yaml); or --flat specified to run on a specified list of pairs of models.

## Configuration

The `model-tracing/config/model_list.yaml` file defines the base and fine-tuned models for the experiments.
## Data

The code downloads and uses the Wikitext 103 dataset for perplexity evaluation.

## Results

The results of the experiments are saved as pickle files. The files contain dictionaries with the following keys:

- `args`: Command-line arguments used for the experiment.
- `commit`: Git commit hash of the code used for the experiment.
- `non-aligned test stat`: Value of the selected statistic for the non-aligned models.
- `aligned test stat`: Value of the selected statistic for the aligned models (if `--align` is True).
- `base loss`: Perplexity of the base model on the evaluation dataset (if `--eval` is True).
- `ft loss`: Perplexity of the fine-tuned model on the evaluation dataset (if `--eval` is True).
- `time`: Total execution time of the experiment.

## Sample commands

### 70B runs
```
 python main.py --base_model_id meta-llama/Llama-2-70b-hf --ft_model_id meta-llama/Meta-Llama-3-70B --stat csu
```

# Experiments

Relevant scripts for running additional experiments described in our paper are in this folder. For example, there are experiments on retraining MLP blocks and evaluating our statistics.

These include `experiments/localized_testing.py` (Section 3.2.1) for fine-grained forensics and layer-matching between two models; `experiments/csu_full.py` (Section 3.2.1) for full parameter-matching between any two model architectures for hybrid models; `experiments/generalized_match.py` (Section 2.3.2, 3.2.3, 3.2.4) for the generalized robust test that involes retraining or distilling GLU MLPs; and `experiments/huref.py` (Appendix F) where we reproduce and break the invariants from a related work (Zeng et al. 2024).
