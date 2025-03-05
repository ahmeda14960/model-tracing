import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import os
from datasets import load_dataset
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import csv
from utils import interpolate_models
import time
import copy
import argparse
import glob


block_size = 512


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def main(args):
    start_time = time.time()
    # Automatically detect CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.environ["WANDB_MODE"] = "disabled"

    # Load models and tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model_list = [
        "meta-llama/Llama-2-7b-hf",
        "codellama/CodeLlama-7b-hf",
        "lmsys/vicuna-7b-v1.5",
        "EleutherAI/llemma_7b",
        "LLM360/Amber",
    ]
    model_pairs = [
        (0, 2),  # LLama2, Vicuna-1.5
        (0, 1),  # LLama2, CodeLlama
        (0, 3),  # LLama2, Lemma
        (1, 3),  # CodeLlama, Lemma
        (0, 4),  # LLama2, Amber
    ]
    models = [
        AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        for model_name in model_list
    ]
    tokenizer = AutoTokenizer.from_pretrained(models[0].config._name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Scan the directory for JSON files based on the test name argument
    columns_ignored = [
        "text",
        "added",
        "id",
        "lang",
        "metadata",
        "source",
        "timestamp",
        "subdomain",
    ]
    json_dir = f"/juice4/scr4/nlp/model-tracing/dolma_program_languages/json_files_{args.test_name}"
    json_files = glob.glob(f"{json_dir}/*.json")
    save_dir = f"/juice4/scr4/nlp/model-tracing/dolma_program_languages/results_{args.test_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for json_file in json_files:
        print(f"Processing {json_file}")

        # Prepare dataset
        eval_dataset = load_dataset("json", data_files=json_file)

        def tokenize_function(examples):
            return tokenizer(examples["text"])

        tokenized_datasets = eval_dataset.map(
            tokenize_function, batched=True, num_proc=4, remove_columns=columns_ignored
        )
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=1,
            num_proc=1,
        )

        # Prepare for evaluation. Batch size is optimized for ~7B model
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_eval_batch_size=3,
            do_eval=True,
            report_to=None,
            dataloader_num_workers=4,
            use_cpu=True,
        )
        alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
        model = copy.deepcopy(models[0])
        trainer = Trainer(model=model, args=training_args, eval_dataset=lm_datasets)
        print("create data loader")
        eval_dataloader = trainer.get_test_dataloader(lm_datasets["train"])

        for idx_a, idx_b in tqdm(model_pairs, desc="Model Interpolation"):
            model_a = models[idx_a]
            model_b = models[idx_b]
            perplexities = []
            model_a_name = model_a.config._name_or_path.split("/")[-1]
            model_b_name = model_b.config._name_or_path.split("/")[-1]

            for alpha in tqdm(
                alphas, desc=f" \n Alpha Perplexities for {model_a_name} and {model_b_name}"
            ):
                interpolated_model = interpolate_models(model_a, model_b, alpha)
                # cast to bfloat16 before GPU
                interpolated_model = interpolated_model.half().to(device)

                start_time = time.time()
                losses = []

                for batch in tqdm(eval_dataloader, desc=f"\n Evaluating {alpha}"):
                    # HF Trainer finds GPU by default
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = interpolated_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                    losses.append(loss.item())

                loss_mean = sum(losses) / len(losses)
                print(f"Loss mean: {loss_mean}")
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Execution time base: {execution_time} seconds")

                perplexity = math.exp(loss_mean)
                perplexities.append(perplexity)

                # Move the model back to CPU
                interpolated_model.to("cpu")

                # Clear the GPU cache
                del interpolated_model, input_ids, attention_mask, labels, outputs, loss
                torch.cuda.empty_cache()

            # Save perplexities and model names to CSV
            json_filename = os.path.splitext(os.path.basename(json_file))[0]
            csv_filename = f"perplexities_{json_filename}.csv"
            csv_full_path = f"{save_dir}/{csv_filename}"
            csv_header = ["Model Pair"] + [f"Alpha {alpha}" for alpha in alphas]
            if not os.path.exists(csv_full_path):
                with open(csv_full_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csv_header)

            with open(csv_full_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                model_pair = f"{model_a_name} vs {model_b_name}"
                row = [model_pair] + perplexities
                writer.writerow(row)

            # Create the plot
            plt.figure(figsize=(8, 6))
            plt.plot(alphas, perplexities)
            plt.xlabel("Alpha")
            plt.ylabel("Perplexity")
            plt.title(f"{model_a_name} (Left) vs {model_b_name} (Right)")

            # Save the plot as a PNG file
            plot_filename = (
                f"alpha_vs_perplexity_{model_a_name}_vs_{model_b_name}_{json_filename}.png"
            )
            plot_full_path = f"{save_dir}/{plot_filename}"
            plt.savefig(plot_full_path, dpi=300, bbox_inches="tight")
            plt.close()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Interpolation")
    parser.add_argument(
        "--test_name", type=str, default="js", help="Test name (e.g., cpp, python, js)"
    )
    args = parser.parse_args()
    main(args)
