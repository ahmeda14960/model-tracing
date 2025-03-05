import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import itertools
import os
from datasets import load_dataset
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import csv
from utils import interpolate_models
import time
import argparse
import glob
import gc

block_size = 2048
"""
Script for running ablation of tests on m2d2 dataset rather
than simply wikitext
"""


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


def load_model(model_name):
    return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_arch = args.model_arch
    if model_arch == "llama":
        model_list = [
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/CodeLlama-7b-Python-hf",
            "meta-llama/CodeLlama-7b-Instruct-hf",
            "codellama/CodeLlama-7b-hf",
            "lmsys/vicuna-7b-v1.5",
            "lmsys/vicuna-7b-v1.1",
            "EleutherAI/llemma_7b",
            "LLM360/Amber",
        ]
    elif model_arch == "olmo":
        model_list = [
            "/scr/ahmedah/olmo/step1000_4B_tokens/seed_0_4B",
            "/scr/ahmedah/olmo/step1000_4B_tokens/seed_42_4B",
        ]

    tokenizer = AutoTokenizer.from_pretrained(model_list[0])
    tokenizer.pad_token = tokenizer.eos_token

    test_cases = [
        {
            "test_name": folder_name,
            "json_dir": f"/juice4/scr4/nlp/model-tracing/m2d2_s2orc/{folder_name}",
            "save_dir": f"/juice4/scr4/nlp/model-tracing/m2d2_s2orc/results_{folder_name}",
            "columns_ignored": ["text", "added", "id", "source", "timestamp", "subdomain"],
        }
        for folder_name in [
            "AI",
            "CV",
            "ET",
            "IM",
            "mtrl-sci",
            "stat-mech",
            "AR",
            "CY",
            "IR",
            "NA",
            "str-el",
            "art",
            "DB",
            "FL",
            "supr-con",
            "CC",
            "DC",
            "GA",
            "LG",
            "phil",
            "CE",
            "dis-nn",
            "GL",
            "LO",
            "CG",
            "DL",
            "GR",
            "MA",
            "quant-gas",
            "CL",
            "DM",
            "GT",
            "mes-hall",
            "CO",
            "DS",
            "HC",
            "MM",
            "soft",
            "CR",
            "EP",
            "HE",
            "MS",
            "SR",
        ]
    ]

    for test_case in test_cases:
        test_name = test_case["test_name"]
        json_dir = test_case["json_dir"]
        save_dir = test_case["save_dir"]
        columns_ignored = ["text", "added", "id", "source", "subdomain"]

        json_files = glob.glob(f"{json_dir}/*.json")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for json_file in json_files:
            print(f"Processing {json_file}")

            eval_dataset = load_dataset("json", data_files=json_file)

            def tokenize_function(examples):
                return tokenizer(examples["text"])

            tokenized_datasets = eval_dataset.map(
                tokenize_function, batched=True, num_proc=4, remove_columns=columns_ignored
            )
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                batch_size=1000,
                num_proc=8,
            )

            training_args = TrainingArguments(
                output_dir="./hf_results",
                per_device_eval_batch_size=15,
                do_eval=True,
                report_to=None,
                dataloader_num_workers=8,
                use_cpu=True,
            )
            alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
            initial_model = load_model(model_list[0])
            trainer = Trainer(model=initial_model, args=training_args, eval_dataset=lm_datasets)
            eval_dataloader = trainer.get_test_dataloader(lm_datasets["train"])
            del initial_model

            model_pairs = list(itertools.combinations(enumerate(model_list), 2))

            base_dir = f"{save_dir}/{test_name}"
            os.makedirs(base_dir, exist_ok=True)
            imgs_dir = os.path.join(base_dir, "imgs")
            os.makedirs(imgs_dir, exist_ok=True)
            csv_dir = os.path.join(base_dir, "csv")
            os.makedirs(csv_dir, exist_ok=True)

            current_model_a, current_model_b = None, None
            current_model_a_name, current_model_b_name = None, None

            for (idx_a, model_a_name), (idx_b, model_b_name) in tqdm(
                model_pairs, desc="Model Interpolation"
            ):
                if idx_a < idx_b:
                    perplexities = []

                    if current_model_a is None or current_model_a_name != model_a_name:
                        if current_model_a is not None:
                            del current_model_a
                            torch.cuda.empty_cache()
                        current_model_a = load_model(model_a_name).to("cpu")
                        current_model_a_name = model_a_name

                    if current_model_b is None or current_model_b_name != model_b_name:
                        if current_model_b is not None:
                            del current_model_b
                            torch.cuda.empty_cache()
                        current_model_b = load_model(model_b_name).to("cpu")
                        current_model_b_name = model_b_name

                    with torch.no_grad():
                        for alpha in tqdm(
                            alphas,
                            desc=f" \n Alpha Perplexities for {model_a_name} and {model_b_name}",
                        ):
                            interpolated_model = interpolate_models(
                                current_model_a, current_model_b, alpha, model_arch=model_arch
                            )
                            interpolated_model = interpolated_model.half().to(device)

                            start_time = time.time()
                            losses = []

                            for batch in tqdm(eval_dataloader, desc=f"\n Evaluating {alpha}"):
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

                            interpolated_model.to("cpu")
                            del interpolated_model, input_ids, attention_mask, labels, outputs, loss
                            torch.cuda.empty_cache()
                            gc.collect()

                    model_a_name = model_a_name.split("/")[-1]
                    model_b_name = model_b_name.split("/")[-1]
                    json_filename = os.path.splitext(os.path.basename(json_file))[0]
                    csv_filename = f"{csv_dir}/perplexities_{json_filename}.csv"
                    csv_header = ["Model Pair"] + [f"Alpha {alpha}" for alpha in alphas]

                    if not os.path.exists(csv_filename):
                        with open(csv_filename, "w", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(csv_header)

                    with open(csv_filename, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        model_pair = f"{model_a_name} vs {model_b_name}"
                        row = [model_pair] + perplexities
                        writer.writerow(row)

                    plt.figure(figsize=(8, 6))
                    plt.plot(alphas, perplexities)
                    plt.xlabel("Alpha")
                    plt.ylabel("Perplexity")
                    plt.title(f"{model_a_name} (Left) vs {model_b_name} (Right)")

                    plot_filename = (
                        f"alpha_vs_perplexity_{model_a_name}_vs_{model_b_name}_{json_filename}.png"
                    )
                    plot_path = f"{imgs_dir}/{plot_filename}"
                    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Interpolation")
    parser.add_argument(
        "--model_arch",
        choices=["llama", "olmo"],
        default="llama",
        help="default model architecture to use",
    )
    args = parser.parse_args()
    main(args)
