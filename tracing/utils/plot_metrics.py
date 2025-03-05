import yaml
from yaml import Loader

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
from scipy.stats import chi2

base = {
    "lmsys/vicuna-7b-v1.5": 1,
    "codellama/CodeLlama-7b-hf": 1,
    "codellama/CodeLlama-7b-Python-hf": 1,
    "codellama/CodeLlama-7b-Instruct-hf": 1,
    "EleutherAI/llemma_7b": 1,
    "microsoft/Orca-2-7b": 1,
    "oh-yeontaek/llama-2-7B-LoRA-assemble": 1,
    "lvkaokao/llama2-7b-hf-instruction-lora": 1,
    "NousResearch/Nous-Hermes-llama-2-7b": 1,
    "lmsys/vicuna-7b-v1.1": 0,
    "yahma/llama-7b-hf": 0,
    "Salesforce/xgen-7b-4k-base": 2,
    "EleutherAI/llemma_7b_muinstruct_camelmath": 1,
    "AlfredPros/CodeLlama-7b-Instruct-Solidity": 1,
    "meta-llama/Llama-2-7b-hf": 1,
    "LLM360/Amber": 3,
    "LLM360/AmberChat": 3,
    "openlm-research/open_llama_7b": 4,
    "openlm-research/open_llama_7b_v2": 5,
    "ibm-granite/granite-7b-base": 6,
    "ibm-granite/granite-7b-instruct": 6,
}

base_ordered = {
    "yahma/llama-7b-hf": 0,
    "lmsys/vicuna-7b-v1.1": 0,
    "meta-llama/Llama-2-7b-hf": 1,
    "lmsys/vicuna-7b-v1.5": 1,
    "codellama/CodeLlama-7b-hf": 1,
    "codellama/CodeLlama-7b-Python-hf": 1,
    "codellama/CodeLlama-7b-Instruct-hf": 1,
    "AlfredPros/CodeLlama-7b-Instruct-Solidity": 1,
    "EleutherAI/llemma_7b": 1,
    "EleutherAI/llemma_7b_muinstruct_camelmath": 1,
    "microsoft/Orca-2-7b": 1,
    "oh-yeontaek/llama-2-7B-LoRA-assemble": 1,
    "lvkaokao/llama2-7b-hf-instruction-lora": 1,
    "NousResearch/Nous-Hermes-llama-2-7b": 1,
    "Salesforce/xgen-7b-4k-base": 2,
    "LLM360/Amber": 3,
    "LLM360/AmberChat": 3,
    "openlm-research/open_llama_7b": 4,
    "openlm-research/open_llama_7b_v2": 5,
    "ibm-granite/granite-7b-base": 6,
    "ibm-granite/granite-7b-instruct": 6,
}

tree = {
    "yahma/llama-7b-hf": "A---",
    "lmsys/vicuna-7b-v1.1": "AA--",
    "meta-llama/Llama-2-7b-hf": "B---",
    "lmsys/vicuna-7b-v1.5": "BA--",
    "codellama/CodeLlama-7b-hf": "BB--",
    "codellama/CodeLlama-7b-Python-hf": "BBA-",
    "codellama/CodeLlama-7b-Instruct-hf": "BBB-",
    "AlfredPros/CodeLlama-7b-Instruct-Solidity": "BBBA",
    "EleutherAI/llemma_7b": "BBC-",
    "EleutherAI/llemma_7b_muinstruct_camelmath": "BBCA",
    "microsoft/Orca-2-7b": "BC--",
    "oh-yeontaek/llama-2-7B-LoRA-assemble": "BD--",
    "lvkaokao/llama2-7b-hf-instruction-lora": "BE--",
    "NousResearch/Nous-Hermes-llama-2-7b": "BF--",
    "Salesforce/xgen-7b-4k-base": "C---",
    "LLM360/Amber": "D---",
    "LLM360/AmberChat": "DA--",
    "openlm-research/open_llama_7b": "E---",
    "openlm-research/open_llama_7b_v2": "F---",
    "ibm-granite/granite-7b-base": "G---",
    "ibm-granite/granite-7b-instruct": "GA--",
}


def get_dict_ft(flat_model_path):
    dict_ft = {}

    model_paths = yaml.load(open(flat_model_path, "r"), Loader=Loader)

    for i in range(len(model_paths)):
        for j in range(i + 1, len(model_paths)):
            model_a = model_paths[i]
            model_b = model_paths[j]

            job_id = model_a.replace("/", "-") + "_AND_" + model_b.replace("/", "-")

            dict_ft[job_id] = base[model_a] == base[model_b]

    return dict_ft


def get_statistic_from_file(filename):

    file = open(filename, "r")

    lines = file.readlines()
    stat = np.nan

    for line in lines:
        if "Namespace" in line and "non-aligned test stat" in line:
            # dict = json.loads(line)
            # print(dict)
            # print(dict['non-aligned test stat'])

            start1 = line.find("non-aligned test stat")
            stat = line[line.find(":", start1) : line.find(",", start1)]
            stat = stat.replace(" ", "")
            stat = stat.replace("(", "")
            stat = stat.replace(":", "")
            stat = stat.replace("tensor", "")
            stat = float(stat)

    return stat


def get_l2_stat_from_file(filename):
    file = open(filename, "r")

    lines = file.readlines()
    stats = []

    for line in lines:
        if len(line) >= 4 and line[4] == " ":
            stats.append(line[:4])

    return float(stats[-1])


def get_layer_statistic_from_file(filename, layer):
    file = open(filename, "r")

    lines = file.readlines()
    stat = np.nan

    for line in lines:
        temp = str(layer) + " "
        if line[: len(temp)] == temp:
            stat = line[line.find("0.") :]
            if "e" in line:
                stat = 0
            # print(layer, stat)
            stat = float(stat)

    return stat


def plot_statistic_scatter(results_path, dict_ft, plot_path):

    x = []
    y = []

    dir_list = os.listdir(results_path)
    for file in dir_list:
        models = file[: file.find(".out")]
        if "huggyllama" in models:
            continue
        print(models)
        ft = int(dict_ft[models])
        stat = get_l2_stat_from_file(results_path + "/" + file)
        # stat = get_statistic_from_file(results_path + '/' + file)
        if not np.isnan(stat):
            y.append(ft)
            # x.append(get_statistic_from_file(results_path + '/' + file))
            x.append(get_l2_stat_from_file(results_path + "/" + file))

    plt.figure(figsize=(10, 1))

    plt.scatter(x, y, s=8)

    plt.xlabel("$p$-value")
    plt.ylabel("Fine-tuned")
    plt.ylim(-0.5, 1.5)
    # plt.title(f"{}")
    plot_filename = f"{plot_path}.png"

    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_statistic_grid(results_path, dict_base, title, plot_path, decimals, log=False):
    models = list(dict_base.keys())
    print(models)

    data = np.full((len(models), len(models)), np.nan)

    for i in range(len(models)):
        for j in range(len(models)):
            model_a = models[i]
            model_b = models[j]

            job_id = model_a.replace("/", "-") + "_AND_" + model_b.replace("/", "-") + ".out"

            if not os.path.exists(results_path + "/" + job_id):
                continue
            print(job_id)

            stat = get_statistic_from_file(results_path + "/" + job_id)
            # stat = get_l2_stat_from_file(results_path + '/' + job_id)

            if log:
                stat = np.log(stat)

            data[i][j] = np.round(stat, decimals=decimals)
            data[j][i] = data[i][j]

    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)
    im = ax.imshow(data, cmap="viridis")

    _ = make_axes_locatable(ax)
    _ = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(models)), labels=models)
    ax.set_yticks(np.arange(len(models)), labels=models)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    texts = []
    for i in range(len(models)):
        text1 = []
        for j in range(len(models)):
            text1.append("")
        texts.append(text1)

    # Loop over data dimensions and create text annotations.
    for i in range(len(models)):
        for j in range(len(models)):
            texts[i][j] = str(data[i][j])
            if data[i][j] == 0.0:
                texts[i][j] = "$\\varepsilon$"
            _ = ax.text(j, i, texts[i][j], ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    plot_filename = f"{plot_path}.png"

    plt.savefig(plot_filename, dpi=500, bbox_inches="tight")
    plt.close()


def plot_statistic_scatter_layer(results_path, dict_ft, plot_path, layer):

    x = []
    y = []

    dir_list = os.listdir(results_path)
    for file in dir_list:
        models = file[: file.find(".out")]
        if "huggyllama" in models:
            continue
        print(models)
        ft = int(dict_ft[models])
        stat = get_layer_statistic_from_file(results_path + "/" + file, layer)
        if not np.isnan(stat):
            x.append(ft)
            y.append(get_layer_statistic_from_file(results_path + "/" + file, layer))

    plt.figure(figsize=(8, 6))

    plt.scatter(x, y, s=2)

    plt.xlabel("Fine tuned")
    plt.ylabel("Test statistic")
    # plt.title(f"{}")
    plot_filename = f"{plot_path}.png"

    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_statistic_grid_layer(
    results_path, dict_base, title, plot_path, decimals, layer, log=False
):
    models = list(dict_base.keys())
    print(models)

    data = np.full((len(models), len(models)), np.nan)

    for i in range(len(models)):
        for j in range(len(models)):
            model_a = models[i]
            model_b = models[j]

            job_id = model_a.replace("/", "-") + "_AND_" + model_b.replace("/", "-") + ".out"

            if not os.path.exists(results_path + "/" + job_id):
                continue

            stat = get_layer_statistic_from_file(results_path + "/" + job_id, layer)

            if log:
                stat = np.log(stat)

            data[i][j] = np.round(stat, decimals=decimals)
            data[j][i] = data[i][j]

    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)
    im = ax.imshow(data)

    _ = make_axes_locatable(ax)
    _ = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(models)), labels=models)
    ax.set_yticks(np.arange(len(models)), labels=models)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(models)):
        for j in range(len(models)):
            _ = ax.text(j, i, data[i, j], ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    plot_filename = f"{plot_path}.png"

    plt.savefig(plot_filename, dpi=500, bbox_inches="tight")
    plt.close()


def plot_histogram(results_path, dict_ft, plot_path):
    indp = []
    not_indp = []

    dir_list = os.listdir(results_path)
    for file in dir_list:
        models = file[: file.find(".out")]
        print(models)
        ft = int(dict_ft[models])
        stat = get_statistic_from_file(results_path + "/" + file)
        if not np.isnan(stat):
            if ft:
                not_indp.append(stat)
            else:
                indp.append(stat)

    plt.figure(figsize=(8, 6))

    plt.hist(indp, bins=20, range=(0, 1), color="blue")
    plt.hist(not_indp, bins=20, range=(0, 1), color="green")

    plt.xlabel("Test statistic value")
    plt.ylabel("Count")
    # plt.title(f"{}")
    plot_filename = f"{plot_path}.png"

    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()


def fisher(pvalues):
    chi_squared = 0
    num_layers = 0
    for pvalue in pvalues:
        if not np.isnan(pvalue):
            chi_squared -= 2 * np.log(pvalue)
            num_layers += 1

    return chi2.sf(chi_squared, df=2 * num_layers)


def plot_statistic_scatter_all_layers(results_path, dict_ft, plot_path):

    x = []
    y = []
    c = []

    dir_list = os.listdir(results_path)

    for layer in range(32):

        for file in dir_list:
            models = file[: file.find(".out")]
            # if("huggyllama" in models): continue
            print(models)
            ft = int(dict_ft[models])
            stat = get_layer_statistic_from_file(results_path + "/" + file, layer)
            if not np.isnan(stat):
                x.append(layer)
                y.append(get_layer_statistic_from_file(results_path + "/" + file, layer))
                if ft:
                    c.append("r")
                else:
                    c.append("b")

    for file in dir_list:
        models = file[: file.find(".out")]
        # if("huggyllama" in models): continue
        ft = int(dict_ft[models])
        stat = get_layer_statistic_from_file(results_path + "/" + file, layer)
        if not np.isnan(stat):
            x.append(layer)
            y.append(get_layer_statistic_from_file(results_path + "/" + file, layer))
            if ft:
                c.append("r")
            else:
                c.append("b")

    plt.figure(figsize=(8, 6))

    plt.scatter(x, y, s=1.5, c=c)

    plt.xlabel("Layer")
    plt.ylabel("p-value")
    # plt.title(f"{}")
    plot_filename = f"{plot_path}.png"

    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pvalue(results_path, dict_ft, plot_path):

    pvalues = []

    dir_list = os.listdir(results_path)

    for layer in range(32):

        for file in dir_list:
            models = file[: file.find(".out")]
            if "huggyllama" in models:
                continue
            print(models)
            ft = int(dict_ft[models])
            if ft is True:
                continue
            stat = get_layer_statistic_from_file(results_path + "/" + file, layer)
            if not np.isnan(stat):
                pvalues.append(stat)
    x = np.arange(0, 1, step=0.001)
    y = []
    print(pvalues)
    print(len(pvalues))

    for i in x:
        counter = 0
        for val in pvalues:
            if val < i:
                counter += 1
        y.append(counter / len(pvalues))

    plt.figure(figsize=(8, 6))

    plt.plot(x, y, ".-")

    # plt.xlabel("Fine tuned")
    # plt.ylabel("Test statistic")
    # plt.title(f"{}")
    # plt.xlim(-10,0)
    # plt.ylim(-10,0)
    plot_filename = f"{plot_path}.png"

    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    dict_ft = get_dict_ft("/nlp/u/salzhu/model-tracing/config/llama_flat.yaml")

    # plot_statistic_scatter("/juice4/scr4/nlp/model-tracing/llama_models_runs/perm_mc_l2_wikitext/logs",
    #                dict_ft, "test_statistic_plots/l2_pvalue_horizontal")

    # plot_statistic_grid("/juice4/scr4/nlp/model-tracing/mlp_match_rand_rot_perm_lap/logs",
    #                     base_ordered, "MLP up/gate matching p-value on permuted model pairs (random inputs for matching)",
    #                     "/nlp/u/salzhu/test_statistic_tables/mlp_match_rand_rot_perm_lap",
    #                     3, log=False)

    # plot_statistic_grid("/juice4/scr4/nlp/model-tracing/csh_0928_reruns/logs",
    #                     base_ordered, "",
    #                     "/nlp/u/salzhu/csh_0929_cols",
    #                     3, log=False)

    # plot_statistic_grid("/juice4/scr4/nlp/model-tracing/mlp_match_rand_rot_perm_lap/logs",
    #                     base_ordered, "",
    #                     "/nlp/u/salzhu/robust_0929",
    #                     3, log=False)

    # plot_statistic_grid("/juice4/scr4/nlp/model-tracing/llama_models_runs/perm_mc_l2_wikitext/logs",
    #                     base_ordered, "",
    #                     "/nlp/u/salzhu/l2_0927",
    #                     3, log=False)

    # plot_statistic_scatter("/juice4/scr4/nlp/model-tracing/mlp_match_rand_rot_perm_lap/logs", dict_ft,
    #                        "/nlp/u/salzhu/test_statistic_plots/mlp_sp_final")

    # plot_statistic_scatter_layer("/juice4/scr4/nlp/model-tracing/mlp_match_rand_rot_perm_lap/logs", dict_ft,
    #                        "/nlp/u/salzhu/test_statistic_plots/mlp_match_rand_rot_perm_lap_layer31", 31)

    # plot_statistic_grid_layer("/juice4/scr4/nlp/model-tracing/mlp_match_rand_rot_perm_lap/logs",
    #                     base_ordered, "MLP up/gate matching p-value on permuted model pairs (random inputs for matching)",
    #                     "/nlp/u/salzhu/test_statistic_tables/mlp_match_rand_rot_perm_lap_layer31",
    #                     3, 31, log=False)

    # plot_statistic_scatter_all_layers("/juice4/scr4/nlp/model-tracing/mlp_match_rand_rot_perm_lap/logs",
    #                     dict_ft,
    #                     "/nlp/u/salzhu/test_statistic_plots/mlp_match_rand_rot_perm_lap_all_layers")

    plot_pvalue(
        "/juice4/scr4/nlp/model-tracing/mlp_match_rand_rot_perm_lap/logs",
        dict_ft,
        "/nlp/u/salzhu/test_statistic_plots/mlp_sp_final",
    )

    # plot_histogram("/juice4/scr4/nlp/model-tracing/mlp_match_med_max_layer0/logs",
    #                dict_ft, "/nlp/u/salzhu/test_statistic_plots/mlp_med_max_histogram")

    # checkpoints = {
    #     "100M": 1e8,
    #     "1B": 1e9,
    #     "10B": 1e10,
    #     "18B": 1.8e10,
    # }

    # checkpoints = {
    #     "100M": 1e8,
    #     "1B": 1e9,
    #     "4B": 4e9,
    #     "8B": 8e9,
    #     "16B": 1.6e10
    # }

    # checkpoints = {
    #     "100M": 1e8,
    #     "1B": 1e9,
    #     "12B": 1.2e10,
    #     "25B": 2.5e10
    # }

    # plot_statistic_olmo_scatter("/juice4/scr4/nlp/model-tracing/olmo_models_runs/final_checkpoint/csw_robust_cols/logs", checkpoints,
    #                             "final checkpoint vs. additional training seed 42", "CSW robust",
    #                             "/nlp/u/salzhu/olmo_plots/final_checkpoint/csw_robust_cols_seed42")
