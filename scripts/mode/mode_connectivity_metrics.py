import pandas as pd
import matplotlib.pyplot as plt
import datetime


def plot_traces(
    results_path,
    metric,
    plot_path,
    model_a_name,
    model_b_name,
    unpermuted_res=False,
    normalize=True,
    alpha_step=0.1,
    end_points=True,
):

    df = pd.read_csv(results_path)

    alphas = [round(alpha * alpha_step, 2) for alpha in range(int(1 / alpha_step + 1))]
    if end_points is False:
        alphas = alphas[1:-1]

    if metric == "loss":

        plt.figure(figsize=(8, 6))
        for index, row in df.iterrows():
            row = row[int(len(row) - (len(row) - 2) / 2) :]
            if normalize:
                row = normalize_trace(row, alpha_step)
            plt.plot(alphas, row, "o-")

        plt.xlabel("Alpha")
        plt.ylabel("Loss")
        plt.title(f"{model_a_name} (Left) vs {model_b_name} (Right)")
        plot_filename = f"{plot_path}_{datetime.datetime.now().timestamp()}.png"

    if metric == "perplexity":

        plt.figure(figsize=(8, 6))
        for index, row in df.iterrows():
            row = row[2 : int(2 + (len(row) - 2) / 2)]
            if normalize:
                row = normalize_trace(row, alpha_step)
            plt.plot(alphas, row, "o-")

        plt.xlabel("Alpha")
        plt.ylabel("Perplexity")
        plt.title(f"{model_a_name} (Left) vs {model_b_name} (Right)")
        plot_filename = f"{plot_path}_{datetime.datetime.now().timestamp()}.png"

    if unpermuted_res is not False:
        plt.plot(alphas, normalize_trace(unpermuted_res, alpha_step))

    # Save the plot as a PNG file

    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_trace(losses, alpha_step, normalize, model_a_name, model_b_name, plot_path):

    plt.figure(figsize=(8, 6))
    if normalize:
        losses = normalize_trace(losses, alpha_step)
    alphas = [round(alpha * alpha_step, 2) for alpha in range(int(1 / alpha_step + 1))]
    plt.plot(alphas, losses, "o-")

    plt.xlabel("Alpha")
    plt.ylabel("Loss")
    plt.title(f"{model_a_name} (Left) vs {model_b_name} (Right)")
    plot_filename = f"{plot_path}_{datetime.datetime.now().timestamp()}.png"

    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()


def normalize_trace(trace, alpha_step):
    slope = trace[-1] - trace[0]
    start = trace[0]
    for i in range(len(trace)):
        trace[i] -= slope * alpha_step * i
        trace[i] -= start
    return trace


def normalize_trace_2(trace, alphas):
    slope = trace[-1] - trace[0]
    start = trace[0]
    for i in range(len(trace)):
        trace[i] -= slope * alphas[i]
        trace[i] -= start
    return trace


def max_loss_ahmed(results_path, num_points=5, normalize=True, alphas=[0.0, 0.3, 0.5, 0.7, 1.0]):
    df = pd.read_csv(results_path)

    max_losses = []

    for index, row in df.iterrows():
        row = row[-num_points:]
        if normalize:
            row = normalize_trace_2(row, alphas)
        max_losses.append(max(row))

    return max_losses


def max_loss_compare(results_path, unpermuted_loss, num_points, normalize=True, alpha_step=0.1):
    df = pd.read_csv(results_path)
    alphas = [round(alpha * alpha_step, 2) for alpha in range(int(1 / alpha_step + 1))]

    permuted_max_losses = []

    for index, row in df.iterrows():
        row = row[-num_points:]
        if normalize:
            row = normalize_trace(row, alpha_step)
        permuted_max_losses.append(max(row))

    if normalize:
        unpermuted_loss = normalize_trace(unpermuted_loss, alpha_step)
    unpermuted_max_loss = max(unpermuted_loss)

    counter = 0
    for m in permuted_max_losses:
        if m > unpermuted_max_loss:
            counter += 1

    return counter, len(permuted_max_losses)


def avg_loss_compare(results_path, unpermuted_loss, num_points, normalize=True, alpha_step=0.1):
    df = pd.read_csv(results_path)
    alphas = [round(alpha * alpha_step, 2) for alpha in range(int(1 / alpha_step + 1))]

    permuted_avg_losses = []

    for index, row in df.iterrows():
        row = row[-num_points:]
        if normalize:
            row = normalize_trace(row, alpha_step)
        permuted_avg_losses.append(sum(row) / len(row))

    if normalize:
        unpermuted_loss = normalize_trace(unpermuted_loss, alpha_step)
    unpermuted_avg_loss = sum(unpermuted_loss) / len(unpermuted_loss)

    counter = 0
    for m in permuted_avg_losses:
        if m > unpermuted_avg_loss:
            counter += 1

    return counter, len(permuted_avg_losses)


def avg_loss_ahmed(results_path, num_points=5, normalize=True, alphas=[0.0, 0.3, 0.5, 0.7, 1.0]):
    df = pd.read_csv(results_path)

    avg_losses = []

    for index, row in df.iterrows():
        row = row[-num_points:]
        if normalize:
            row = normalize_trace_2(row, alphas)
        avg_losses.append(sum(row) / len(row))

    return avg_losses


def compute_p_value(counter, total):
    return (total - counter - 1) / total
