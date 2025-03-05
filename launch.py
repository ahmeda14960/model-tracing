import yaml
from yaml import Loader

import subprocess
import argparse


parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument("--slurm", default="nlprun -g 1 -d a6000 -r 80G -a model-tracing", type=str)
parser.add_argument("--python", default="python experiment.py", type=str)
parser.add_argument("--models", default="config/llama_flat.yaml", type=str)
parser.add_argument("--save", default="/juice4/scr4/nlp/model-tracing", type=str)
parser.add_argument("--flat", default="all", type=str)

args = parser.parse_args()

model_paths = yaml.load(open(args.models, "r"), Loader=Loader)

subprocess.run(f"mkdir -p {args.save}/logs", shell=True)
subprocess.run(f"mkdir -p {args.save}/results", shell=True)

if args.flat == "all":
    for i in range(len(model_paths)):
        for j in range(i + 1, len(model_paths)):
            model_a = model_paths[i]
            model_b = model_paths[j]

            job_id = model_a.replace("/", "-") + "_AND_" + model_b.replace("/", "-")
            if "miqu" not in job_id:
                continue
            if job_id[0] == "-":
                job_id = job_id[1:]

            log_path = args.save + "/logs/" + job_id + ".out"
            results_path = args.save + "/results/" + job_id + ".p"

            job = (
                args.slurm + f" -o {log_path} -n {job_id}"
                f" '{args.python}"
                + f" --base_model_id {model_a} --ft_model_id {model_b} --save {results_path}'"
            )
            subprocess.run(job, shell=True)

elif args.flat == "split":
    base_models = model_paths["base_models"]
    ft_models = model_paths["ft_models"]

    for base_model in base_models:
        for ft_model in ft_models:
            job_id = base_model.replace("/", "-") + "_AND_" + ft_model.replace("/", "-")

            log_path = args.save + "/logs/" + job_id + ".out"
            results_path = args.save + "/results/" + job_id + ".p"

            job = (
                args.slurm + f" -o {log_path} -n {job_id}"
                f" '{args.python}"
                + f" --base_model_id {base_model} --ft_model_id {ft_model} --save {results_path}'"
            )
            subprocess.run(job, shell=True)

elif args.flat == "specified":
    models_1 = model_paths["models_1"]
    models_2 = model_paths["models_2"]

    names_1 = model_paths["names_1"]
    names_2 = model_paths["names_2"]

    for i in range(len(models_1)):
        model_a = models_1[i]
        model_b = models_2[i]

        name_a = names_1[i]
        name_b = names_2[i]

        job_id = name_a.replace("/", "-") + "_AND_" + name_b.replace("/", "-")

        log_path = args.save + "/logs/" + job_id + ".out"
        results_path = args.save + "/results/" + job_id + ".p"

        job = (
            args.slurm + f" -o {log_path} -n {job_id}"
            f" '{args.python}"
            + f" --base_model_id {model_a} --ft_model_id {model_b} --save {results_path}'"
        )
        subprocess.run(job, shell=True)
