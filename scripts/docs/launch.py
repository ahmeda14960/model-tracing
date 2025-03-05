from yaml import safe_load
import subprocess
import argparse
import os


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return safe_load(file)


def main(args):
    # Load configurations
    config = load_yaml(args.config)

    # Create necessary directories
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(f"{config['save_dir']}/logs", exist_ok=True)
    os.makedirs(f"{config['save_dir']}/results", exist_ok=True)

    # Prepare base command
    base_cmd = f"{args.slurm} python {args.script}"

    # Launch jobs
    for dataset in config["datasets"]:
        for model_arch in config["model_architectures"]:
            job_id = f"{dataset}_{model_arch}"
            log_path = f"{config['save_dir']}/logs/{job_id}.out"
            results_path = f"{config['save_dir']}/results/{job_id}"

            cmd = (
                f"{base_cmd} "
                f"--model_arch {model_arch} "
                f"--test_name {dataset} "
                f"--save_dir {results_path}"
            )

            if "dolma" in dataset.lower():
                cmd += f" --json_dir {config['dolma_json_dir']}/{dataset}"
            elif "m2d2" in dataset.lower():
                cmd += f" --json_dir {config['m2d2_json_dir']}/{dataset}"

            full_cmd = f"{args.slurm} -o {log_path} -J {job_id} '{cmd}'"
            print(f"Launching job: {full_cmd}")
            subprocess.run(full_cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch interpolation experiments on SLURM")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument(
        "--slurm",
        default="srun --partition=your-partition --time=24:00:00 --mem=64G --gres=gpu:1",
        help="SLURM command",
    )
    parser.add_argument("--script", default="interpolation_script.py", help="Python script to run")
    args = parser.parse_args()
    main(args)
