import subprocess
import sys

from benchmark import benchmark_model
from cs336_basics.config import Config
from tabulate import tabulate

experiments = ["small", "medium", "large", "xl"] # "2.7B"]
base_cfg = "cs336_systems/base.yaml"


def get_specs(model_name):
    predefined_models = {
        "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
        "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
        "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
        "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
        "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    }
    if model_name not in predefined_models.keys():
        raise NotImplementedError(
            f"{model_name} is not a known model name. \
                                  Implemented models:\n{predefined_models.keys()}"
        )
    else:
        return predefined_models[model_name]


def run_benchmark(config_file, experiment_name):
    # modifications = [["--" + k, v] for k, v in get_specs(experiment_name).items()]
    # print(modifications)
    config = Config.from_yaml(config_file)
    for context_length in [128, 256, 512, 1024]:
        setattr(config, "context_length", context_length)
        for key, value in get_specs(experiment_name).items():
            setattr(config, key, value)
    # [cmd.extend(modification) for modification in modifications]

    print(f"Starting benchmark: {experiment_name}")
    print("-" * 50)

    result = benchmark_model(config, memory_profile=False, num_trials=10, warmup_steps=2, backward=True)
    print(f"Completed: {experiment_name}")

    # print(f"{result = }")
    return result


def print_results(table):
    #print(table)
    print(
        tabulate(
            table,
            headers=["Experiment_name", "Forward time", "Forward std", "backward time", "backward std"],
            tablefmt="grid",
        )
    )


def main():
    print(f"Running {len(experiments)} experiments")
    experimental_results = []

    for i, experiment_name in enumerate(experiments):
        print(f"\n[{i}/{len(experiments)}] Running experiment...")

        success = run_benchmark(base_cfg, experiment_name)

        if not success:
            print(f"\nStopping due to failed experiment: {experiment_name}")
            sys.exit()
        experimental_results.append([experiment_name, *success])

    print("\n All experiments completed.")
    print_results(experimental_results)


if __name__ == "__main__":
    main()
