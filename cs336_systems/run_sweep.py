import subprocess
import sys

experiments = ["small", "medium", "large", "xl", "2.7B"]
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


def run_benchmarks(config_file, experiment_name):
    modifications = [["--" + k, v] for k, v in get_specs(experiment_name).items()]
    print(modifications)
    cmd = ["uv", "run", "cs336_systems/benchmark.py", "--config", config_file]
    print(config_file)
    # [cmd.extend(modification) for modification in modifications]
    print(cmd)

    print(f"Starting benchmark: {experiment_name}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, check=True)
        print(f"Completed: {experiment_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed: {experiment_name} (exit code: {e.returncode})")
        return False

    print(f"{result = }")
    return True


def export_results(table):
    pass


def main():
    print(f"Running {len(experiments)} experiments")
    experimental_results = {}

    for i, experiment_name in enumerate(experiments):
        print(f"\n[{i}/{len(experiments)}] Running experiment...")

        success = run_benchmarks(base_cfg, experiment_name)

        if not success:
            print(f"\nStopping due to failed experiment: {experiment_name}")
            sys.exit()
        experimental_results[experiment_name] = success

    print("\n All experiments completed.")
    export_results(experiment_name)


if __name__ == "__main__":
    main()
