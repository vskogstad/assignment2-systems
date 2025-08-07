from cs336_basics.train as train
from typing import Callable
import pandas as pd
import timeit
import torch

def benchmark_variations(base_config, parameter: dict[list] | None = None):
    if parameter:
        p_name, params = parameter
    else:
        p_name, params = "No parameter sweep", [1]

    for param in params:
        base_config = base_config.update(p_name, param)
        run(base_config)


def benchmark(description: str, run: Callable, num_trials: int, warmup_steps: int):
    


        # Do warmup
        for _ in range(warmup_steps):
            run()


        timings: list[float] = []
        for trial in range(num_trials):
            # Training loop
            start_time = timeit.default_timer()
            x, y = get_random_data()
            
            optimizer.zero_grad()
            y_pred = model(x)
            # loss = cross_entropy(pred=y_pred, targets=y)

            # Trying to search for nan-source using regular cross entropy
            loss = F.cross_entropy(rearrange(y_pred, "b s v -> (b s) v"), rearrange(y, "b s -> (b s)").long())
            middle = timeit.default_timer()

            loss.backward()"""
            

            
            run()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = timeit.default_timer()
            timings.append(start_time - end_time)
        
        return

def generate_data():
    pass


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()