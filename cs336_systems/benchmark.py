from cs336_basics.train as train
from typing import Callable
import pandas as pd
import timeit
import torch
import numpy as np

def benchmark_variations(base_config, parameter: dict[list] | None = None):
    if parameter:
        p_name, params = parameter
    else:
        p_name, params = "No parameter sweep", [1]

    for param in params:
        base_config = base_config.update(p_name, param)
        run(base_config)


def run_transformer(dim=128, num_layers=16, batch_size=128, num_steps=5):
    pass


def get_specs(model_name):

    predefined_models = {
    'small': {
        'd_model': 768,
        'd_ff': 3072,
        'num_layers': 12,
        'num_heads': 12
    },
    'medium': {
        'd_model': 1024,
        'd_ff': 4096,
        'num_layers': 24,
        'num_heads': 16
    },
    'large': {
        'd_model': 1280,
        'd_ff': 5120,
        'num_layers': 36,
        'num_heads': 20
    },
    'xl': {
        'd_model': 1600,
        'd_ff': 6400,
        'num_layers': 48,
        'num_heads': 25
    },
    '2.7B': {
        'd_model': 2560,
        'd_ff': 10240,
        'num_layers': 32,
        'num_heads': 32
    }
}
    if model_name not in predefined_models.keys():
        raise NotImplementedError(f"{model_name} is not a known model name. \
                                  Implemented models:\n{predefined_models.keys()}")
    else:
        return predefined_models[model_name]

def benchmark_model(model, loss_func, num_trials: int, warmup_steps: int, backward: bool):
        """
        Not sure how to do the warmup separately without a lot of code repetition.
        So I just do everything together and delete data for warmup steps afterwards.

        Using deafult loss now, should probably load it separately

        """
    


        

        timings: list[float] = []
        backward_timings: list[float] = []

        for trial in range(warmup_steps + num_trials): # Do warmup with timing, discard data afterwards
            # Training loop
            start_time = timeit.default_timer()
            x = torch.randn(batch_size, dim, device)
            print(x) 
            
            # forward pass
            y_pred = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_forward = timeit.default_timer()
            # loss = cross_entropy(pred=y_pred, targets=y)
            if backward:
                # Trying to search for nan-source using regular cross entropy
                
                loss = F.cross_entropy(rearrange(y_pred, "b s v -> (b s) v"), rearrange(y, "b s -> (b s)").long())
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()    
                end_backward = timeit.default_timer()
                backward_time = end_backward - end_forward
                backward_timings.append(backward_time)

            forward_time = start_time - end_forward
            timings.append(forward_time)
        print("Before " + timings)
        timings = timings[warmup_steps:]
        print(timings)
        print(f"The forward pass takes an average of: {np.mean(timings)} with standard deviation of: {np.std(timings)}")
        if backward:
            backward_timings = backward_timings[warmup_steps:]
            print(f"The backward pass takes an average of: {backward_timings.mean()} with standard deviation of: {np.std(backward_timings)}")
        return


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()