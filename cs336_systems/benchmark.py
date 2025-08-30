import timeit
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.cuda.nvtx as nvtx
from cs336_basics.config import Config, get_parser
from cs336_basics.model import Transformer, scaled_dot_product_attention
#cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

from cs336_basics.train_model import get_model
from einops import rearrange


@torch.compile()
def annotated_scaled_dot_product_attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    seq_len = Q.shape[-2]
    # print(f"{Q.shape=}  {K.shape=} | {V.shape=}")
    # Q^T K / sqrt(d_k)
    with nvtx.range("computing attention scores"):
        attn = einsum(Q, K, "b ... sq d_k, b ... sk d_k -> b ... sq sk") / math.sqrt(d_k)
    
    # apply mask if included
    if mask is not None:
        attn = attn.masked_fill(mask[:seq_len, :seq_len] == False, float("-inf"))
    with nvtx.range("computing softmax"):    
        result = einsum(softmax(x=attn, dimension=-1), V, "b ... sq sk, b ... sk d_v -> b ... sq d_v")

    return result
scaled_dot_product_attention = annotated_scaled_dot_product_attention

def benchmark_model(cfg: Config, memory_profile: bool, num_trials: int, warmup_steps: int, backward: bool):
    """
    Not sure how to do the warmup separately without a lot of code repetition.
    So I just do everything together and delete data for warmup steps afterwards.

    Using deafult loss now, should probably load it separately

    """
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cfg.dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32

    model = get_model(cfg, device) if device == torch.device("cpu") else torch.compile(get_model(cfg, device))

    timings: list[float] = []
    backward_timings: list[float] = []
    results = []

    for trial in range(warmup_steps + num_trials): 
        # Training loop
        if memory_profile and trial == warmup_steps:
            # Start recording memory history.
            torch.cuda.memory._record_memory_history(max_entries=1000000)
        start_time = timeit.default_timer()   # Do warmup with timing, discard data afterwards
        x = torch.randint(high=cfg.vocab_size, size=(cfg.d_model, cfg.batch_size), device=device)
        y = torch.randint(high=cfg.vocab_size, size=(cfg.d_model, cfg.batch_size), device=device)

        # forward pass
        y_pred = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_forward = timeit.default_timer()
        # loss = cross_entropy(pred=y_pred, targets=y)
        if backward:
            # Trying to search for nan-source using regular cross entropy
            import torch.nn.functional as F

            loss = F.cross_entropy(rearrange(y_pred, "b s v -> (b s) v"), rearrange(y, "b s -> (b s)").long())
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_backward = timeit.default_timer()
            backward_time = end_backward - end_forward
            backward_timings.append(backward_time)

        forward_time = end_forward - start_time
        timings.append(forward_time)

    if memory_profile:
        # Save a pickle file to be loaded by PyTorch's online tool.
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None) # Stop recording history.
    # Remove timings for warmup-steps
    timings = timings[warmup_steps:]
    results.extend([np.mean(timings), np.std(timings)])

    print(
        f"The forward pass takes an average of: {np.mean(timings):.5f} with standard deviation of: {np.std(timings):.5f}"
    )
    if backward:
        backward_timings = backward_timings[warmup_steps:]
        print(
            f"The backward pass takes an average of: {np.mean(backward_timings):.5f} with standard deviation of: {np.std(backward_timings):.5f}"
        )
        results.extend([np.mean(backward_timings), np.std(backward_timings)])
    return results


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # load config
    config = Config.from_yaml(args.config)
    config.update_from_args(args)
    benchmark_model(cfg=config, loss_func=None, num_trials=10, warmup_steps=5, backward=True)
