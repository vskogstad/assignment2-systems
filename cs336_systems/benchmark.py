import math
import timeit
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.cuda.nvtx as nvtx
from cs336_basics.config import Config, get_parser
from cs336_basics.model import Transformer
from cs336_basics.train_model import get_model, get_optimizer, softmax
from einops import einsum, rearrange


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


import cs336_basics.model

cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def benchmark_model(cfg: Config, memory_profile: bool, num_trials: int, warmup_steps: int, backward: bool):
    """
    Not sure how to do the warmup separately without a lot of code repetition.
    So I just do everything together and delete data for warmup steps afterwards.

    Using deafult loss now, should probably load it separately

    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cfg.dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32

    with nvtx.range("loading model"):
        model = (
            get_model(cfg, device)
            if (device == torch.device("cpu") or not cfg.compile_model)
            else torch.compile(get_model(cfg, device))
        )

    with nvtx.range("loading optimizer"):
        pass
        # optimizer = get_optimizer(cfg=cfg, model=model)

    timings: list[float] = []
    backward_timings: list[float] = []
    results = []

    for trial in range(warmup_steps + num_trials):
        # Training loop
        nvtx.range_push(f"step {trial} forward")
        if memory_profile and trial == warmup_steps:
            # Start recording memory history.
            torch.cuda.memory._record_memory_history(max_entries=1000000)
        start_time = timeit.default_timer()  # Do warmup with timing, discard data afterwards
        x = torch.randint(high=cfg.vocab_size, size=(cfg.d_model, cfg.batch_size), device=device)
        y = torch.randint(high=cfg.vocab_size, size=(cfg.d_model, cfg.batch_size), device=device)

        # forward pass
        y_pred = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_forward = timeit.default_timer()
        nvtx.range_pop()
        # loss = cross_entropy(pred=y_pred, targets=y)
        if backward:
            nvtx.range_push(f"step {trial} backward")
            # Trying to search for nan-source using regular cross entropy
            import torch.nn.functional as F

            loss = F.cross_entropy(rearrange(y_pred, "b s v -> (b s) v"), rearrange(y, "b s -> (b s)").long())
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_backward = timeit.default_timer()
            backward_time = end_backward - end_forward
            backward_timings.append(backward_time)
            nvtx.range_pop()
        forward_time = end_forward - start_time
        timings.append(forward_time)

    if memory_profile:
        # Save a pickle file to be loaded by PyTorch's online tool.
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)  # Stop recording history.
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


@torch.compile()
# @dynamo.disable
def scaled_dot_product_attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    seq_len = Q.shape[-2]
    # print(f"{Q.shape=}  {K.shape=} | {V.shape=}")
    # Q^T K / sqrt(d_k)
    attn = einsum(Q, K, "b ... sq d_k, b ... sk d_k -> b ... sq sk") / math.sqrt(d_k)
    # apply mask if included
    if mask is not None:
        m = mask.to(bool)
        attn = attn.masked_fill(~mask, float("-inf"))
    result = einsum(softmax(x=attn, dimension=-1), V, "b ... sq sk, b ... sk d_v -> b ... sq d_v")

    return result


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_dim"],
        x_vals=[128**i for i in range(2, 16)],
        line_arg="attention_function",
        line_vals=["Triton_torch_bw", "Tri-Dao", "Torch.compile"],
        line_names=["Triton fw, torch bw", "nn.scaled_dot..", "torch.compile()"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name="Attention",  # name for the plot. Used also as a file name for saving the plot.
        args={
            "batch_dim": 1,
            "head_dim": 64,
            "dtype": torch.dtype.bfloat16,
        },  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark_attention(batch_dim, seq_dim, head_dim, dtype, attention_function):
    """
    Based on the benchmarking sample from triton-tutorials:
    https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
    """
    from cs336_basics.model import scaled_dot_product_attention as torch_compile_attn
    from torch.nn.Functonal import scaled_dot_product_attention as tri_dao_attn

    Q = torch.randn(batch_dim, seq_dim, head_dim, dtype=dtype)
    K = torch.randn(batch_dim, seq_dim, head_dim, dtype=dtype)
    V = torch.randn(batch_dim, seq_dim, head_dim, dtype=dtype)

    mask = None
    if attention_function == "Triton_torch_bw":
        ms = triton.testing.do_bench(lambda: torch_compile_attn(Q, K, V, mask))
    if attention_function == "Tri-Dao":
        ms = triton.testing.do_bench(lambda: tri_dao_attn(Q, K, V, mask))
    if attention_function == "Torch.compile":
        ms = triton.testing.do_bench(lambda: torch_compile_attn(Q, K, V, mask))
    GBperSec = lambda ms: 2 * (Q.numel() + K.numel() + V.numel()) * Q.element_size() * 1e-9 / (ms * 1e-3)
    return GBperSec(ms)


benchmark_attention.run(show_plots=True, print_data=True)

if __name__ == "__main__":
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # load config
    config = Config.from_yaml(args.config)
    config.update_from_args(args)
    benchmark_model(cfg=config, loss_func=None, num_trials=10, warmup_steps=5, backward=True)
    benchmark_model(cfg=config, loss_func=None, num_trials=10, warmup_steps=5, backward=True)
