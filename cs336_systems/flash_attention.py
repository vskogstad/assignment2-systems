import math
import triton
import triton.language as tl
import torch

from einops import einsum, rearrange, reduce

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    
):


    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    # Set up K_block_ptr, V_block_ptr, O_block_ptr, L_block_ptr similarly    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape = (N_KEYS, D),
        strides = (stride_kk, stride_kd),
        offsets = (0, 0),
        block_shape = (K_TILE_SIZE, D),
        order=(1,0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape = (N_KEYS, D),
        strides = (stride_vk, stride_vd),
        offsets = (0, 0),
        block_shape = (K_TILE_SIZE, D),
        order=(1,0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES,),
        strides = (stride_lq,),
        offsets = (query_tile_index*Q_TILE_SIZE,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,),
    )
    

    m_curr = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32) 
    l_j = tl.full((Q_TILE_SIZE,), 1, dtype=tl.float32)
    o_j = tl.full((Q_TILE_SIZE, D), 1, dtype=tl.float32)
    q = tl.load(Q_block_ptr, boundary_check = (0,1), padding_option="zero")
    num_k_tiles = query_tile_index + 1 if is_causal else N_KEYS//K_TILE_SIZE
    
    for j in range(num_k_tiles):

        k_j = tl.load(K_block_ptr, boundary_check = (0,1), padding_option="zero")
        v_j = tl.load(V_block_ptr, boundary_check = (0,1), padding_option="zero")
        #tl.device_print("k_j", k_j)
        s_ij = tl.dot(q, tl.trans(k_j)) * scale
        if is_causal and j == query_tile_index:
            #tl.device_print("j", j)
            row_idx = tl.arange(0, Q_TILE_SIZE)
            col_idx = tl.arange(0, K_TILE_SIZE)
            causal_mask = row_idx[:, None] >= col_idx[None, :]
            s_ij = tl.where(causal_mask, s_ij, float("-inf"))
        
        m_j = tl.max(s_ij, axis=-1)
        m_j = tl.maximum(m_curr, m_j)  # find max values for all tiles up to and including j
        p_ij = tl.exp(s_ij - m_j[:, None]) # running_numerator (this is NOT a normalized complete P_ij tile)  Bq x Bk
        l_j = tl.exp(m_curr - m_j) * l_j + tl.sum(p_ij, axis=-1)
        # running_denominator: add current sum, scale previous sum by e(m_{j-i} - m_{j})
        o_j = tl.exp(m_curr - m_j)[:, None] * o_j + tl.dot(p_ij.to(v_j.dtype), v_j)  # running output. Bq, Bq @ Bq, d -> Bq, d
        m_curr = m_j # update m j-1
        
        # advance k, v - pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    o_j = (1 / l_j)[:, None] * o_j
    l_j = m_curr + tl.log(l_j)
    
    tl.store(O_block_ptr, o_j.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, l_j.to(L_block_ptr.type.element_ty))
    
@torch.compile()
def flash_bwd_pytorch(Q, K, V, L, O, dO, is_causal):
    b, Nq, d = Q.shape
    scale = 1 / math.sqrt(d)
    
    D = torch.sum(dO * O, dim=-1)

    S = einsum(Q, K, "b Bq d, b Bk d -> b Bq Bk") * scale
    if is_causal:
        mask = torch.triu(torch.ones(Nq,Nq, device = Q.device, dtype = torch.bool), diagonal=1)
        S=S.masked_fill(mask, float("-inf"))

    P = torch.exp(S - L.unsqueeze(-1)) # Don't need running softmax as we have stored L

    dV = P.transpose(-2, -1) @ dO
    dP = dO@ V.transpose(-2, -1)
    dS = P * (dP - D.unsqueeze(-1)) * scale
    dQ = dS @ K  # Must be atomic add in triton kernel for correctness.
    dK = dS.transpose(-2, -1) @ Q

    return dQ, dK, dV
    
def flash_bwd_pytorch_tiled(Q, K, V, L, O, dO, is_causal):
    b, Nq, d = Q.shape
    b, Nk, d = K.shape
    scale = 1/math.sqrt(d)

    Bq = 16
    Bk = 16

    Tq = Nq // Bq
    Tk = Nk // Bk

    dQ = torch.zeros_like(Q)  # b, Nq, d
    dK = torch.zeros_like(K)  # b, Nk, d
    dV = torch.zeros_like(V)  # b, Nk, d

    # Precomputing D
    D = torch.sum(dO * O, dim=-1)

    # calculating each batch independently
    Q_b = torch.split(Q, split_size_or_sections=1, dim=0)
    K_b = torch.split(K, 1, dim=0)
    V_b = torch.split(V, 1, dim=0)
    L_b = torch.split(L, 1, dim=0)
    O_b = torch.split(O, 1, dim=0)
    dO_b = torch.split(dO, 1, dim=0)
    D_b = torch.split(D, 1, dim=0)

    for batch in range(b):
        # tiling Q, K and V matrices
        Q_tiled = torch.split(Q_b[batch], split_size_or_sections=Bq, dim=1)  # torch split splits by "split size"!
        K_tiled = torch.split(K_b[batch], Bk, dim=1)
        V_tiled = torch.split(V_b[batch], Bk, dim=1)
        L_tiled = torch.split(L_b[batch], Bk, dim=1)
        O_tiled = torch.split(O_b[batch], Bq, dim=1)
        dO_tiled = torch.split(dO_b[batch], Bq, dim=1)
        D_tiled = torch.split(D_b[batch], Bq, dim=1)

        # implementing flash attention algo
        for j in range(Tk): #Tk
            K_j = K_tiled[j].squeeze(0)
            V_j = V_tiled[j].squeeze(0)
            dK_j = torch.zeros_like(K_j)
            dV_j = torch.zeros_like(V_j)
            
            
            for i in range(Tq): #Tq

                Q_i = Q_tiled[i].squeeze(0)
                L_i = L_tiled[i].squeeze(0)
                O_i = O_tiled[i].squeeze(0)
                dO_i = dO_tiled[i].squeeze(0)
                D_i = D_tiled[i].squeeze(0)

                
                S_ij = einsum(Q_i, K_j, "Bq d, Bk d -> Bq Bk") * scale  # compute pre-softmax attention
                P_ij = torch.exp(S_ij - L_i[:, None]) # Don't need running softmax as we have stored L

                dV_j += P_ij.T @ dO_i
                dP_ij = dO_i @ V_j.T
                dS_ij = P_ij * (dP_ij - D_i[:, None]) * scale
                dQ[batch, Bq*i:Bq*(i+1), :] += dS_ij @ K_j  # Must be atomic add in triton kernel for correctness.
                dK_j += dS_ij.T @ Q_i
                
            dK[batch, Bk*j:Bk*(j+1), :] = dK_j
            dV[batch, Bk*j:Bk*(j+1), :] = dV_j

    return dQ, dK, dV

class TritonFlashAttentionAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q_ptr, K_ptr, V_ptr, is_causal: bool = False):
        b, N_QUERIES, D = Q_ptr.shape
        b, N_KEYS, D = K_ptr.shape
        scale = 1 / math.sqrt(D)
        ctx.is_causal = is_causal
        O_ptr = torch.empty((Q_ptr.shape), device=Q_ptr.device, dtype=Q_ptr.dtype)
        L_ptr = torch.empty((b, N_QUERIES), device=Q_ptr.device, dtype=Q_ptr.dtype)
        
        #BLOCK_SIZE = 1024
        stride_qb, stride_qq, stride_qd = (Q_ptr.stride(0), Q_ptr.stride(1), Q_ptr.stride(2))
        stride_kb, stride_kk, stride_kd = (K_ptr.stride(0), K_ptr.stride(1), K_ptr.stride(2))
        stride_vb, stride_vk, stride_vd = (V_ptr.stride(0), V_ptr.stride(1), V_ptr.stride(2))
        stride_ob, stride_oq, stride_od = (O_ptr.stride(0), O_ptr.stride(1), O_ptr.stride(2))
        stride_lb, stride_lq            = (L_ptr.stride(0), L_ptr.stride(1))
        

        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64
        Tq = N_QUERIES // Q_TILE_SIZE

        grid = (Tq, b) # launch independent batches and Q-tiles across SM's
        flash_fwd_kernel[grid](
            Q_ptr, K_ptr, V_ptr,
            O_ptr, L_ptr,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            N_QUERIES, N_KEYS,
            scale,
            is_causal,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,)

        ctx.save_for_backward(Q_ptr, K_ptr, V_ptr, L_ptr, O_ptr)
        return O_ptr

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, L, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        dQ, dK, dV = flash_bwd_pytorch(Q, K, V, L, O, dO, is_causal)            
        
        return  dQ, dK, dV, None




class FlashAttentionAutogradFuncion(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        print(f"this is {Q.shape=}, {K.shape=} and {V.shape=}")
        ctx.is_causal = is_causal

        b, Nq, d = Q.shape
        b, Nk, d = K.shape
        d_fac = 1/math.sqrt(d)

        Bq = 16
        Bk = 16

        Tq = Nq // Bq
        Tk = Nk // Bk

        O_global = torch.zeros_like(Q)  # b, Nq, d
        L_global = torch.zeros((b, Nq))  # b, Nq

        # calculating each batch independently
        Q_b = torch.split(Q, split_size_or_sections=1, dim=0)
        K_b = torch.split(K, 1, dim=0)
        V_b = torch.split(V, 1, dim=0)

        for batch in range(b):
            # tiling Q, K and V matrices
            Q_tiled = torch.split(Q_b[batch], split_size_or_sections=Bq, dim=1)  # torch split splits by "split size"!
            K_tiled = torch.split(K_b[batch], Bk, dim=1)
            V_tiled = torch.split(V_b[batch], Bk, dim=1)

            # implementing flash attention algo
            for i in range(Tq): #Tq

                q_i = Q_tiled[i].squeeze(0)
                # print(f"{q_i.shape = }")
                m_i = torch.ones((Bq,)) * float("-inf")
                l_i = torch.zeros((Bq,))
                o_i = torch.zeros((Bq, d))
                # print(f"{o_i.shape = }")
                for j in range(Tk): #Tk
                    k_j = K_tiled[j].squeeze(0)
                    v_j = V_tiled[j].squeeze(0)

                    # running softmax
                    s_ij = einsum(q_i, k_j, "Bq d, Bk d -> Bq Bk") * d_fac  # compute pre-softmax attention
                    m_j = torch.max(s_ij, dim=-1).values  # max_value in each row of current tile
                    #print(f"{m_j = } -- {s_ij[:,-1] = }")
                    m_j = torch.max(m_i, m_j)  # update max-values for each row.
                    #print(f"{s_ij = } -- {m_j = }")
                    p_ij = torch.exp(s_ij - m_j.unsqueeze(1))
                    #print(f"{p_ij == torch.exp(s_ij - m_j)} ")
                    # running_numerator (this is NOT a normalized complete P_ij tile)  Bq x Bk
                    l_i = torch.exp(m_i - m_j) * l_i + torch.sum(p_ij, dim=-1)
                    #print(f"-----{torch.exp(m_i - m_j) = } \n----{torch.exp(m_i - m_j).shape = } ")
                    # running_denominator: add current sum, scale previous sum by e(m_{j-i} - m_{j})
                    o_i = torch.diag(torch.exp(m_i - m_j)) @ o_i + p_ij @ v_j
                    #print(f"{l_i = } -- {o_i = }")
                    # running output. diag() as we here have a matrix of values we need to rescale instead of a scalar for l_i
                    # Bq, Bq @ Bq, d -> Bq, d
                    m_i = m_j # update m j-1
                O_i = torch.diag(1 / l_i) @ o_i
                #print(p_ij)
                L_i = m_j + torch.log(l_i)  # TODO: Need to understand wtf this is.
                # write to out
                O_global[batch, i * Bq : (i + 1) * Bq, :] = O_i
                L_global[batch, i * Bq : (i + 1) * Bq] = L_i

        ctx.save_for_backward(Q, K, V, L_global, O_global)


        return O_global


    @staticmethod
    def backward(ctx, dO):
        Q, K, V, L, O = ctx.saved_tensors
        is_causal = ctx.is_causal

        b, Nq, d = Q.shape
        b, Nk, d = K.shape
        scale = 1/math.sqrt(d)

        Bq = 16
        Bk = 16

        Tq = Nq // Bq
        Tk = Nk // Bk

        dQ = torch.zeros_like(Q)  # b, Nq, d
        dK = torch.zeros_like(K)  # b, Nk, d
        dV = torch.zeros_like(V)  # b, Nk, d

        # Precomputing D
        D = torch.sum(dO * O, dim=-1)


        # calculating each batch independently
        Q_b = torch.split(Q, split_size_or_sections=1, dim=0)
        K_b = torch.split(K, 1, dim=0)
        V_b = torch.split(V, 1, dim=0)
        L_b = torch.split(L, 1, dim=0)
        O_b = torch.split(O, 1, dim=0)
        dO_b = torch.split(dO, 1, dim=0)
        D_b = torch.split(D, 1, dim=0)

        for batch in range(b):
            # tiling Q, K and V matrices
            Q_tiled = torch.split(Q_b[batch], split_size_or_sections=Bq, dim=1)  # torch split splits by "split size"!
            K_tiled = torch.split(K_b[batch], Bk, dim=1)
            V_tiled = torch.split(V_b[batch], Bk, dim=1)
            L_tiled = torch.split(L_b[batch], Bk, dim=1)
            O_tiled = torch.split(O_b[batch], Bq, dim=1)
            dO_tiled = torch.split(dO_b[batch], Bq, dim=1)
            D_tiled = torch.split(D_b[batch], Bq, dim=1)

            # implementing flash attention algo
            for j in range(Tk): #Tk
                K_j = K_tiled[j].squeeze(0)
                V_j = V_tiled[j].squeeze(0)
                dK_j = torch.zeros_like(K_j)
                dV_j = torch.zeros_like(V_j)
                
                
                for i in range(Tq): #Tq

                    Q_i = Q_tiled[i].squeeze(0)
                    L_i = L_tiled[i].squeeze(0)
                    O_i = O_tiled[i].squeeze(0)
                    dO_i = dO_tiled[i].squeeze(0)
                    D_i = D_tiled[i].squeeze(0)

                    
                    S_ij = einsum(Q_i, K_j, "Bq d, Bk d -> Bq Bk") * scale  # compute pre-softmax attention
                    P_ij = torch.exp(S_ij - L_i[:, None]) # Don't need running softmax as we have stored L

                    dV_j += P_ij.T @ dO_i
                    dP_ij = dO_i @ V_j.T
                    dS_ij = P_ij * (dP_ij - D_i[:, None]) * scale
                    dQ[batch, Bq*i:Bq*(i+1), :] += dS_ij @ K_j  # Must be atomic add in triton kernel for correctness.
                    dK_j += dS_ij.T @ Q_i
                    
                dK[batch, Bk*j:Bk*(j+1), :] = dK_j
                dV[batch, Bk*j:Bk*(j+1), :] = dV_j
                
        
        return  dQ, dK, dV, None



class AttentionAutogradFuncion(torch.autograd.Function):
    """Basic implementation to get the logic right"""

    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        b, s, d = Q.shape
        print(f"this is {Q.shape=}, {K.shape=} and {V.shape=}")
        print(torch.max(torch.tensor(((1, 2, 3), (4, 5, 6)))))
        S = einsum(Q, K, "b sq d, b sk d -> b sq sk") / math.sqrt(d)
        print(f"{S.shape = }")
        s_max = torch.max(S, dim=2, keepdim=True).values
        print(f"{s_max.shape = }")
        numerator = torch.exp(S - s_max)
        denominator = torch.sum(numerator, dim=2, keepdim=True)
        print(f"{numerator.shape = } -- {denominator.shape = }")
        P = numerator / denominator
        print(f"{P.shape = }")
        O = einsum(P, V, "b sq sk, b sk d -> b sq d")
        L = torch.log(reduce(torch.exp(S), "b sq sk -> b sq", "sum"))
        print(f"{O.shape = } -- {L.shape = }")
        ctx.save_for_backward(L, O)
        return O

    @staticmethod
    def backward(q, k, v, o, do):
        return NotImplementedError




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
        x_vals=[2**i for i in range(7, 12)], #16
        line_arg="attention_function",
        line_vals=["Triton_torch_bw", "FA_torch", "Torch.compile"],
        line_names=["Triton fw, torch bw", "nn.scaled_dot..", "torch.compile()"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name="Attention",  # name for the plot. Used also as a file name for saving the plot.
        args={
            "batch_dim": 128*16,
            "head_dim": 64,
            "dtype": torch.bfloat16,
            "is_causal": True,
            "direction": "both",
        },  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark_attention(batch_dim, seq_dim, head_dim, dtype, attention_function, is_causal, direction):
    """
    Based on the benchmarking sample from triton-tutorials:
    https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
    """
    from cs336_basics.model import scaled_dot_product_attention as compiled_sdpa
    from torch.nn.functional import scaled_dot_product_attention as nn_sdpa

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    Q = torch.randn(batch_dim, seq_dim, head_dim, device=DEVICE, dtype=dtype, requires_grad=True)
    K = torch.randn(batch_dim, seq_dim, head_dim, device=DEVICE, dtype=dtype, requires_grad=True)
    V = torch.randn(batch_dim, seq_dim, head_dim, device=DEVICE, dtype=dtype, requires_grad=True)

    
    #mask = torch.tril(torch.ones((seq_dim, seq_dim), device=DEVICE, dtype=dtype))
    
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if attention_function == "Triton_torch_bw":
        #ms = triton.testing.do_bench(lambda: TritonFlashAttentionAutogradFunction.apply(Q, K, V, is_causal))
        ms = test_wrapper(lambda: TritonFlashAttentionAutogradFunction.apply(Q, K, V, is_causal), direction, Q, K, V)
    if attention_function == "FA_torch":
        Q = Q.unsqueeze(1)  # Add head dimension: (batch, 1, seq, d)
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)
        ms = test_wrapper(lambda: nn_sdpa(Q, K, V, is_causal=is_causal), direction, Q, K, V)
        Q = Q.squeeze(1)  # Remove head dimension: (batch, seq, d)
        K = K.squeeze(1)
        V = V.squeeze(1)
    if attention_function == "Torch.compile":
        ms = test_wrapper(lambda: compiled_sdpa(Q, K, V, mask=None), direction, Q, K, V)
    
    #GBperSec = lambda ms: 2 * (Q.numel() + K.numel() + V.numel()) * Q.element_size() * 1e-9 / (ms * 1e-3)
    causal_scale = 1 / (1 + is_causal)
    if direction == "both":
        direction_scale = 3
    elif direction == "backward":
        direction_scale = 2
    else: 
        direction_scale = 1
    #print(causal_scale)
    tflops = dtype.itemsize * batch_dim * seq_dim * seq_dim * head_dim * causal_scale * direction_scale * 1e-12 / (ms * 1e-3) 
    
    return tflops

def test_wrapper(func, direction, Q, K, V):
    if direction == "forward":
        ms_fwd = triton.testing.do_bench(func)
        return ms_fwd
    elif direction == "backward":
        dO = torch.randn(Q.shape, device=Q.device, dtype=Q.dtype)
        ms_fwd = triton.testing.do_bench(func)
        ms_fwd_bwd = triton.testing.do_bench(lambda: func().backward(dO))
        return ms_fwd_bwd - ms_fwd  #Estimate backward time
    elif direction == "both":
        # Forward + backward
        dO = torch.randn(Q.shape, device=Q.device, dtype=Q.dtype)
        def fwd_bwd():
            Q.grad, K.grad, V.grad = None, None, None
            out = func()
            out.backward(dO)
        ms_both = triton.testing.do_bench(fwd_bwd)
        return ms_both
    else:
        raise NotImplementedError('Function can only be run with direction = "forward", "backward" or "both".')


def test_timing_flash_forward_backward():
    n_heads = 16
    d_head = 64
    sequence_length = 16384
    q, k, v = torch.randn(3, n_heads, sequence_length, d_head, device='cuda', dtype=torch.bfloat16, requires_grad=True
    )

    flash = torch.compile(TritonFlashAttentionAutogradFunction.apply)

    def flash_forward_backward():
        o = flash(q, k, v, True)
        loss = o.sum()
        loss.backward()

    results = triton.testing.do_bench(flash_forward_backward, rep=1000, warmup=100) #rep=10000, warmup=1000)
    print(results)

if __name__ == "__main__":
    #benchmark_attention.run(show_plots=True, print_data=True)
    
    test_timing_flash_forward_backward()



