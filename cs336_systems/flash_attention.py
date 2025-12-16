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
    
    # TODO: Implement the flash attention algorithm 
    #O_local = tl.zeros((Q_TILE_SIZE, D)) 
    #L_local = tl.zeros((Q_TILE_SIZE,))

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
        
        m_j = tl.maximum(m_curr, m_j)
        #tl.device_print("m_j-after", m_j)
        p_ij = tl.exp(s_ij - m_j[:, None])
        # running_numerator (this is NOT a normalized complete P_ij tile)  Bq x Bk

        l_j = tl.exp(m_curr - m_j) * l_j + tl.sum(p_ij, axis=-1)
        #print(f"-----{torch.exp(m_i - m_j) = } \n----{torch.exp(m_i - m_j).shape = } ")
        # running_denominator: add current sum, scale previous sum by e(m_{j-i} - m_{j})
        #
        o_j = tl.exp(m_curr - m_j)[:, None] * o_j + tl.dot(p_ij.to(v_j.dtype), v_j)
        #print(f"{l_i = } -- {o_i = }")
        # running output. diag() as we here have a matrix of values we need to rescale instead of a scalar for l_i
        # Bq, Bq @ Bq, d -> Bq, d
        m_curr = m_j # update m j-1
        # advance k, v - pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    o_j = (1 / l_j)[:, None] * o_j
    l_j = m_curr + tl.log(l_j)
    
    tl.store(O_block_ptr, o_j.to(tl.float32))
    tl.store(L_block_ptr, l_j)
    

class TritonFlashAttentionAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q_ptr, K_ptr, V_ptr, is_causal: bool = False):
        b, N_QUERIES, D = Q_ptr.shape
        b, N_KEYS, D = K_ptr.shape
        scale = 1 / math.sqrt(D)
        O_ptr = torch.empty((Q_ptr.shape), device=Q_ptr.device)
        L_ptr = torch.empty((b, N_QUERIES), device=Q_ptr.device)
        
        #BLOCK_SIZE = 1024
        stride_qb, stride_qq, stride_qd = (Q_ptr.stride(0), Q_ptr.stride(1), Q_ptr.stride(2))
        stride_kb, stride_kk, stride_kd = (K_ptr.stride(0), K_ptr.stride(1), K_ptr.stride(2))
        stride_vb, stride_vk, stride_vd = (V_ptr.stride(0), V_ptr.stride(1), V_ptr.stride(2))
        stride_ob, stride_oq, stride_od = (O_ptr.stride(0), O_ptr.stride(1), O_ptr.stride(2))
        stride_lb, stride_lq            = (L_ptr.stride(0), L_ptr.stride(1))
        

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
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

        ctx.save_for_backward(L_ptr, O_ptr)
        return O_ptr

    @staticmethod
    def backward(ctx, grad_out):
        return NotImplementedError


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
        d_fac = 1/math.sqrt(d)

        Bq = 16
        Bk = 16

        Tq = Nq // Bq
        Tk = Nk // Bk

        dQ = torch.zeros_like(Q)  # b, Nq, d
        dK = torch.zeros_like(K)  # b, Nk, d
        dV = torch.zeros_like(V)  # b, Nk, d

        # calculating each batch independently
        Q_b = torch.split(Q, split_size_or_sections=1, dim=0)
        K_b = torch.split(K, 1, dim=0)
        V_b = torch.split(V, 1, dim=0)
        L_b = torch.split(L, 1, dim=0)
        O_b = torch.split(O, 1, dim=0)

        for batch in range(b):
            # tiling Q, K and V matrices
            Q_tiled = torch.split(Q_b[batch], split_size_or_sections=Bq, dim=1)  # torch split splits by "split size"!
            K_tiled = torch.split(K_b[batch], Bk, dim=1)
            V_tiled = torch.split(V_b[batch], Bk, dim=1)
            L_tiled = torch.split(L_b[batch], Bk, dim=1)
            O_tiled = torch.split(O_b[batch], Bk, dim=1)
            # implementing flash attention algo
            for i in range(Tq): #Tq

                q_i = Q_tiled[i].squeeze(0)
                l_i = L_tiled[i].squeeze(0)
                o_i = O_tiled[i].squeeze(0)
                # print(f"{o_i.shape = }")
                for j in range(Tk): #Tk
                    k_j = K_tiled[j].squeeze(0)
                    v_j = V_tiled[j].squeeze(0)

                    # running softmax
                    s_ij = einsum(q_i, k_j, "Bq d, Bk d -> Bq Bk") * d_fac  # compute pre-softmax attention
                    print(f"{s_ij.shape = } -- {l_i.shape = }")
                    p_ij = torch.exp(s_ij - L)
                    

                    #dV[batch, Tq*i:Tq*(i+1), Tk*j:Tk*(j+1)] = torch.transpose(p_ij) @ dO
                

        dQ, dK = Q, K
        
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


